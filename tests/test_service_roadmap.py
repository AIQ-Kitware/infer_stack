"""User contracts at the seams between placement, serving policy and onboarding."""
import copy
import json
from types import SimpleNamespace

import pytest
from test_kube_worker import Worker, run_test
from test_leasing_kubeai import vllm

from infer_stack.backends.kubeai import node_gpus, render_models
from infer_stack.kube.context import command, remember
from infer_stack.kube.preflight import storage_facts
from infer_stack.leasing.catalog import Catalog, CatalogError
from infer_stack.leasing.cluster_placement import narrow
from infer_stack.leasing.serving_policy import policy


def catalog(placement=None, runtime=None):
    return Catalog.from_dict({'models': {'m': {'source': 'hf://org/m'}},
                              'endpoints': {'q': {'engine': 'vllm', 'model': 'm',
                                                  'placement': placement or {}, 'runtime': runtime or {}}}})


@pytest.mark.parametrize('runtime', [{'min_replicas': -1}, {'max_replicas': 0},
                                    {'min_replicas': 2}, {'min_replicas': False},
                                    {'target_requests': 0}, {'scale_down_delay_seconds': -1},
                                    {'load_balancing': 'random'}, {'max_replicas': 1.5}])
def test_bad_policy_refused_before_leasing(runtime):
    with pytest.raises(ValueError):
        policy(runtime)
    with pytest.raises(CatalogError):
        catalog(runtime=runtime).resolve_requests(['q'])
    rendered = render_models([vllm('g', **runtime)], namespace='kubeai', default_resource_profile='gpu')
    assert rendered.unrenderable == {'g'} and not rendered.docs


def test_zero_and_autoscaling_are_rendered_without_coercion():
    g = vllm('g', min_replicas=0, max_replicas=4, target_requests=8,
             scale_down_delay_seconds=0, load_balancing='PrefixHash')
    spec = render_models([g], namespace='kubeai', default_resource_profile='gpu').docs[0]['spec']
    assert (spec['minReplicas'], spec['maxReplicas'], spec['targetRequests'], spec['scaleDownDelaySeconds']) == (0, 4, 8, 0)
    assert spec['loadBalancing'] == {'strategy': 'PrefixHash'}
    legacy = catalog().resolve_requests(['q'])[0]
    explicit_default = catalog(runtime={'min_replicas': 1, 'max_replicas': 1}).resolve_requests(['q'])[0]
    on_demand = catalog(runtime={'min_replicas': 0}).resolve_requests(['q'])[0]
    assert legacy.compat_key == explicit_default.compat_key != on_demand.compat_key


def test_narrowing_is_conjunctive_and_does_not_edit_catalog():
    req = catalog({'pool': 'a6000'}).resolve_requests(['q'])[0]
    narrowed = narrow([req], node='worker.lab')[0]
    assert narrowed.spec['placement'] == {'pool': 'a6000', 'node': 'worker.lab'}
    assert req.spec['placement'] == {'pool': 'a6000'}
    assert req.compat_key != narrowed.compat_key
    with pytest.raises(ValueError, match='conflicts with approved'):
        narrow([req], pool='other')
    from infer_stack.leasing.profile import CatalogUnion
    union = CatalogUnion.from_sources([catalog({'pool': 'a6000'}).source])
    assert union.request_matches(narrowed)
    forged = copy.deepcopy(narrowed)
    forged.spec['runtime']['image'] = 'unapproved'
    assert not union.request_matches(forged)


def test_admin_classification_profiles_enforce_node_and_pool():
    from infer_stack.kube.classification import labels, profiles
    w = Worker(count=4)
    w.nodes[1]['metadata']['labels'].update(labels(pool='a6000', machine_class='4xa6000'))
    generated = profiles(w.nodes)
    assert any(p['nodeSelector'] == {'kubernetes.io/hostname': 'namek', 'infer-stack.ai/pool': 'a6000'} for p in generated.values())
    assert any(p['nodeSelector'].get('infer-stack.ai/pool') == 'a6000'
               and 'kubernetes.io/hostname' not in p['nodeSelector'] for p in generated.values())


def cluster():
    w = Worker(name='worker.lab', count=4)
    w.nodes[1]['metadata']['labels']['infer-stack.ai/pool'] = 'a6000'
    return node_gpus(w.nodes)


def test_profile_must_enforce_policy_not_just_happen_to_match_today():
    g = vllm('g', profile=None)
    g.spec['placement'] = {'node': 'worker.lab', 'pool': 'a6000'}
    profiles = {'unsafe': {'nodeSelector': {'nvidia.com/gpu.product': 'NVIDIA-RTX-3090'}}}
    out = render_models([g], namespace='kubeai', default_resource_profile='unsafe', profiles=profiles, nodes=cluster())
    assert out.unrenderable == {'g'}
    profiles['safe'] = {'nodeSelector': {'kubernetes.io/hostname': 'worker.lab', 'infer-stack.ai/pool': 'a6000'}}
    profiles['safe'].update(requests={'nvidia.com/gpu': '1'}, limits={'nvidia.com/gpu': '1'})
    out = render_models([g], namespace='kubeai', default_resource_profile='unsafe', profiles=profiles, nodes=cluster())
    assert not out.errors
    assert out.docs[0]['spec']['resourceProfile'] == 'safe:1'


def test_dynamic_alias_cannot_mix_constrained_deployments():
    a, b = vllm('a'), vllm('b')
    for g in [a, b]:
        g.served = {'q': {'protocol': 'chat'}}
    b.spec['placement'] = {'pool': 'a6000'}
    b.spec['runtime'].pop('resource_profile')
    profiles = {'pool': {'nodeSelector': {'infer-stack.ai/pool': 'a6000'}}}
    profiles['pool'].update(requests={'nvidia.com/gpu': '1'}, limits={'nvidia.com/gpu': '1'})
    out = render_models([a, b], namespace='kubeai', default_resource_profile='gpu',
                        unique_names=True, profiles=profiles, nodes=cluster())
    assert out.unrenderable == {'b'}
    assert 'scoped inference routing' in out.errors[0]


def test_saved_context_applies_to_helm_and_kubectl_but_explicit_wins(tmp_path):
    remember(tmp_path / 'private.yaml')
    for tool in ['kubectl', 'helm']:
        assert command([tool, 'version']) == [tool, '--kubeconfig', str(tmp_path / 'private.yaml'), 'version']
        explicit = [tool, '--kubeconfig=other', 'version']
        assert command(explicit) == explicit
    assert command(['docker', 'ps']) == ['docker', 'ps']


def test_storage_preflight_keeps_thresholds_and_reclaim_distinct(tmp_path):
    config = tmp_path / 'kubelet.yaml'
    config.write_text('evictionHard: {imagefs.available: 5%}\nevictionMinimumReclaim: {imagefs.available: 10%}\n')
    stat = SimpleNamespace(f_blocks=100, f_frsize=1024**3, f_bavail=25, f_favail=30, f_files=100)
    facts = storage_facts(tmp_path / 'uncreated' / 'k3s', image_gib=14,
                          kubelet_config=config, statvfs=lambda p: stat)
    row = facts['thresholds']['imagefs.available']
    assert row['hard_bytes'] == 5 * 1024**3
    assert row['minimum_reclaim_bytes'] == 10 * 1024**3
    assert row['headroom_after_image_bytes'] == -4 * 1024**3
    assert facts['warnings'] and not (tmp_path / 'uncreated').exists()


def test_multigpu_acceptance_checks_serving_devices_and_tp():
    w = Worker(count=4)
    w.model_gpu_count = 4
    original = w.manager.run
    captured = []
    def run(args, **kw):
        if 'exec' in args:
            return original(['nvidia-smi'])
        if args[:2] == ['kubectl', 'create']:
            doc = json.loads(kw['input_text'])
            if doc['kind'] == 'Model':
                captured.append(copy.deepcopy(doc))
        return original(args, **kw)
    w.manager.run = run
    result = run_test(w, serve_gpus=4)
    assert result['generation_verified'] and len(result['serving_devices']) == 4
    assert captured[0]['spec']['resourceProfile'].endswith(':4')
    assert '--tensor-parallel-size=4' in captured[0]['spec']['args']
    assert not w.resources
