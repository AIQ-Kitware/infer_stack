"""``runtime.shm_size``: /dev/shm for a vLLM engine on Compose (queue item 8b).

Opt-in: an endpoint that does not set it renders exactly as before, so no
running engine is recreated by the upgrade. A KubeAI pod already mounts a
memory-backed /dev/shm, so the key is Compose's alone.
"""

import pytest

from infer_stack.leasing.catalog import Catalog, CatalogError
from infer_stack.leasing.compose import _SHM_WARNED, _vllm_service
from infer_stack.leasing.models import Deployment, DeploymentState

IMAGES = {'vllm': 'vllm/vllm-openai:latest'}


def _deployment(runtime, gid='dep'):
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {'engine': 'vllm', 'hf_model_id': 'org/m', 'served_model_name': 'm',
         'runtime': runtime, 'reclaim': 'stop'},
        {'m': {'served_model_name': 'm', 'protocol': 'chat'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )


def test_shm_size_renders_when_set_and_is_absent_otherwise():
    svc = _vllm_service(_deployment({'tensor_parallel_size': 2, 'shm_size': '16g'}),
                        [0, 1], None, IMAGES, {})
    assert svc['shm_size'] == '16g'
    plain = _vllm_service(_deployment({}), [0], None, IMAGES, {})
    assert 'shm_size' not in plain           # today's catalogs render unchanged
    assert not any('shm' in arg for arg in plain['command'])


def test_a_parallel_engine_without_it_is_warned_once(monkeypatch):
    from infer_stack import _log

    seen = []
    monkeypatch.setattr(_log.logger, 'warning', lambda msg, *a: seen.append(msg.format(*a)))
    _SHM_WARNED.discard('tp2')
    for _ in range(3):
        _vllm_service(_deployment({'tensor_parallel_size': 2}, 'tp2'), [0, 1], None, IMAGES, {})
    _vllm_service(_deployment({}, 'tp1'), [0], None, IMAGES, {})
    assert len(seen) == 1 and 'runtime.shm_size' in seen[0] and '2 GPUs' in seen[0]


def test_the_catalog_checks_shm_size():
    def catalog(shm):
        return {'models': {'m': {'source': 'hf://org/m'}},
                'endpoints': {'e': {'engine': 'vllm', 'model': 'm',
                                    'runtime': {'shm_size': shm}}}}

    Catalog.from_dict(catalog('16g'))
    with pytest.raises(CatalogError, match='shm_size must be a size'):
        Catalog.from_dict(catalog('plenty'))
