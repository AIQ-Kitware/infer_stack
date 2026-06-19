"""Tests for single-host GPU placement over the union of deployments."""

from __future__ import annotations

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import plan_placement
from infer_stack.leasing.models import Deployment, DeploymentState


def vllm(gid, *, tp=1, dp=1, gpu_indices=None, t=0.0):
    runtime = {'tensor_parallel_size': tp, 'data_parallel_size': dp}
    if gpu_indices is not None:
        runtime['gpu_indices'] = gpu_indices
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {'engine': 'vllm', 'runtime': runtime}, {}, DeploymentState.LIVE, t, t,
    )


def ollama(gid, *, gpu_indices=(), t=0.0):
    return Deployment(
        gid, 'ck-' + gid, 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'gpu_indices': list(gpu_indices)}, {},
        DeploymentState.LIVE, t, t,
    )


def inv(spec='4x80', *, display=()):
    inventory = simulate_inventory(spec)
    for i in display:
        inventory['gpus'][i]['display_active'] = True
    return inventory


def test_single_vllm_gets_gpu0():
    plan = plan_placement([vllm('a')], inv())
    assert plan.ok
    assert plan.assignments == {'a': [0]}


def test_two_vllm_pack_sequentially():
    plan = plan_placement([vllm('a', t=0), vllm('b', t=1)], inv())
    assert plan.assignments == {'a': [0], 'b': [1]}


def test_tensor_parallel_spans_gpus():
    plan = plan_placement([vllm('big', tp=2, t=0), vllm('s', t=1)], inv())
    assert plan.assignments == {'big': [0, 1], 's': [2]}


def test_data_parallel_multiplies_count():
    plan = plan_placement([vllm('a', tp=2, dp=2)], inv())
    assert plan.assignments == {'a': [0, 1, 2, 3]}


def test_insufficient_gpus_errors():
    plan = plan_placement([vllm('a', tp=2), vllm('b', tp=2), vllm('c')], inv('2x80'))
    assert not plan.ok
    assert plan.assignments['a'] == [0, 1]
    assert any('c' in e for e in plan.errors)


def test_allowed_gpus_preserves_real_indices():
    plan = plan_placement([vllm('a'), vllm('b', t=1)], inv(), allowed_gpus=[2, 3])
    assert plan.assignments == {'a': [2], 'b': [3]}


def test_reserved_gpus_are_avoided():
    plan = plan_placement([vllm('a'), vllm('b', t=1)], inv(), reserved=[0])
    assert plan.assignments == {'a': [1], 'b': [2]}


def test_display_gpu_used_by_default():
    # default is to use every GPU (so a single-GPU/display host works)
    plan = plan_placement([vllm('a')], inv(display=[0]))
    assert plan.assignments == {'a': [0]}


def test_display_gpu_skipped_when_opted_in():
    plan = plan_placement([vllm('a')], inv(display=[0]), skip_display=True)
    assert plan.assignments == {'a': [1]}


def test_ollama_explicit_indices_claimed_and_avoided():
    plan = plan_placement(
        [ollama('daemon', gpu_indices=[0], t=0), vllm('v', t=1)], inv()
    )
    assert plan.assignments['daemon'] == [0]
    assert plan.assignments['v'] == [1]   # first-fit skips the claimed GPU


def test_ollama_cpu_gets_no_gpus():
    plan = plan_placement([ollama('cpu')], inv())
    assert plan.assignments == {'cpu': []}


def test_explicit_conflict_errors():
    plan = plan_placement(
        [ollama('d1', gpu_indices=[0], t=0), ollama('d2', gpu_indices=[0], t=1)],
        inv(),
    )
    assert not plan.ok
    assert any('already in use' in e for e in plan.errors)


def test_explicit_out_of_pool_errors():
    plan = plan_placement([ollama('d', gpu_indices=[7])], inv('4x80'))
    assert not plan.ok
    assert any('not in pool' in e for e in plan.errors)


def test_pinned_deployments_stay_put_when_new_arrives():
    # 'a' already runs on [0,1]; adding 'b' must not reshuffle 'a'
    plan = plan_placement(
        [vllm('a', tp=2, t=0), vllm('b', t=1)], inv(), pinned={'a': [0, 1]}
    )
    assert plan.assignments == {'a': [0, 1], 'b': [2]}


def test_invalid_pin_is_replaced():
    # pinned GPU 9 doesn't exist -> 'a' is re-placed by first-fit
    plan = plan_placement([vllm('a')], inv('4x80'), pinned={'a': [9]})
    assert plan.assignments == {'a': [0]}


def test_placement_is_deterministic_by_created_at():
    deployments = [vllm('z', t=1), vllm('a', t=0)]
    plan = plan_placement(deployments, inv())
    # earlier created_at ('a', t=0) takes gpu 0 regardless of list order
    assert plan.assignments == {'a': [0], 'z': [1]}
