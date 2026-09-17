"""Planner admission mode (plan step P3, tests 1-7): demand, not age, decides."""

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.leasing.placement import plan_placement


def dep(gid, *, tp=1, t=0.0, state=DeploymentState.LIVE, **runtime):
    return Deployment(gid, 'ck', 'vllm', 'shared-compatible', {},
                      {'engine': 'vllm', 'runtime': {'tensor_parallel_size': tp, **runtime}},
                      {}, state, t, t)


INV = simulate_inventory('2x80')


def test_incident_old_idle_resident_yields_to_new_live_demand():
    idle = dep('old-idle', tp=2, t=0.0, state=DeploymentState.IDLE)
    live = dep('new-live', tp=2, t=5.0)
    # Today: creation order lets the idle resident keep both GPUs.
    legacy = plan_placement([idle, live], INV, pinned={'old-idle': [0, 1]})
    assert 'new-live' not in legacy.assignments
    plan = plan_placement([idle, live], INV, required_ids={'new-live'},
                          optional_hints={'old-idle': [0, 1]})
    assert plan.assignments == {'new-live': [0, 1]}
    assert plan.displaced == ['old-idle'] and not plan.errors


def test_pinned_idle_resident_loses_its_gpu_to_live_demand():
    idle = dep('idle', t=0.0, state=DeploymentState.IDLE)
    a, b = dep('a', t=1.0), dep('b', t=2.0)
    plan = plan_placement([idle, a, b], INV, required_ids={'a', 'b'},
                          optional_hints={'idle': [0]})
    assert sorted(plan.assignments) == ['a', 'b'] and plan.displaced == ['idle']


def test_non_resident_idle_deployment_is_absent_from_the_plan():
    idle = dep('idle', state=DeploymentState.IDLE)
    plan = plan_placement([idle, dep('a')], INV, required_ids={'a'}, optional_hints={})
    assert plan.assignments == {'a': [0]}
    assert 'idle' not in plan.displaced and not plan.errors


def test_optional_resident_keeps_free_gpus():
    idle = dep('idle', state=DeploymentState.IDLE)
    plan = plan_placement([idle, dep('a', t=1.0)], INV, required_ids={'a'},
                          optional_hints={'idle': [1]})
    assert plan.assignments == {'a': [0], 'idle': [1]} and not plan.displaced


def test_reused_deployment_adopts_its_resident_gpus_as_hard():
    plan = plan_placement([dep('reused'), dep('new', t=1.0)], INV,
                          hard={'reused': [1]}, required_ids={'new'})
    assert plan.assignments == {'reused': [1], 'new': [0]}


def test_hard_allocation_outside_allowed_gpus_stays_valid():
    plan = plan_placement([dep('theirs'), dep('mine', t=1.0)], INV,
                          allowed_gpus=[0], hard={'theirs': [1]}, required_ids={'mine'})
    assert plan.assignments == {'theirs': [1], 'mine': [0]}


def test_invalid_hard_allocation_is_degraded_not_replaced():
    plan = plan_placement([dep('gone')], INV, hard={'gone': [7]})
    assert plan.degraded == ['gone'] and 'gone' not in plan.assignments
    clash = plan_placement([dep('a'), dep('b', t=1.0)], INV, hard={'a': [0], 'b': [0]})
    assert clash.assignments == {'a': [0]} and clash.degraded == ['b']


def test_omitting_the_keywords_is_identical_to_today():
    deps = [dep('a', tp=2, t=0.0), dep('b', t=1.0, state=DeploymentState.IDLE)]
    base = plan_placement(deps, INV, pinned={'b': [0]})
    again = plan_placement(deps, INV, pinned={'b': [0]}, required_ids=None,
                           hard=None, optional_hints=None)
    assert base == again and base.degraded == [] and base.displaced == []
