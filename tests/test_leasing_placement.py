"""Tests for single-host GPU placement over the union of deployments."""

from __future__ import annotations

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import plan_placement
from infer_stack.leasing.models import Deployment, DeploymentState


def vllm(gid, *, tp=1, pp=1, dp=1, gpu_indices=None, vram=None, floor=None,
         t=0.0):
    runtime = {'tensor_parallel_size': tp,
               'pipeline_parallel_size': pp,
               'data_parallel_size': dp}
    if gpu_indices is not None:
        runtime['gpu_indices'] = gpu_indices
    spec = {'engine': 'vllm', 'runtime': runtime}
    placement = {}
    if vram is not None:
        placement['min_vram_gib'] = vram
    if floor is not None:
        placement['floor_vram_gib'] = floor
    if placement:
        spec['placement'] = placement
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        spec, {}, DeploymentState.LIVE, t, t,
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


def test_pipeline_parallel_multiplies_count():
    """Regression: pp was ignored, so a tp=1,pp=2 deployment got ONE GPU and
    vLLM crashed with insufficient devices (or ran unsharded and OOM'd)."""
    plan = plan_placement([vllm('a', tp=1, pp=2)], inv())
    assert plan.assignments == {'a': [0, 1]}
    plan = plan_placement([vllm('b', tp=2, pp=2)], inv())
    assert plan.assignments == {'b': [0, 1, 2, 3]}


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


def test_pin_outside_allowed_gpus_is_honored():
    # Slurm shared-stack case: job A's acquire restricts allowed_gpus to its own
    # GPU, but job B's already-running model (pinned on another GPU) must stay
    # put — not be deferred and reshuffled onto A's GPU.
    a = vllm('a', t=0.0)   # the new acquire (job A), allowed only GPU 0
    b = vllm('b', t=1.0)   # job B, already running on GPU 1
    plan = plan_placement(
        [a, b], inv('4x80'), allowed_gpus=[0], pinned={'b': [1]},
    )
    assert plan.ok, plan.errors
    assert plan.assignments['b'] == [1]   # pin honored despite being outside allow-list
    assert plan.assignments['a'] == [0]   # new deployment lands on its allowed GPU


def test_multi_gpu_pin_outside_allowed_is_honored():
    a = vllm('a', t=0.0)
    big = vllm('big', tp=2, t=1.0)   # 2-GPU job already running on [2, 3]
    plan = plan_placement(
        [a, big], inv('4x80'), allowed_gpus=[0], pinned={'big': [2, 3]},
    )
    assert plan.ok, plan.errors
    assert plan.assignments['big'] == [2, 3]
    assert plan.assignments['a'] == [0]


def test_new_deployment_restricted_to_allowed_gpus():
    # A fresh (unpinned) deployment may only land inside allowed_gpus.
    plan = plan_placement([vllm('a')], inv('4x80'), allowed_gpus=[2])
    assert plan.assignments == {'a': [2]}


# ---------------------------------------------------------------------------
# VRAM-aware placement (docs/planning/vram-aware-placement.md).
#
# The motivating host is yardrat: GPU0 = 48 GiB (RTX 8000), GPU1 = 16 GiB
# (RTX 5000). A 9B model (~24 GiB declared) must NEVER land on GPU1; small
# models should gravitate to GPU1 (best-fit) so the big card stays free.
# ---------------------------------------------------------------------------


def test_simulate_inventory_heterogeneous():
    inventory = simulate_inventory('48,16')
    sizes = [g['memory_gib'] for g in inventory['gpus']]
    assert sizes == [48.0, 16.0]
    assert [g['index'] for g in inventory['gpus']] == [0, 1]
    # count syntax composes with the comma list
    inventory = simulate_inventory('2x48,16')
    assert [g['memory_gib'] for g in inventory['gpus']] == [48.0, 48.0, 16.0]
    # the homogeneous legacy form still works
    assert [g['memory_gib'] for g in simulate_inventory('4x96')['gpus']] == [96.0] * 4


def test_simulate_inventory_invalid_spec_errors():
    import pytest
    with pytest.raises(ValueError):
        simulate_inventory('bogus')
    with pytest.raises(ValueError):
        simulate_inventory('48,,16')


def test_vram_requirement_excludes_small_gpu():
    # 9B-class deployment: only the 48 GiB card is eligible.
    plan = plan_placement([vllm('big', vram=24)], inv('48,16'))
    assert plan.ok, plan.errors
    assert plan.assignments == {'big': [0]}


def test_yardrat_scenario():
    # big -> the 48; one small -> the 16; the second small queues (no free
    # eligible GPU) — and big is NEVER on GPU1.
    plan = plan_placement(
        [vllm('big', vram=24, t=0), vllm('small', vram=12, t=1),
         vllm('tiny', vram=3, t=2)],
        inv('48,16'),
    )
    assert plan.assignments['big'] == [0]
    assert plan.assignments['small'] == [1]
    assert 'tiny' not in plan.assignments
    assert any('tiny' in e for e in plan.errors)


def test_anti_starvation_most_constrained_first():
    # Arrival order says small first — but index-order first-fit would park it
    # on GPU0 and strand big (which fits ONLY GPU0). Most-constrained-first +
    # best-fit must place big on 0 and small on 1 regardless of created_at.
    plan = plan_placement(
        [vllm('small', vram=12, t=0), vllm('big', vram=24, t=1)],
        inv('48,16'),
    )
    assert plan.ok, plan.errors
    assert plan.assignments == {'big': [0], 'small': [1]}


def test_declared_best_fit_prefers_smallest_eligible():
    # A lone small model takes the SMALL card, leaving the big one free.
    plan = plan_placement([vllm('small', vram=12)], inv('48,16'))
    assert plan.assignments == {'small': [1]}


def test_undeclared_deployments_keep_legacy_index_order():
    # No declaration -> exactly today's behavior (index-order first-fit), so
    # existing catalogs see byte-identical plans. (A yardrat 9B endpoint that
    # has not yet declared min_vram_gib still lands on GPU0 as it does today.)
    plan = plan_placement([vllm('a', t=0), vllm('b', t=1)], inv('48,16'))
    assert plan.assignments == {'a': [0], 'b': [1]}


def test_mixed_declared_and_undeclared():
    # The declared-constrained deployment places first even if it arrived
    # later; the undeclared one then takes the remaining GPU legacy-style.
    plan = plan_placement(
        [vllm('legacy', t=0), vllm('big', vram=24, t=1)], inv('48,16')
    )
    assert plan.ok, plan.errors
    assert plan.assignments == {'big': [0], 'legacy': [1]}


def test_no_eligible_gpu_exists_is_a_clear_error():
    # Requirement exceeds every GPU on the host: the error must name the
    # requirement and what the host actually has (copy-pasteable diagnosis).
    plan = plan_placement([vllm('big', vram=24)], inv('1x16'))
    assert not plan.ok
    (err,) = plan.errors
    assert 'big' in err and '24' in err and '16' in err


def test_floor_clamps_low_declaration():
    # A too-low declared guess is clamped up by the weight-bytes floor: a 9B
    # declared at 8 GiB (weights 19.3 GB) still never lands on the 16er.
    plan = plan_placement([vllm('big', vram=8, floor=19.3)], inv('48,16'))
    assert plan.assignments == {'big': [0]}


def test_tp_per_shard_requirement():
    # tp=2 with a 24 GiB per-shard requirement needs TWO eligible GPUs.
    plan = plan_placement([vllm('big', tp=2, vram=24)], inv('2x48,16'))
    assert plan.assignments == {'big': [0, 1]}
    plan = plan_placement([vllm('big', tp=2, vram=24)], inv('48,2x16'))
    assert not plan.ok
    assert any('big' in e for e in plan.errors)


def test_allowed_gpus_intersects_eligibility():
    # GPU0 is eligible but not allowed; GPU1 is allowed but ineligible.
    plan = plan_placement(
        [vllm('big', vram=24)], inv('48,16'), allowed_gpus=[1]
    )
    assert not plan.ok
    assert any('big' in e for e in plan.errors)


def test_pinned_wins_over_declaration_with_warning():
    # Stability beats a newly added declaration: the pin is honored, but the
    # disagreement is surfaced as a warning.
    plan = plan_placement(
        [vllm('a', vram=24)], inv('48,16'), pinned={'a': [1]}
    )
    assert plan.assignments == {'a': [1]}
    assert any('a' in w and '24' in w for w in plan.warnings)


def test_explicit_indices_win_over_declaration_with_warning():
    # An operator's explicit gpu_indices is an override, not a bug — honored,
    # but warned about when it contradicts the declared requirement.
    plan = plan_placement(
        [vllm('a', gpu_indices=[1], vram=24)], inv('48,16')
    )
    assert plan.assignments == {'a': [1]}
    assert any('a' in w and '24' in w for w in plan.warnings)


def test_best_fit_ties_break_by_index():
    # Equal-size eligible GPUs: deterministic index order.
    plan = plan_placement(
        [vllm('a', vram=12, t=0), vllm('b', vram=12, t=1)], inv('2x16,48')
    )
    assert plan.assignments == {'a': [0], 'b': [1]}
