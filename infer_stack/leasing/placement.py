"""Single-host GPU placement over the union of live deployment groups.

The ledger deliberately does no GPU assignment — it only tracks demand. A
single-host backend (Compose) must place the *whole live set* of groups onto the
GPU pool at once: this is the "minimal single-host placer because nothing else
will" from the redesign. Multi-node / bin-packing / preemption are explicitly
out of scope (that is KubeAI/k8s/Slurm territory).

Placement rules, applied in a deterministic order so the result is stable across
reconciles:

1. **pinned** groups (already realized) keep their current GPUs when still valid,
   so adding/removing a group does not reshuffle running models.
2. **explicit** groups (an Ollama daemon's ``gpu_indices``, or a vLLM group with
   an explicit placement) claim exactly those GPUs.
3. **first-fit** groups (vLLM needing ``tensor_parallel_size × data_parallel_size``
   GPUs) fill the remaining pool in order.

The pool is the inventory minus display GPUs (optional), minus anything not in
``allowed_gpus``, minus ``reserved`` (raw-GPU reservations, Phase 2). Real GPU
indices are preserved throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..hardware import available_gpu_indices as _available_gpu_indices
from ..hardware import first_fit as _first_fit
from .models import DeploymentGroup

OLLAMA = 'ollama'


@dataclass
class GpuPlan:
    """Result of placing a set of groups: ``group_id -> gpu indices``."""

    assignments: dict[str, list[int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def available_indices(
    inventory: dict[str, Any],
    *,
    allowed_gpus: list[int] | None = None,
    reserved: list[int] | tuple[int, ...] = (),
    skip_display: bool = True,
) -> list[int]:
    """The placeable GPU pool, in real-index order.

    Reuses the resolver's display-GPU handling, then applies the allow-list and
    raw-GPU reservations.
    """
    indices = _available_gpu_indices(
        inventory, 'auto' if skip_display else False
    )
    reserved_set = set(reserved)
    if allowed_gpus is not None:
        allowed_set = set(allowed_gpus)
        indices = [i for i in indices if i in allowed_set]
    return [i for i in indices if i not in reserved_set]


def required_gpu_count(group: DeploymentGroup) -> int:
    """GPUs a vLLM group needs = tensor_parallel × data_parallel."""
    runtime = group.spec.get('runtime', {}) or {}
    tp = int(runtime.get('tensor_parallel_size', 1) or 1)
    dp = int(runtime.get('data_parallel_size', 1) or 1)
    return max(1, tp * dp)


def explicit_indices(group: DeploymentGroup) -> list[int] | None:
    """Indices a group pins explicitly, or ``None`` if it wants first-fit.

    Ollama daemons always pin (possibly to ``[]`` for CPU); a vLLM group pins
    only if its runtime carries ``gpu_indices``.
    """
    spec = group.spec
    if spec.get('engine') == OLLAMA:
        return [int(i) for i in (spec.get('gpu_indices') or [])]
    runtime = spec.get('runtime', {}) or {}
    if runtime.get('gpu_indices'):
        return [int(i) for i in runtime['gpu_indices']]
    return None


def _sorted(groups: list[DeploymentGroup]) -> list[DeploymentGroup]:
    return sorted(groups, key=lambda g: (g.created_at, g.id))


def plan_placement(
    groups: list[DeploymentGroup],
    inventory: dict[str, Any],
    *,
    allowed_gpus: list[int] | None = None,
    reserved: list[int] | tuple[int, ...] = (),
    pinned: dict[str, list[int]] | None = None,
    skip_display: bool = True,
) -> GpuPlan:
    """Assign GPUs to every group, or record per-group placement errors.

    Example:
        >>> from infer_stack.hardware import simulate_inventory
        >>> from infer_stack.leasing.models import DeploymentGroup, GroupState
        >>> def vllm(gid, tp=1, t=0.0):
        ...     return DeploymentGroup(gid, 'ck', 'vllm', 'shared-compatible',
        ...         {}, {'engine': 'vllm', 'runtime': {'tensor_parallel_size': tp}},
        ...         {}, GroupState.LIVE, t, t)
        >>> plan = plan_placement([vllm('a', tp=2), vllm('b', t=1.0)],
        ...                       simulate_inventory('4x80'))
        >>> plan.assignments
        {'a': [0, 1], 'b': [2]}
    """
    pinned = pinned or {}
    pool = available_indices(
        inventory,
        allowed_gpus=allowed_gpus,
        reserved=reserved,
        skip_display=skip_display,
    )
    pool_set = set(pool)
    used: set[int] = set()
    plan = GpuPlan()

    ordered = _sorted(groups)

    # 1) pinned groups that are still fully placeable keep their GPUs.
    deferred: list[DeploymentGroup] = []
    for group in ordered:
        want = pinned.get(group.id)
        if want is None:
            deferred.append(group)
            continue
        want = [int(i) for i in want]
        if all(i in pool_set for i in want) and not (used & set(want)):
            plan.assignments[group.id] = want
            used.update(want)
        else:
            deferred.append(group)  # pin no longer valid; replace below

    # 2) explicit-placement groups.
    first_fit_groups: list[DeploymentGroup] = []
    for group in deferred:
        want = explicit_indices(group)
        if want is None:
            first_fit_groups.append(group)
            continue
        if not want:  # e.g. CPU-only Ollama daemon
            plan.assignments[group.id] = []
            continue
        invalid = [i for i in want if i not in pool_set]
        clash = used & set(want)
        if invalid or clash:
            reason = []
            if invalid:
                reason.append(f'gpus {invalid} not in pool {sorted(pool_set)}')
            if clash:
                reason.append(f'gpus {sorted(clash)} already in use')
            plan.errors.append(
                f'{group.id}: explicit placement failed ({"; ".join(reason)})'
            )
            continue
        plan.assignments[group.id] = want
        used.update(want)

    # 3) first-fit groups fill what remains.
    for group in first_fit_groups:
        count = required_gpu_count(group)
        remaining = [i for i in pool if i not in used]
        indices, error = _first_fit(remaining, count)
        if error:
            plan.errors.append(f'{group.id}: {error}')
            continue
        plan.assignments[group.id] = indices
        used.update(indices)

    return plan
