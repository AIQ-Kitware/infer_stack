"""Single-host GPU placement over the union of live deployment deployments.

The ledger deliberately does no GPU assignment — it only tracks demand. A
single-host backend (Compose) must place the *whole live set* of deployments onto the
GPU pool at once: this is the "minimal single-host placer because nothing else
will" from the redesign. Multi-node / bin-packing / preemption are explicitly
out of scope (that is KubeAI/k8s/Slurm territory).

Placement rules, applied in a deterministic order so the result is stable across
reconciles:

1. **pinned** deployments (already realized) keep their current GPUs when still valid,
   so adding/removing a deployment does not reshuffle running models.
2. **explicit** deployments (an Ollama daemon's ``gpu_indices``, or a vLLM deployment with
   an explicit placement) claim exactly those GPUs.
3. **first-fit** deployments (vLLM needing ``tensor_parallel_size × data_parallel_size``
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
from .models import Deployment

OLLAMA = 'ollama'


@dataclass
class GpuPlan:
    """Result of placing a set of deployments: ``deployment_id -> gpu indices``."""

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
    skip_display: bool = False,
) -> list[int]:
    """The placeable GPU pool, in real-index order.

    Reuses the resolver's display-GPU handling, then applies the allow-list and
    raw-GPU reservations. ``skip_display`` defaults to *off* — every GPU is
    placeable, so a single-GPU host (whose only GPU drives the display) works;
    set it to leave a monitor's GPU free on a multi-GPU workstation.
    """
    indices = _available_gpu_indices(
        inventory, 'auto' if skip_display else False
    )
    reserved_set = set(reserved)
    if allowed_gpus is not None:
        allowed_set = set(allowed_gpus)
        indices = [i for i in indices if i in allowed_set]
    return [i for i in indices if i not in reserved_set]


def required_gpu_count(deployment: Deployment) -> int:
    """GPUs a vLLM deployment needs = tensor_parallel × data_parallel."""
    runtime = deployment.spec.get('runtime', {}) or {}
    tp = int(runtime.get('tensor_parallel_size', 1) or 1)
    dp = int(runtime.get('data_parallel_size', 1) or 1)
    return max(1, tp * dp)


def explicit_indices(deployment: Deployment) -> list[int] | None:
    """Indices a deployment pins explicitly, or ``None`` if it wants first-fit.

    Ollama daemons always pin (possibly to ``[]`` for CPU); a vLLM deployment pins
    only if its runtime carries ``gpu_indices``.
    """
    spec = deployment.spec
    if spec.get('engine') == OLLAMA:
        return [int(i) for i in (spec.get('gpu_indices') or [])]
    runtime = spec.get('runtime', {}) or {}
    if runtime.get('gpu_indices'):
        return [int(i) for i in runtime['gpu_indices']]
    return None


def _sorted(deployments: list[Deployment]) -> list[Deployment]:
    return sorted(deployments, key=lambda g: (g.created_at, g.id))


def plan_placement(
    deployments: list[Deployment],
    inventory: dict[str, Any],
    *,
    allowed_gpus: list[int] | None = None,
    reserved: list[int] | tuple[int, ...] = (),
    pinned: dict[str, list[int]] | None = None,
    skip_display: bool = False,
) -> GpuPlan:
    """Assign GPUs to every deployment, or record per-deployment placement errors.

    Example:
        >>> from infer_stack.hardware import simulate_inventory
        >>> from infer_stack.leasing.models import Deployment, DeploymentState
        >>> def vllm(gid, tp=1, t=0.0):
        ...     return Deployment(gid, 'ck', 'vllm', 'shared-compatible',
        ...         {}, {'engine': 'vllm', 'runtime': {'tensor_parallel_size': tp}},
        ...         {}, DeploymentState.LIVE, t, t)
        >>> plan = plan_placement([vllm('a', tp=2), vllm('b', t=1.0)],
        ...                       simulate_inventory('4x80'))
        >>> plan.assignments
        {'a': [0, 1], 'b': [2]}
    """
    pinned = pinned or {}
    # New placements (steps 2-3) are restricted to ``allowed_gpus``. But an
    # already-placed (pinned) deployment keeps its GPU as long as that GPU
    # physically exists and isn't otherwise taken — *even if it falls outside this
    # call's allowed_gpus*. In Slurm mode each job's ``acquire`` passes only its
    # own ``$SLURM_JOB_GPUS`` as ``allowed_gpus`` against the shared stack, so
    # other jobs' running models sit on GPUs outside this call's allow-list;
    # validating pins against ``allowed_gpus`` would wrongly defer + reshuffle
    # them. Pin-validity therefore uses the full pool; ``allowed_gpus`` gates only
    # where *new* deployments may land.
    pool = available_indices(
        inventory,
        allowed_gpus=allowed_gpus,
        reserved=reserved,
        skip_display=skip_display,
    )
    pool_set = set(pool)
    pin_pool_set = set(
        available_indices(
            inventory,
            allowed_gpus=None,
            reserved=reserved,
            skip_display=skip_display,
        )
    )
    used: set[int] = set()
    plan = GpuPlan()

    ordered = _sorted(deployments)

    # 1) pinned deployments that are still physically placeable keep their GPUs.
    deferred: list[Deployment] = []
    for deployment in ordered:
        want = pinned.get(deployment.id)
        if want is None:
            deferred.append(deployment)
            continue
        want = [int(i) for i in want]
        if all(i in pin_pool_set for i in want) and not (used & set(want)):
            plan.assignments[deployment.id] = want
            used.update(want)
        else:
            deferred.append(deployment)  # pin no longer valid; replace below

    # 2) explicit-placement deployments.
    first_fit_deployments: list[Deployment] = []
    for deployment in deferred:
        want = explicit_indices(deployment)
        if want is None:
            first_fit_deployments.append(deployment)
            continue
        if not want:  # e.g. CPU-only Ollama daemon
            plan.assignments[deployment.id] = []
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
                f'{deployment.id}: explicit placement failed ({"; ".join(reason)})'
            )
            continue
        plan.assignments[deployment.id] = want
        used.update(want)

    # 3) first-fit deployments fill what remains.
    for deployment in first_fit_deployments:
        count = required_gpu_count(deployment)
        remaining = [i for i in pool if i not in used]
        indices, error = _first_fit(remaining, count)
        if error:
            plan.errors.append(f'{deployment.id}: {error}')
            continue
        plan.assignments[deployment.id] = indices
        used.update(indices)

    return plan
