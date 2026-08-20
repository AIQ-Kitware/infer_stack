"""Single-host GPU placement over the union of live deployment deployments.

The ledger deliberately does no GPU assignment — it only tracks demand. A
single-host backend (Compose) must place the *whole live set* of deployments onto the
GPU pool at once: this is the "minimal single-host placer because nothing else
will" from the redesign. Multi-node placement, preemption, and migration are
explicitly out of scope (that is KubeAI/k8s/Slurm territory); single-host
VRAM-eligibility filtering + greedy best-fit IS in scope — see
``docs/planning/vram-aware-placement.md`` for the objective and the scope
amendment. At this tool's scale (≤ 8 GPUs, a handful of deployments) greedy
best-fit is adequate; there is deliberately no bin-packing solver.

Placement rules, applied in a deterministic order so the result is stable across
reconciles:

1. **pinned** deployments (already realized) keep their current GPUs when still valid,
   so adding/removing a deployment does not reshuffle running models. A pin
   that contradicts a (newer) VRAM declaration is honored but warned about.
2. **explicit** deployments (an Ollama daemon's ``gpu_indices``, or a vLLM deployment with
   an explicit placement) claim exactly those GPUs — again honored over a
   contradicting declaration, with a warning.
3. **fit** deployments (vLLM needing ``tp × pp × dp`` GPUs) fill the remaining
   pool. A deployment declaring ``placement.min_vram_gib`` may only land on
   GPUs with at least that much memory (per shard); such deployments are
   placed most-constrained-first and take the *smallest* eligible free GPUs
   (best-fit), so small models gravitate to small cards and the big card
   stays free for the model that needs it. Deployments with no declaration
   keep the legacy index-order first-fit — byte-identical plans to the
   pre-VRAM-aware planner.

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
    """Result of placing a set of deployments: ``deployment_id -> gpu indices``.

    ``warnings`` carry honored-but-suspect decisions (a pin or explicit index
    that contradicts a declared VRAM requirement); they never fail the plan.
    """

    assignments: dict[str, list[int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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
    """GPUs a deployment needs.

    A reservation asks for a plain ``reserved_gpu_count`` (count-based first-fit,
    never a pinned index). A vLLM deployment needs tensor × pipeline × data
    parallelism.

    A ``runtime.simulator`` deployment needs **none**: it serves the API from
    CPU. That is not a detail -- ``max(1, ...)`` below would otherwise make
    every simulator endpoint unplaceable on a GPU-less host, which is exactly
    the host a simulator exists to serve.
    """
    reserved = deployment.spec.get('reserved_gpu_count')
    if reserved:
        return max(1, int(reserved))
    runtime = deployment.spec.get('runtime', {}) or {}
    if runtime.get('simulator'):
        return 0
    tp = int(runtime.get('tensor_parallel_size', 1) or 1)
    pp = int(runtime.get('pipeline_parallel_size', 1) or 1)
    dp = int(runtime.get('data_parallel_size', 1) or 1)
    return max(1, tp * pp * dp)


def weight_shard_count(deployment: Deployment) -> int:
    """How many ways a deployment's WEIGHTS are split across its GPUs.

    ``tensor_parallel_size`` and ``pipeline_parallel_size`` shard the model --
    each GPU holds its slice. ``data_parallel_size`` REPLICATES it: every
    replica needs the whole thing, so it must not divide the floor. That is
    why this is not simply :func:`required_gpu_count`, which multiplies all
    three.
    """
    if deployment.spec.get('reserved_gpu_count'):
        return 1
    runtime = deployment.spec.get('runtime', {}) or {}
    tp = int(runtime.get('tensor_parallel_size', 1) or 1)
    pp = int(runtime.get('pipeline_parallel_size', 1) or 1)
    return max(1, tp * pp)


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


def declared_min_vram(deployment: Deployment) -> float:
    """The catalog-declared (or measured-overlay) requirement, 0.0 if none.

    Declaring is the deployment's opt-in to best-fit selection; the floor
    alone (below) never is.
    """
    placement = deployment.spec.get('placement', {}) or {}
    return float(placement.get('min_vram_gib') or 0.0)


def min_vram_per_gpu(deployment: Deployment) -> float:
    """The per-GPU VRAM *eligibility* requirement in GiB (0.0 = unconstrained).

    ``placement.min_vram_gib`` is the declared best guess (catalog) or a
    recorded measurement; ``placement.floor_vram_gib`` is the machine-derived
    weight-bytes floor (a guaranteed underestimate of need, enriched at plan
    time once the weights are in the local HF cache). The effective
    requirement is ``max(declared, floor)``: the floor clamps an unsoundly
    low guess, so a 9B declared at 8 GiB still never lands on a 16-GiB card.

    The floor affects ELIGIBILITY only — a floored-but-undeclared deployment
    keeps legacy index-order selection (see ``plan_placement`` tier 3), so
    the mere act of downloading weights never changes where an existing
    catalog's deployments land on a host where everything fits everywhere.

    ⚠️ The floor is a WHOLE-MODEL figure and this function returns a PER-GPU
    one, so it is divided by how many ways the weights are sharded. Without
    that division a tensor-parallel deployment demands the entire model on
    each of its cards, which no card can satisfy unless one could have held
    the whole model alone -- i.e. tensor parallelism becomes unusable exactly
    when it is needed. Observed: qwen2.5-72b at tensor_parallel_size 2 asked
    for 135.43 GiB on each of two 95.59 GiB cards ("the pool can never satisfy
    that"), where ~68 GiB per card is the real requirement.

    The declared value is NOT divided. ``placement.min_vram_gib`` is per-GPU
    by convention -- the catalogs in this repo declare 72 for a 51.8 GiB model
    on one card -- so dividing it would silently halve every hand-written
    requirement.
    """
    placement = deployment.spec.get('placement', {}) or {}
    floor = float(placement.get('floor_vram_gib') or 0.0)
    floor = floor / weight_shard_count(deployment)
    return max(declared_min_vram(deployment), floor)


def _memory_map(inventory: dict[str, Any]) -> dict[int, float]:
    """Real GPU index -> memory_gib (0.0 when the inventory lacks it)."""
    return {
        g['index']: float(g.get('memory_gib') or 0.0)
        for g in inventory.get('gpus', [])
    }


def _format_pool(pool: list[int], memory: dict[int, float]) -> str:
    """Human/copy-pasteable pool summary: ``gpu0=48.0GiB, gpu1=16.0GiB``."""
    if not pool:
        return '(empty)'
    return ', '.join(f'gpu{i}={memory.get(i, 0.0):g}GiB' for i in pool)


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
    memory = _memory_map(inventory)

    def declaration_warning(
        deployment: Deployment, want: list[int], kind: str
    ) -> None:
        """Warn when an honored pin/explicit index contradicts a declared
        VRAM requirement — the override wins (stability / operator intent),
        but the disagreement must be visible, not silent."""
        req = min_vram_per_gpu(deployment)
        below = [i for i in want if memory.get(i, 0.0) < req]
        if req > 0 and below:
            detail = ', '.join(
                f'gpu{i}={memory.get(i, 0.0):g}GiB' for i in below
            )
            plan.warnings.append(
                f'{deployment.id}: {kind} placement on {want} contradicts '
                f'its declared min_vram_gib={req:g} ({detail} is below it); '
                f'{kind} honored — fix the {kind} or the declaration'
            )

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
            declaration_warning(deployment, want, 'pinned')
        else:
            deferred.append(deployment)  # pin no longer valid; replace below

    # 2) explicit-placement deployments.
    fit_deployments: list[Deployment] = []
    for deployment in deferred:
        want = explicit_indices(deployment)
        if want is None:
            fit_deployments.append(deployment)
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
        declaration_warning(deployment, want, 'explicit')

    # 3) fit deployments fill what remains. Constrained deployments (declared
    # min_vram_gib and/or floor) go most-constrained-first and take the
    # SMALLEST eligible free GPUs (best-fit), so small models gravitate to
    # small cards and the only GPU a big model fits on stays free for it.
    # Unconstrained deployments keep the legacy index-order first-fit —
    # byte-identical plans for catalogs that declare nothing. Sort key and
    # eligibility use the static pool (not the free set) so the order is
    # deterministic and independent of placement progress.
    def fit_order(deployment: Deployment) -> tuple:
        req = min_vram_per_gpu(deployment)
        n_eligible = sum(1 for i in pool if memory.get(i, 0.0) >= req)
        return (n_eligible, deployment.created_at, deployment.id)

    for deployment in sorted(fit_deployments, key=fit_order):
        count = required_gpu_count(deployment)
        req = min_vram_per_gpu(deployment)
        if req <= 0:
            remaining = [i for i in pool if i not in used]
            indices, error = _first_fit(remaining, count)
            if error:
                plan.errors.append(f'{deployment.id}: {error}')
                continue
        else:
            eligible = [i for i in pool if memory.get(i, 0.0) >= req]
            if len(eligible) < count:
                # Permanent: this pool can NEVER satisfy the requirement —
                # say so with the inventory, copy-pasteably, instead of the
                # generic shortfall (which reads as "wait and retry").
                plan.errors.append(
                    f'{deployment.id}: needs {count} GPU(s) with '
                    f'>= {req:g} GiB each, but the pool can never satisfy '
                    f'that ({len(eligible)} eligible; '
                    f'pool: {_format_pool(pool, memory)})'
                )
                continue
            free = [i for i in eligible if i not in used]
            if len(free) < count:
                # Transient: eligible GPUs exist but are busy — the queue
                # case; a later converge can succeed.
                plan.errors.append(
                    f'{deployment.id}: need {count} eligible GPU(s) '
                    f'(>= {req:g} GiB) but only {len(free)} free'
                )
                continue
            if declared_min_vram(deployment) > 0:
                # Declaring opts into best-fit (smallest eligible free GPUs).
                best = sorted(
                    free, key=lambda i: (memory.get(i, 0.0), i)
                )[:count]
                indices = sorted(best)
            else:
                # Floor-only (weights merely present in the HF cache): the
                # floor gates eligibility, but selection stays legacy
                # index-order — downloading weights must never move an
                # undeclared deployment on a host where everything fits.
                indices = free[:count]
        plan.assignments[deployment.id] = indices
        used.update(indices)

    return plan
