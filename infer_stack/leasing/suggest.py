"""Seed a catalog from server introspection — the suggestion pool + the join.

The leasing controller reads a hand-built ``catalog.yaml``, but a fresh host
shouldn't start empty. This module turns *what the server is* (the detected GPU
inventory) plus *what's worth running* (a curated, hardware-independent pool of
models) into a concrete, fits-this-box catalog the user can review and merge.

The design is that **seeding is a pure function**
``inventory × pool → suggested catalog`` — the same compiler/controller split
the rest of the leasing redesign rests on. Nothing here is
baked into the catalog; re-run it on a new box and you get that box's
suggestions. The pool lives in ``templates/suggestion-pool.yaml`` (lifted from
the real entries of the legacy ``default-vllm-models.yaml``).

Two layers, kept apart on purpose:

* :class:`SuggestionModel` — intrinsic, portable model facts (footprint, min
  per-GPU VRAM, preferred GPU count, context window, sane vLLM defaults).
* :func:`suggest_catalog` — derives the *server-specific* layer (which models
  fit, what ``max_model_len`` / ``gpu_memory_utilization`` / ``dtype`` to use,
  which one to keep warm) by joining the pool against an inventory dict.

Example:
    >>> from infer_stack.hardware import simulate_inventory
    >>> # a single 48 GiB GPU (yardrat's free Quadro RTX 8000)
    >>> out = suggest_catalog(simulate_inventory('1x48'))
    >>> 'qwen2.5-7b' in out['models'] and 'qwen2.5-72b' not in out['models']
    True
    >>> out['endpoints']['qwen2.5-7b']['runtime']['max_model_len']
    32768
    >>> # the 72B (needs 2 GPUs) only appears once there are two
    >>> 'qwen2.5-72b' in suggest_catalog(simulate_inventory('2x80'))['models']
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..hardware import available_gpu_indices

__all__ = [
    'SuggestionModel',
    'builtin_pool',
    'load_pool',
    'fits_on',
    'derive_runtime',
    'suggest_catalog',
]


@dataclass
class SuggestionModel:
    """One pool entry: intrinsic facts about a model, independent of hardware."""

    name: str
    hf_model_id: str
    served_model_name: str | None = None
    family: str | None = None
    modalities: list[str] = field(default_factory=lambda: ['text'])
    memory_class_gib: float = 0.0
    min_vram_gib_per_replica: float = 0.0
    preferred_gpu_count: int = 1
    context_window: int | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(cls, name: str, spec: dict[str, Any]) -> SuggestionModel:
        return cls(
            name=name,
            hf_model_id=spec['hf_model_id'],
            served_model_name=spec.get('served_model_name'),
            family=spec.get('family'),
            modalities=list(spec.get('modalities') or ['text']),
            memory_class_gib=spec.get('memory_class_gib', 0) or 0,
            min_vram_gib_per_replica=(
                spec.get('min_vram_gib_per_replica')
                or spec.get('memory_class_gib', 0)
                or 0
            ),
            preferred_gpu_count=int(spec.get('preferred_gpu_count', 1) or 1),
            context_window=spec.get('context_window'),
            defaults=dict(spec.get('defaults') or {}),
        )


# ---------------------------------------------------------------------------
# pool loading
# ---------------------------------------------------------------------------


def load_pool(path: str | Path) -> dict[str, SuggestionModel]:
    """Parse a suggestion-pool YAML file into typed entries."""
    data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    return _pool_from_dict(data)


def builtin_pool() -> dict[str, SuggestionModel]:
    """The curated pool shipped in ``templates/suggestion-pool.yaml``."""
    from importlib.resources import files

    text = (
        files('infer_stack')
        .joinpath('templates/suggestion-pool.yaml')
        .read_text(encoding='utf-8')
    )
    return _pool_from_dict(yaml.safe_load(text) or {})


def _pool_from_dict(data: dict[str, Any]) -> dict[str, SuggestionModel]:
    entries = data.get('vllm_models') or {}
    return {
        name: SuggestionModel.from_entry(name, spec or {})
        for name, spec in entries.items()
    }


# ---------------------------------------------------------------------------
# the join: inventory × pool → derived runtime
# ---------------------------------------------------------------------------

# GPUs that predate Ampere (compute capability < 8.0) have no bf16, so vLLM must
# be pinned to fp16 (``--dtype=half``). The inventory does not yet carry the
# compute capability (``detect_inventory`` queries name/memory/display only), so
# we sniff the GPU *name*. This is the one place that wants a real signal: adding
# ``compute_cap`` to the nvidia-smi query in hardware.py would make this exact.
_PRE_AMPERE_NAME_HINTS = (
    'quadro rtx',  # Turing Quadro (RTX 8000/6000/5000/4000) — e.g. yardrat
    'titan rtx',
    'titan v',
    'tesla t4',
    ' t4',
    'tesla v100',
    'v100',
    'rtx 20',  # GeForce RTX 2080 etc. (Turing)
    'gtx 16',  # GTX 1660 (Turing, no tensor cores)
    'gtx 10',  # Pascal
    'tesla p100',
    'tesla p40',
)


def _needs_fp16(gpu_name: str | None) -> bool:
    name = (gpu_name or '').lower()
    return any(h in name for h in _PRE_AMPERE_NAME_HINTS)


def _gpu_mem(gpu: dict[str, Any]) -> float:
    return float(gpu.get('memory_gib') or 0.0)


def fits_on(model: SuggestionModel, gpus: list[dict[str, Any]]) -> bool:
    """True iff ``preferred_gpu_count`` GPUs each hold one replica's VRAM."""
    big_enough = [g for g in gpus if _gpu_mem(g) >= model.min_vram_gib_per_replica]
    return len(big_enough) >= model.preferred_gpu_count


def _host_gpus(
    model: SuggestionModel, gpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The GPUs this model would most plausibly land on (smallest that fit).

    Picking the *smallest* viable GPUs (rather than the largest) makes the
    derived ``gpu_memory_utilization`` honest: it must reserve enough on the
    tightest GPU the placer might choose, not the roomiest.
    """
    big_enough = sorted(
        (g for g in gpus if _gpu_mem(g) >= model.min_vram_gib_per_replica),
        key=_gpu_mem,
    )
    return big_enough[: model.preferred_gpu_count]


def derive_runtime(
    model: SuggestionModel, gpus: list[dict[str, Any]]
) -> dict[str, Any]:
    """Derive a concrete vLLM runtime block for ``model`` on ``gpus``.

    * ``max_model_len`` — the pool default, clamped to the context window.
    * ``gpu_memory_utilization`` — sized from footprint ÷ host-GPU VRAM (with KV
      headroom) so a small model on a big GPU does not greedily claim it all,
      matching the hand-tuned 0.2-0.4 values in the leasing demo.
    * ``tensor_parallel_size`` — ``preferred_gpu_count`` when > 1.
    * ``extra_args: [--dtype=half]`` — only on pre-Ampere GPUs (no bf16).
    """
    host = _host_gpus(model, gpus)
    host_mem = min((_gpu_mem(g) for g in host), default=0.0)

    runtime: dict[str, Any] = {}

    want_len = model.defaults.get('max_model_len') or model.context_window
    if want_len and model.context_window:
        want_len = min(want_len, model.context_window)
    if want_len:
        runtime['max_model_len'] = int(want_len)

    # Footprint over the (smallest) host GPU, padded ~30%, then bounded to a sane
    # band. Crucially this is a *floor-raiser*, never a floor-lowerer: the pool's
    # own ``gpu_memory_utilization`` default encodes the fraction the model needs
    # for its context's KV cache, so sizing *below* it (as the bare footprint
    # ratio can on a big GPU) starves the KV cache and the engine OOMs at
    # startup. We therefore take the max of the footprint estimate and the pool
    # default, so the computed value can only *raise* the reservation on a
    # smaller GPU where the model needs a bigger slice. Fall back to the default
    # when host GPU mem is unknown.
    default_util = model.defaults.get('gpu_memory_utilization')
    if host_mem > 0 and model.min_vram_gib_per_replica > 0:
        footprint = (model.min_vram_gib_per_replica * 1.3) / host_mem
        util = footprint if default_util is None else max(footprint, default_util)
        util = max(0.2, min(0.92, round(util, 2)))
    else:
        util = default_util if default_util is not None else 0.9
    runtime['gpu_memory_utilization'] = util

    if model.preferred_gpu_count > 1:
        runtime['tensor_parallel_size'] = model.preferred_gpu_count

    if model.defaults.get('enable_prefix_caching'):
        runtime['enable_prefix_caching'] = True

    if any(_needs_fp16(g.get('name')) for g in host):
        runtime['extra_args'] = ['--dtype=half']

    return runtime


# ---------------------------------------------------------------------------
# the top-level pure function
# ---------------------------------------------------------------------------


def suggest_catalog(
    inventory: dict[str, Any],
    *,
    pool: dict[str, SuggestionModel] | None = None,
    reserve_display_gpu: str | bool | None = False,
) -> dict[str, Any]:
    """Join an inventory against the pool into a mergeable catalog fragment.

    Returns ``{'models': {...}, 'endpoints': {...}}`` shaped exactly like a
    ``catalog.yaml`` (so it round-trips through
    :meth:`~infer_stack.leasing.catalog.Catalog.from_dict` and merges into an
    existing catalog without rewriting it). The single largest fitting model is
    marked ``reclaim: keep-warm`` (worth the resident GPU to avoid cold-start
    thrash); the rest ``reclaim: stop`` — mirroring the leasing demo.

    Pure and offline: pass ``simulate_inventory('2x80')`` to suggest for
    hardware you do not have in front of you.
    """
    pool = builtin_pool() if pool is None else pool
    all_gpus = inventory.get('gpus') or []
    allowed = set(available_gpu_indices(inventory, reserve_display_gpu))
    gpus = [g for g in all_gpus if g.get('index') in allowed]

    fitting = [m for m in pool.values() if fits_on(m, gpus)]
    # Largest first, so the keep-warm pick is the biggest thing this box can run.
    fitting.sort(key=lambda m: m.min_vram_gib_per_replica, reverse=True)

    models: dict[str, Any] = {}
    endpoints: dict[str, Any] = {}
    for rank, model in enumerate(fitting):
        models[model.name] = {'source': f'hf://{model.hf_model_id}'}
        endpoint: dict[str, Any] = {'engine': 'vllm', 'model': model.name}
        runtime = derive_runtime(model, gpus)
        if runtime:
            endpoint['runtime'] = runtime
        endpoint['reclaim'] = {'policy': 'keep-warm' if rank == 0 else 'stop'}
        endpoints[model.name] = endpoint

    return {'models': models, 'endpoints': endpoints}
