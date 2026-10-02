"""Seed a catalog from server introspection — the suggestion pool + the join.

The leasing controller reads a hand-built ``catalog.yaml``, but a fresh host
shouldn't start empty. This module turns *what the server is* (the detected GPU
inventory) plus *what's worth running* (a curated pool of models and, where
needed, explicitly hardware-gated measured profiles) into a concrete,
fits-this-box catalog the user can review and merge.

The design is that **seeding is a pure function**
``inventory × pool → suggested catalog`` — the same compiler/controller split
the rest of the leasing redesign rests on. Nothing here is
baked into the catalog; re-run it on a new box and you get that box's
suggestions. The pool lives in ``templates/suggestion-pool.yaml`` (lifted from
the real entries of the legacy ``default-vllm-models.yaml``).

Two layers, kept apart on purpose:

* :class:`SuggestionModel` — model facts (footprint, min per-GPU VRAM,
  preferred GPU count, context window, sane vLLM defaults), plus optional
  hardware-gated endpoint variants for measured serving profiles. Variants are
  data: they override generic endpoint runtime fields and never add
  model-specific renderer behavior.
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

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..hardware import available_gpu_indices
from .launch import translate_legacy

__all__ = [
    'SuggestionModel',
    'builtin_pool',
    'load_pool',
    'fits_on',
    'derive_runtime',
    'migrate_known_suggestion_aliases',
    'suggest_catalog',
]


_OLD_DBIRKS_QWEN38_NAME = 'qwen3.8-27b'
_DBIRKS_QWEN38_NAME = 'qwen3.8-27b-dbirks-hyperqwen'
_DBIRKS_QWEN38_SOURCE = 'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound'


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
    # Optional hardware allow-list for a measured/tuned serving profile. These
    # are case-insensitive substrings of the detected nvidia-smi GPU name. A
    # portable model leaves the list empty.
    gpu_name_hints: list[str] = field(default_factory=list)
    # A model source that cannot be useful before Ampere can still set this
    # coarse gate.  More commonly, serving-profile hardware requirements belong
    # in ``default_hardware`` / endpoint variants so one model source can expose
    # different launch strategies on different GPU classes.
    requires_ampere: bool = False
    # Optional class constraints for the *default endpoint* only.  Supported
    # keys mirror variant gates: min/max compute capability, min/max per-GPU
    # VRAM, and gpu_name_hints.  This lets a model source remain suggestible on
    # (say) high-VRAM Turing while the normal default endpoint stays Ampere+.
    default_hardware: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    # Optional named endpoint variants. Each variant can gate itself on GPU
    # capability / VRAM classes (and, for true measured exceptions, name
    # substrings) then overlay generic runtime fields on the model defaults.
    # The generated endpoint is named ``<model>-<variant>`` and still references
    # the same model source. This remains suggestion-time convenience, not a
    # runtime recipe mechanism.
    endpoint_variants: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            gpu_name_hints=[
                str(v).lower() for v in (spec.get('gpu_name_hints') or [])
            ],
            requires_ampere=bool(spec.get('requires_ampere', False)),
            default_hardware=copy.deepcopy(spec.get('default_hardware') or {}),
            defaults=copy.deepcopy(spec.get('defaults') or {}),
            endpoint_variants=copy.deepcopy(spec.get('endpoint_variants') or {}),
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


def migrate_known_suggestion_aliases(data: dict[str, Any]) -> list[str]:
    """Rename obsolete identities emitted by earlier infer-stack suggestions.

    This is deliberately signature-gated. In particular, an unsuffixed
    ``qwen3.8-27b`` entry is *not* assumed to be ours: it is migrated only when
    the model source is the dbirks W4A16 derivative, the same-named endpoint
    uses HyperQwen's recipe, and no other endpoint references the old model.
    That leaves a user-authored or official ``Qwen/Qwen3.8-27B`` entry alone.
    """
    models = data.get('models') or {}
    endpoints = data.get('endpoints') or {}
    if _DBIRKS_QWEN38_NAME in models or _DBIRKS_QWEN38_NAME in endpoints:
        return []

    old_model = models.get(_OLD_DBIRKS_QWEN38_NAME) or {}
    old_endpoint = endpoints.get(_OLD_DBIRKS_QWEN38_NAME) or {}
    runtime = old_endpoint.get('runtime') or {}
    other_refs = [
        name for name, endpoint in endpoints.items()
        if name != _OLD_DBIRKS_QWEN38_NAME
        and endpoint.get('model') == _OLD_DBIRKS_QWEN38_NAME
    ]
    if (
        old_model.get('source') != _DBIRKS_QWEN38_SOURCE
        or old_endpoint.get('model') != _OLD_DBIRKS_QWEN38_NAME
        # The name an older suggestion wrote, or the generic launch it means.
        or translate_legacy(runtime).get('command') != ['single']
        or other_refs
    ):
        return []

    models[_DBIRKS_QWEN38_NAME] = models.pop(_OLD_DBIRKS_QWEN38_NAME)
    migrated_endpoint = endpoints.pop(_OLD_DBIRKS_QWEN38_NAME)
    migrated_endpoint['model'] = _DBIRKS_QWEN38_NAME
    endpoints[_DBIRKS_QWEN38_NAME] = migrated_endpoint
    return [f'{_OLD_DBIRKS_QWEN38_NAME} -> {_DBIRKS_QWEN38_NAME}']


# ---------------------------------------------------------------------------
# the join: inventory × pool → derived runtime
# ---------------------------------------------------------------------------

# GPUs that predate Ampere (compute capability < 8.0) have no native bf16, so
# stock vLLM must be pinned to fp16 (``--dtype=half``).  Current inventories carry
# compute capability when nvidia-smi exposes it; these name hints remain only as
# a compatibility fallback for older drivers and historical/synthetic inventory
# dictionaries that predate that field.
_TURING_NAME_HINTS = (
    'quadro rtx',  # RTX 8000/6000/5000/4000 — e.g. yardrat
    'titan rtx',
    'tesla t4',
    ' t4',
    'rtx 20',  # GeForce RTX 2080 etc.
    'gtx 16',  # GTX 1660
)

_PRE_AMPERE_NAME_HINTS = _TURING_NAME_HINTS + (
    'titan v',
    'tesla v100',
    'v100',
    'gtx 10',  # Pascal
    'tesla p100',
    'tesla p40',
)


def _needs_fp16(gpu_name: str | None) -> bool:
    name = (gpu_name or '').lower()
    return any(h in name for h in _PRE_AMPERE_NAME_HINTS)


def _gpu_mem(gpu: dict[str, Any]) -> float:
    return float(gpu.get('memory_gib') or 0.0)


def _gpu_compute_cap(gpu: dict[str, Any]) -> float | None:
    value = gpu.get('compute_cap')
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_needs_fp16(gpu: dict[str, Any]) -> bool:
    cap = _gpu_compute_cap(gpu)
    if cap is not None:
        return cap < 8.0
    return _needs_fp16(gpu.get('name'))


def _name_matches_hints(gpu: dict[str, Any], hints: list[str]) -> bool:
    if not hints:
        return True
    name = str(gpu.get('name') or '').lower()
    return any(str(hint).lower() in name for hint in hints)


def _compute_cap_matches(
    gpu: dict[str, Any], *, min_cap: float | None, max_cap: float | None
) -> bool:
    """Match a capability class, with a conservative name fallback.

    Real inventories now carry ``compute_cap``.  Older nvidia-smi versions and
    historical/synthetic inventory dictionaries may not.  For the only coarse
    boundary infer-stack previously understood (pre-Ampere vs Ampere+), retain
    the established GPU-name fallback instead of making an unknown capability
    silently exclude all suggestions.
    """
    cap = _gpu_compute_cap(gpu)
    if cap is not None:
        if min_cap is not None and cap < min_cap:
            return False
        if max_cap is not None and cap > max_cap:
            return False
        return True

    name = str(gpu.get('name') or '').lower()
    pre_ampere = _needs_fp16(name)
    if min_cap is not None and min_cap >= 8.0:
        return not pre_ampere
    if max_cap is not None and max_cap < 8.0:
        # The full-context HyperQwen evidence is specifically sm75.  When an
        # old driver cannot expose compute_cap, distinguish Turing from older
        # Pascal/Volta rather than treating every pre-Ampere card as sm75.
        if min_cap is not None and min_cap >= 7.5:
            return any(hint in name for hint in _TURING_NAME_HINTS)
        return pre_ampere
    return True


def _gpu_matches_constraints(
    gpu: dict[str, Any], constraints: dict[str, Any]
) -> bool:
    hints = [str(v).lower() for v in (constraints.get('gpu_name_hints') or [])]
    if not _name_matches_hints(gpu, hints):
        return False

    min_vram = constraints.get('min_vram_gib_per_replica')
    if min_vram is not None and _gpu_mem(gpu) < float(min_vram):
        return False
    max_vram = constraints.get('max_vram_gib_per_replica')
    if max_vram is not None and _gpu_mem(gpu) > float(max_vram):
        return False

    min_cap = constraints.get('min_compute_cap')
    max_cap = constraints.get('max_compute_cap')
    return _compute_cap_matches(
        gpu,
        min_cap=None if min_cap is None else float(min_cap),
        max_cap=None if max_cap is None else float(max_cap),
    )


def _gpu_is_eligible(model: SuggestionModel, gpu: dict[str, Any]) -> bool:
    if _gpu_mem(gpu) < model.min_vram_gib_per_replica:
        return False
    if model.requires_ampere and not _compute_cap_matches(
        gpu, min_cap=8.0, max_cap=None
    ):
        return False
    if not _name_matches_hints(gpu, model.gpu_name_hints):
        return False
    return True


def _profile_host_gpus(
    model: SuggestionModel,
    constraints: dict[str, Any],
    gpus: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Smallest concrete GPUs satisfying model + serving-profile constraints."""
    min_vram = float(
        constraints.get('min_vram_gib_per_replica')
        or model.min_vram_gib_per_replica
        or 0.0
    )
    count = int(
        constraints.get('preferred_gpu_count')
        or model.preferred_gpu_count
        or 1
    )
    eligible = sorted(
        (
            gpu for gpu in gpus
            if _gpu_is_eligible(model, gpu)
            and _gpu_mem(gpu) >= min_vram
            and _gpu_matches_constraints(gpu, constraints)
        ),
        key=_gpu_mem,
    )
    return eligible[:count] if len(eligible) >= count else []


def _default_host_gpus(
    model: SuggestionModel, gpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return _profile_host_gpus(model, model.default_hardware, gpus)


def _variant_host_gpus(
    model: SuggestionModel, variant: dict[str, Any], gpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return _profile_host_gpus(model, variant, gpus)


def _profile_needs_pin(
    model: SuggestionModel,
    constraints: dict[str, Any],
    gpus: list[dict[str, Any]],
) -> bool:
    """Whether runtime placement cannot express this suggestion's class gate.

    ``placement.min_vram_gib`` is portable, but today's live catalog has no
    compute-capability/max-VRAM predicate.  Keep a class-based suggestion
    portable when every GPU that could satisfy the min-VRAM floor is also in
    the class; otherwise exact-pin the endpoint selected during suggestion so a
    heterogeneous host cannot later move it onto incompatible hardware.
    """
    class_keys = {
        'min_compute_cap', 'max_compute_cap', 'max_vram_gib_per_replica',
        'gpu_name_hints',
    }
    if not any(constraints.get(key) not in (None, [], '') for key in class_keys):
        return False
    min_vram = float(
        constraints.get('min_vram_gib_per_replica')
        or model.min_vram_gib_per_replica
        or 0.0
    )
    generic_candidates = [
        gpu for gpu in gpus
        if _gpu_is_eligible(model, gpu) and _gpu_mem(gpu) >= min_vram
    ]
    return any(
        not _gpu_matches_constraints(gpu, constraints)
        for gpu in generic_candidates
    )


def _host_gpus(
    model: SuggestionModel, gpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Backward-compatible helper for the model's default endpoint profile."""
    return _default_host_gpus(model, gpus)


def fits_on(model: SuggestionModel, gpus: list[dict[str, Any]]) -> bool:
    """True iff the default endpoint or at least one variant fits this host."""
    if _default_host_gpus(model, gpus):
        return True
    return any(
        _variant_host_gpus(model, variant, gpus)
        for variant in model.endpoint_variants.values()
    )


def _merge_runtime(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Deep-merge a suggestion variant's generic runtime overrides.

    Nested mappings such as ``env`` are merged so a variant can change only
    ``SPEC`` / ``CTX`` while retaining the launcher's templated PORT, MAX_LEN,
    served-name and mount contract. Lists/scalars replace the base value.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_runtime(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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
    # ``gpus`` may already be the concrete host set selected for a hardware-
    # gated variant, so do not re-apply the default endpoint's constraints here.
    # Only the model-level source constraints remain relevant to runtime sizing.
    host = sorted(
        (g for g in gpus if _gpu_is_eligible(model, g)),
        key=_gpu_mem,
    )[: model.preferred_gpu_count]
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
        # 0.92 is the generic upper bound, but a measured pool default is
        # authoritative. In particular HyperQwen's 3090 profile requires
        # 0.93; applying the generic clamp after max(default, estimate) would
        # contradict the "never lower the default" rule above.
        upper = max(0.92, float(default_util or 0.0))
        util = max(0.2, min(upper, round(util, 2)))
    else:
        util = default_util if default_util is not None else 0.9
    runtime['gpu_memory_utilization'] = util

    if model.preferred_gpu_count > 1:
        runtime['tensor_parallel_size'] = model.preferred_gpu_count
        # Workers across GPUs talk through /dev/shm; Docker's default is
        # 64 MiB (vLLM's Docker guidance: a few GiB, or ipc: host).
        runtime['shm_size'] = '16g'

    if model.defaults.get('enable_prefix_caching'):
        runtime['enable_prefix_caching'] = True

    if model.defaults.get('image'):
        runtime['image'] = str(model.defaults['image'])

    # Generic launch fields (leasing.launch) come from the pool data as-is:
    # a model's launcher knowledge lives there, not in code.
    for key in ('command', 'env', 'mounts'):
        if model.defaults.get(key):
            runtime[key] = copy.deepcopy(model.defaults[key])

    if any(_gpu_needs_fp16(g) for g in host) and 'command' not in runtime:
        # Stock vLLM only: a custom launcher takes no vLLM flags directly.
        runtime['extra_args'] = ['--dtype=half']

    return runtime


#: A simulator endpoint for a host without a GPU (``catalog suggest
#: --simulator``): llm-d-inference-sim answers like vLLM with random text, so
#: the whole workflow (acquire, the gateway, a request, release) runs here.
#: Never a result. The same entry as dev/e2e_tests/catalog-mock.yaml's.
SIMULATOR_FRAGMENT: dict[str, Any] = {
    'models': {'smol135': {'source': 'hf://HuggingFaceTB/SmolLM2-135M-Instruct'}},
    'endpoints': {'mock-smol': {
        'engine': 'vllm',
        'model': 'smol135',
        'runtime': {
            'image': 'ghcr.io/llm-d/llm-d-inference-sim:v0.9.0',
            'max_model_len': 2048,
            'max_num_seqs': 8,
            'simulator': {'kind': 'llm-d-sim', 'mode': 'random', 'seed': 20260731,
                          'time_to_first_token': '120ms',
                          'inter_token_latency': '8ms', 'startup_duration': '10s'},
        },
        'reclaim': {'policy': 'stop'},
    }},
}


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

        default_host = _default_host_gpus(model, gpus)
        if default_host:
            endpoint: dict[str, Any] = {'engine': 'vllm', 'model': model.name}
            if model.min_vram_gib_per_replica > 0:
                endpoint['placement'] = {
                    'min_vram_gib': model.min_vram_gib_per_replica,
                }
            if model.gpu_name_hints or _profile_needs_pin(
                model, model.default_hardware, gpus
            ):
                # Hardware-class gates are suggestion-time facts.  The live
                # placement schema currently expresses only min VRAM + exact
                # indices, so pin only when a heterogeneous host could otherwise
                # move the generated endpoint outside the selected class.
                endpoint.setdefault('placement', {})['gpu_indices'] = [
                    int(g['index']) for g in default_host
                ]
            runtime = derive_runtime(model, default_host)
            if runtime:
                endpoint['runtime'] = runtime
            endpoint['reclaim'] = {
                'policy': 'keep-warm' if rank == 0 else 'stop'
            }
            endpoints[model.name] = endpoint

        # Variants are additional endpoint suggestions over the same model, not
        # duplicate model identities.  A model can now be useful solely through
        # variants (for example the full-context Turing HyperQwen path) even when
        # its ordinary default endpoint targets another GPU class.
        matching_variants: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
        for variant_name, variant in model.endpoint_variants.items():
            variant_host = _variant_host_gpus(model, variant, gpus)
            if variant_host:
                matching_variants.append((variant_name, variant, variant_host))

        preferred_variant = None
        if not default_host and matching_variants:
            preferred_variant = next(
                (
                    name for name, variant, _host in matching_variants
                    if variant.get('preferred')
                ),
                matching_variants[0][0],
            )

        for variant_name, variant, variant_host in matching_variants:
            variant_endpoint: dict[str, Any] = {
                'engine': 'vllm',
                'model': model.name,
            }
            placement: dict[str, Any] = {}
            min_vram = variant.get('min_vram_gib_per_replica')
            if min_vram is None:
                min_vram = model.min_vram_gib_per_replica
            if min_vram:
                placement['min_vram_gib'] = min_vram
            if model.gpu_name_hints or _profile_needs_pin(model, variant, gpus):
                placement['gpu_indices'] = [
                    int(g['index']) for g in variant_host
                ]
            if placement:
                variant_endpoint['placement'] = placement

            variant_runtime = derive_runtime(model, variant_host)
            runtime_overrides = dict(variant.get('runtime') or {})
            # Most variants want deep inheritance (e.g. change only CTX/SPEC),
            # but a different launcher may need a wholly different env mapping.
            # Keep replacement explicit in suggestion data rather than adding
            # model-specific branches here.
            for key in variant.get('runtime_replace_keys') or []:
                variant_runtime.pop(str(key), None)
            variant_runtime = _merge_runtime(
                variant_runtime, runtime_overrides
            )
            if variant_runtime:
                variant_endpoint['runtime'] = variant_runtime

            policy = variant.get('reclaim_policy')
            if policy is None:
                policy = (
                    'keep-warm'
                    if rank == 0 and preferred_variant == variant_name
                    else 'stop'
                )
            variant_endpoint['reclaim'] = {'policy': str(policy)}
            endpoints[f'{model.name}-{variant_name}'] = variant_endpoint

    return {'models': models, 'endpoints': endpoints}
