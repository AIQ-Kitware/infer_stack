"""VRAM requirement sources: weight-bytes floor, measurement, overlay store.

Phase 3 of ``docs/planning/vram-aware-placement.md``. The placement planner
consumes ``spec['placement']['min_vram_gib']`` (declared best guess or a
recorded measurement) and ``spec['placement']['floor_vram_gib']`` (the
weight-bytes floor); this module produces those numbers:

* :func:`weight_floor_gib` — fp16-ish weight bytes from the local HF hub
  cache. A guaranteed *underestimate* of need (a GPU that cannot even hold
  the weights can never work), never the final number. Offline, stat-only,
  fail-open to ``None`` when the model has not been downloaded yet.
* :func:`parse_vllm_memory_profile` / :func:`derive_min_vram_gib` — the real
  number, from vLLM's own memory-profiling log lines. ⚠️ Deliberately NOT
  from ``nvidia-smi memory.used``: vLLM preallocates KV cache to fill
  ``gpu_memory_utilization``, so observed usage reflects the *knob* on *that
  card* (a 0.8B "uses" ~41 GiB of a 48 GiB card at 0.85), not the model's
  requirement.
* :class:`Measurements` — the machine-managed overlay
  (``<state_dir>/measurements.json``). Measured values are recorded here and
  consulted at plan time; they never silently rewrite ``catalog.yaml`` (the
  catalog is a hand-edited, git-tracked recorded fact — promotion into it is
  an explicit operator action).
* :func:`looks_like_cuda_oom` — the failure-path classifier behind the
  guided error ("declared eligibility OOM'd → here is the exact command
  that computes the right number").
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# OOM detection (the guided-failure path)
# ---------------------------------------------------------------------------

#: Substrings that identify a CUDA/VRAM exhaustion in engine logs. Kept to
#: explicit allocator/vLLM signatures — a generic 'error' match would
#: misdiagnose unrelated crashes as OOM and send the operator measuring.
OOM_SIGNATURES = (
    'CUDA out of memory',
    'torch.OutOfMemoryError',
    'OutOfMemoryError',
    'cudaErrorMemoryAllocation',
    # vLLM startup guard: free memory below the requested utilization slice.
    'is less than desired GPU memory utilization',
    # vLLM KV-cache sizing failures on undersized cards.
    'No available memory for the cache blocks',
    'Available KV cache memory is negative',
)


def looks_like_cuda_oom(log_text: str) -> bool:
    """True when an engine log tail reads as VRAM exhaustion."""
    return any(sig in log_text for sig in OOM_SIGNATURES)


# ---------------------------------------------------------------------------
# vLLM memory-profile parsing (the measurement)
# ---------------------------------------------------------------------------

# Serving images are version-pinned per endpoint, so within one deployment the
# log format is stable — but different endpoints pin different vLLM releases,
# so the parser accepts every known phrasing of each component and uses the
# LAST occurrence (a restarted container logs multiple serves; the final one
# describes the running process).
_PATTERNS: dict[str, tuple[str, ...]] = {
    'weights_gib': (
        r'weights memory:\s*([\d.]+)\s*GiB',
        r'model weights take\s*([\d.]+)\s*GiB',
        r'Model loading took\s*([\d.]+)\s*GiB',
    ),
    'non_torch_gib': (
        r'non-torch forward increase memory:\s*([\d.]+)\s*GiB',
        r'non_torch_memory takes\s*([\d.]+)\s*GiB',
        r'non_torch_memory=([\d.]+)\s*GiB',
    ),
    'activation_gib': (
        r'torch peak memory increase:\s*([\d.]+)\s*GiB',
        r'PyTorch activation peak memory takes\s*([\d.]+)\s*GiB',
        r'peak_torch_memory=([\d.]+)\s*GiB',
    ),
    'non_kv_total_gib': (
        r'Total non KV cache memory:\s*([\d.]+)\s*GiB',
    ),
}


def parse_vllm_memory_profile(log_text: str) -> dict[str, float] | None:
    """Extract vLLM's memory-profiling breakdown from a serve log.

    Returns the components found (keys of ``_PATTERNS``), or ``None`` when
    the log carries no weights figure at all — without weights there is
    nothing sound to derive.

    Example:
        >>> text = ('... Memory profiling takes 11.20 seconds. '
        ...         'Total non KV cache memory: 20.19GiB; '
        ...         'torch peak memory increase: 1.42GiB; '
        ...         'non-torch forward increase memory: 0.51GiB; '
        ...         'weights memory: 18.26GiB.')
        >>> profile = parse_vllm_memory_profile(text)
        >>> profile['weights_gib']
        18.26
        >>> profile['non_kv_total_gib']
        20.19
    """
    profile: dict[str, float] = {}
    for key, patterns in _PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, log_text)
            if matches:
                profile[key] = float(matches[-1])
                break
    if 'weights_gib' not in profile:
        return None
    return profile


def derive_min_vram_gib(
    profile: dict[str, float],
    *,
    kv_budget_gib: float = 2.0,
    margin_fraction: float = 0.05,
) -> float:
    """Turn a profiling breakdown into a placement requirement (GiB per GPU).

    ``non_kv_total`` (weights + non-torch + activation peak, as vLLM itself
    sums it) when present, else the component sum; plus a chosen KV budget —
    the profiling line describes what the *engine* needs before KV, and the
    KV cache is OUR serving choice (max_model_len / max_num_seqs), not a
    property of the model — plus a small safety margin for allocator
    fragmentation. Rounded up to 0.1 GiB.

    Example:
        >>> derive_min_vram_gib({'weights_gib': 18.26,
        ...                      'non_torch_gib': 0.51,
        ...                      'activation_gib': 1.42})
        23.2
    """
    base = profile.get('non_kv_total_gib')
    if base is None:
        base = (
            profile.get('weights_gib', 0.0)
            + profile.get('non_torch_gib', 0.0)
            + profile.get('activation_gib', 0.0)
        )
    value = base * (1.0 + margin_fraction) + kv_budget_gib
    return math.ceil(value * 10) / 10


# ---------------------------------------------------------------------------
# Weight-bytes floor (offline, always sound, never sufficient)
# ---------------------------------------------------------------------------


def weight_floor_gib(
    hf_model_id: str | None, hf_cache: str | Path | None
) -> float | None:
    """Weight bytes (GiB) from the local HF hub cache, or ``None`` if absent.

    Stat-only over ``<hf_cache>/hub/models--Org--Name/snapshots/*/`` weight
    files (symlinks into ``blobs/`` are followed). Multiple snapshots (an old
    and a new revision) would double-count, so the LARGEST single snapshot
    wins — the floor must stay a sound underestimate of one served revision,
    not a sum across revisions. Fail-open: any surprise (no cache yet, model
    not downloaded, unreadable entry) returns ``None`` and placement simply
    proceeds floor-less, exactly as before this feature.
    """
    if not hf_model_id or not hf_cache:
        return None
    try:
        model_dir = (
            Path(hf_cache)
            / 'hub'
            / ('models--' + str(hf_model_id).replace('/', '--'))
            / 'snapshots'
        )
        if not model_dir.is_dir():
            return None
        best_bytes = 0
        for snapshot in model_dir.iterdir():
            if not snapshot.is_dir():
                continue
            total = 0
            for entry in snapshot.rglob('*'):
                if entry.suffix in ('.safetensors', '.bin', '.gguf'):
                    try:
                        total += os.path.getsize(entry)  # follows symlinks
                    except OSError:
                        continue
            best_bytes = max(best_bytes, total)
        if best_bytes <= 0:
            return None
        return round(best_bytes / GIB, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The measurements overlay
# ---------------------------------------------------------------------------


def measurement_key(
    *,
    model_ref: str,
    image: str | None = None,
    dtype: str | None = None,
    max_model_len: int | None = None,
) -> str:
    """The identity a measurement is valid for.

    Keyed by exactly the things that change the requirement: the model, the
    engine image (allocator/kernel changes move the numbers), the served
    dtype, and the context length (drives activation + KV sizing).
    """
    return '|'.join(
        [
            str(model_ref),
            str(image or 'default-image'),
            str(dtype or 'auto'),
            str(max_model_len or 0),
        ]
    )


def measurement_key_for_spec(spec: dict[str, Any]) -> str:
    """The measurement key for a resolved deployment/endpoint spec dict."""
    runtime = spec.get('runtime', {}) or {}
    return measurement_key(
        model_ref=spec.get('hf_model_id') or '',
        image=runtime.get('image'),
        dtype=spec.get('dtype'),
        max_model_len=runtime.get('max_model_len'),
    )


class Measurements:
    """Machine-managed measured-requirement overlay (JSON, fail-open).

    Lives beside the leasing state (``<state_dir>/measurements.json``). The
    resolver consults it for deployments whose catalog entry declares
    nothing; ``infer-stack measure --record`` writes it. It is deliberately
    NOT the catalog: promotion of a measured number into ``catalog.yaml`` is
    an explicit, git-diffable operator action.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            # Missing or corrupt overlay must never block placement.
            return {}

    def get_min_vram_gib(self, key: str) -> float | None:
        entry = self._load().get(key)
        if isinstance(entry, dict):
            value = entry.get('min_vram_gib')
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    def record(self, key: str, min_vram_gib: float, **details: Any) -> None:
        data = self._load()
        data[key] = {
            'min_vram_gib': float(min_vram_gib),
            'recorded_at': time.time(),
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, self.path)
