from __future__ import annotations

import csv
import subprocess
from copy import deepcopy
from typing import Any


def _run(cmd: list[str], *, timeout: float = 20.0) -> str:
    # Bounded: a wedged driver can make nvidia-smi hang indefinitely, and this
    # runs on interactive paths (placement, the TUI's system pane). A timeout
    # degrades to the same empty-inventory result as nvidia-smi being absent.
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout
        )
    except Exception:
        return ''
    return out


def simulate_inventory(spec: str) -> dict[str, Any]:
    """Build a fake inventory from a compact GPU specification.

    Comma-separated entries are ``M`` / ``NxM`` for memory-only simulation or
    ``M@CC`` / ``NxM@CC`` when compute capability matters to a suggestion.
    Examples: ``4x96@12.0`` approximates a four-GPU Blackwell workstation,
    ``48@7.5`` a high-VRAM Turing card, and ``48,16`` keeps the historical
    memory-only spelling.  The optional capability makes class-gated catalog
    suggestions testable without hard-coding product names.
    """
    entries: list[tuple[float, float | None]] = []
    try:
        for raw_entry in spec.lower().split(','):
            raw_entry = raw_entry.strip()
            if not raw_entry:
                raise ValueError
            if '@' in raw_entry:
                shape, cap_str = raw_entry.rsplit('@', 1)
                compute_cap = float(cap_str)
            else:
                shape = raw_entry
                compute_cap = None
            if 'x' in shape:
                count_str, gib_str = shape.split('x', 1)
                entries.extend(
                    [(float(gib_str), compute_cap)] * int(count_str)
                )
            else:
                entries.append((float(shape), compute_cap))
        if not entries:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError(
            f'Invalid --simulate-hardware spec {spec!r}. Expected '
            'comma-separated NxM[@CC] or M[@CC] entries '
            '(e.g. 4x96@12.0, 2x80@9.0, 48@7.5,16).'
        )
    gpus = []
    for i, (memory_gib, compute_cap) in enumerate(entries):
        suffix = '' if compute_cap is None else f', sm{int(round(compute_cap * 10))}'
        gpu = {
            'index': i,
            'uuid': f'GPU-simulated-{i:04d}',
            'name': f'Simulated GPU ({memory_gib:.0f}GiB{suffix})',
            'memory_mib': int(memory_gib * 1024),
            'memory_gib': memory_gib,
            'display_active': False,
        }
        if compute_cap is not None:
            gpu['compute_cap'] = compute_cap
        gpus.append(gpu)
    return {'gpu_count': len(gpus), 'gpus': gpus}


def detect_inventory() -> dict[str, Any]:
    query = _run(
        [
            'nvidia-smi',
            '--query-gpu=index,uuid,name,memory.total,display_active',
            '--format=csv,noheader,nounits',
        ]
    )
    # Keep compute capability optional.  Asking for it in a second query means
    # an older nvidia-smi that does not expose ``compute_cap`` cannot turn an
    # otherwise healthy host into an empty inventory.
    cap_query = _run(
        [
            'nvidia-smi',
            '--query-gpu=index,compute_cap',
            '--format=csv,noheader,nounits',
        ]
    )
    caps: dict[int, float] = {}
    if cap_query:
        reader = csv.reader(
            line for line in cap_query.splitlines() if line.strip()
        )
        for row in reader:
            if len(row) < 2:
                continue
            try:
                caps[int(row[0].strip())] = float(row[1].strip())
            except ValueError:
                continue

    gpus: list[dict[str, Any]] = []
    if query:
        reader = csv.reader(line for line in query.splitlines() if line.strip())
        for row in reader:
            if len(row) < 5:
                continue
            idx, uuid, name, mem, display_active = [x.strip() for x in row[:5]]
            index = int(idx)
            gpu = {
                'index': index,
                'uuid': uuid,
                'name': name,
                'memory_mib': int(float(mem)),
                'memory_gib': round(int(float(mem)) / 1024, 2),
                'display_active': display_active.lower()
                in {'enabled', 'active', 'on', 'true'},
            }
            if index in caps:
                gpu['compute_cap'] = caps[index]
            gpus.append(gpu)
    return {
        'gpu_count': len(gpus),
        'gpus': gpus,
    }


# ---------------------------------------------------------------------------
# Low-level GPU-pool placement primitives.
#
# Shared by the legacy resolver (single profile) and the leasing placement
# planner (the union of live deployment groups), so there is one home for
# "which GPUs are available" and "first-fit N of them".
# ---------------------------------------------------------------------------


def available_gpu_indices(
    inventory: dict[str, Any], reserve_display_gpu: str | bool | None
) -> list[int]:
    """GPU indices in the inventory, optionally skipping display-active ones."""
    gpus = deepcopy(inventory.get('gpus', []))
    if reserve_display_gpu in ('auto', True):
        return [g['index'] for g in gpus if not g.get('display_active')]
    return [g['index'] for g in gpus]


def first_fit(available: list[int], count: int) -> tuple[list[int], str | None]:
    """Take the first ``count`` available indices, or report the shortfall."""
    if len(available) < count:
        return (
            available[:],
            f'need {count} GPUs but only {len(available)} available',
        )
    return available[:count], None


def resolve_gpu_indices(
    *,
    name: str,
    placement: dict[str, Any],
    topology: dict[str, Any],
    preferred_gpu_count: int,
    available: list[int],
) -> tuple[list[int], str | None]:
    """Resolve a single runtime's GPU indices from its placement/topology."""
    strategy = placement.get('strategy', 'first_fit')
    if strategy in {'exact', 'multi_gpu', 'single_gpu'}:
        gpu_indices = list(placement.get('gpu_indices', []))
        if not gpu_indices:
            return (
                [],
                f'{name} uses {strategy} placement but no gpu_indices were provided',
            )
        return gpu_indices, None
    gpu_count = int(
        placement.get(
            'gpu_count',
            topology.get('tensor_parallel_size', preferred_gpu_count) or 1,
        )
    )
    return first_fit(available, gpu_count)
