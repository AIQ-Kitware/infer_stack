from __future__ import annotations

import csv
import subprocess
from copy import deepcopy
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ''
    return out


def simulate_inventory(spec: str) -> dict[str, Any]:
    """Build a fake inventory from a spec string like '4x96' (4 GPUs × 96 GiB)."""
    try:
        count_str, gib_str = spec.lower().split('x', 1)
        gpu_count = int(count_str)
        memory_gib = float(gib_str)
    except (ValueError, AttributeError):
        raise ValueError(
            f'Invalid --simulate-hardware spec {spec!r}. Expected format: NxM (e.g. 4x96, 2x80).'
        )
    memory_mib = int(memory_gib * 1024)
    gpus = [
        {
            'index': i,
            'uuid': f'GPU-simulated-{i:04d}',
            'name': f'Simulated GPU ({memory_gib:.0f}GiB)',
            'memory_mib': memory_mib,
            'memory_gib': memory_gib,
            'display_active': False,
        }
        for i in range(gpu_count)
    ]
    return {'gpu_count': gpu_count, 'gpus': gpus}


def detect_inventory() -> dict[str, Any]:
    query = _run(
        [
            'nvidia-smi',
            '--query-gpu=index,uuid,name,memory.total,display_active',
            '--format=csv,noheader,nounits',
        ]
    )
    gpus: list[dict[str, Any]] = []
    if query:
        reader = csv.reader(line for line in query.splitlines() if line.strip())
        for row in reader:
            if len(row) < 5:
                continue
            idx, uuid, name, mem, display_active = [x.strip() for x in row[:5]]
            gpus.append(
                {
                    'index': int(idx),
                    'uuid': uuid,
                    'name': name,
                    'memory_mib': int(float(mem)),
                    'memory_gib': round(int(float(mem)) / 1024, 2),
                    'display_active': display_active.lower()
                    in {'enabled', 'active', 'on', 'true'},
                }
            )
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
