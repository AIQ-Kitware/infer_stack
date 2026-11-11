"""Read-only storage facts for a proposed K3s/containerd data root."""
from __future__ import annotations

import math
import os
from pathlib import Path

import yaml


def _bytes(value, total: int) -> float:
    """Kubelet percentage or binary-byte threshold; never guess unknown units."""
    text = str(value)
    if text.endswith('%'):
        number = float(text[:-1])
        if not 0 <= number <= 100:
            raise ValueError('percentage outside 0..100')
        return total * number / 100
    for suffix, scale in [('Ki', 1024), ('Mi', 1024**2), ('Gi', 1024**3), ('Ti', 1024**4)]:
        if text.endswith(suffix):
            return float(text[:-len(suffix)]) * scale
    return float(text)


def storage_facts(data_dir: str | Path, *, image_gib: float | None = None,
                  kubelet_config: str | Path | None = None, statvfs=os.statvfs) -> dict:
    """Report known facts and uncertainty without installing or creating paths.

    Image headroom is an explicit operator estimate of pull plus extraction,
    not compressed image size. Thresholds apply only when the chosen filesystem
    is the corresponding nodefs/imagefs; v1 reports both candidates separately.
    """
    path = Path(data_dir).expanduser()
    if not path.is_absolute():
        raise ValueError('K3s data directory must be absolute')
    if image_gib is not None and (not math.isfinite(image_gib) or image_gib < 0):
        raise ValueError('Image headroom GiB must be finite and nonnegative')
    existing = path
    while not existing.exists():
        existing = existing.parent
    stats = statvfs(existing)
    total = stats.f_blocks * stats.f_frsize
    free = stats.f_bavail * stats.f_frsize
    inodes = stats.f_favail
    config, warnings = {}, []
    if kubelet_config:
        config = yaml.safe_load(Path(kubelet_config).expanduser().read_text()) or {}
        if not isinstance(config, dict):
            raise ValueError('Kubelet configuration must be a mapping')
    else:
        warnings.append('Kubelet eviction/reclaim thresholds are unknown; provide --kubelet-config for threshold headroom')
    thresholds = {}
    for signal in ['nodefs.available', 'imagefs.available']:
        hard = (config.get('evictionHard') or {}).get(signal)
        reclaim = (config.get('evictionMinimumReclaim') or {}).get(signal, '0')
        if hard is None:
            continue
        try:
            hard_bytes, reclaim_bytes = _bytes(hard, total), _bytes(reclaim, total)
            if not all(math.isfinite(n) and n >= 0 for n in [hard_bytes, reclaim_bytes]):
                raise ValueError('invalid threshold')
        except ValueError:
            warnings.append(f'Cannot interpret {signal} threshold; headroom unknown')
            continue
        target = hard_bytes + reclaim_bytes
        headroom = free - target
        after = headroom - (image_gib or 0) * 1024**3
        thresholds[signal] = {'hard_bytes': hard_bytes, 'minimum_reclaim_bytes': reclaim_bytes,
                              'recovery_target_bytes': target, 'headroom_bytes': headroom,
                              'headroom_after_image_bytes': after if image_gib is not None else None}
        if after < 0:
            warnings.append(f'{signal}: proposed image pull/extraction or current storage crosses recovery headroom (if this filesystem supplies {signal.split(".")[0]})')
    if inodes == 0 and stats.f_files:
        warnings.append('No free inodes on proposed data filesystem')
    if image_gib is not None and free < image_gib * 1024**3:
        warnings.append('Image pull/extraction estimate exceeds available bytes')
    return {'data_dir': str(path), 'existing_ancestor': str(existing),
            'filesystem_device': os.stat(existing).st_dev, 'total_bytes': total,
            'available_bytes': free, 'available_inodes': inodes,
            'image_headroom_gib': image_gib, 'thresholds': thresholds, 'warnings': warnings}
