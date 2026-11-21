"""Administrator-owned labels and explicit selector profiles for shared pools."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from ..backends.kubeai import GPU_PRODUCT_LABEL, node_gpus
from ..leasing.cluster_placement import HOSTNAME_LABEL, POOL_LABEL

LABELS = {'pool': POOL_LABEL, 'machine_class': 'infer-stack.ai/class', 'site': 'infer-stack.ai/site'}


def labels(*, pool=None, machine_class=None, site=None) -> dict[str, str]:
    out = {}
    for key, value in [('pool', pool), ('machine_class', machine_class), ('site', site)]:
        if value is None:
            continue
        if not isinstance(value, str) or len(value) > 63 or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?', value):
            raise ValueError(f'{key} must be a nonempty Kubernetes label value')
        out[LABELS[key]] = value
    return out


def profiles(nodes: list[dict]) -> dict:
    """Node and pool/product profiles; never choose an actual GPU allocation."""
    out = {}
    for name, facts in sorted(node_gpus(nodes).items()):
        if not facts['count']:
            continue
        selectors = []
        hostname = facts['labels'].get(HOSTNAME_LABEL)
        pool = facts['labels'].get(POOL_LABEL)
        if hostname:
            selectors.append({HOSTNAME_LABEL: hostname})
            if pool:
                selectors.append({HOSTNAME_LABEL: hostname, POOL_LABEL: pool})
        if pool and facts['product']:
            selectors.append({POOL_LABEL: pool, GPU_PRODUCT_LABEL: facts['product']})
        for selector in selectors:
            digest = hashlib.sha256(json.dumps(selector, sort_keys=True).encode()).hexdigest()[:12]
            out[f'infer-stack-placement-{digest}'] = {
                'imageName': 'nvidia-gpu', 'runtimeClassName': 'nvidia',
                'requests': {'nvidia.com/gpu': '1'}, 'limits': {'nvidia.com/gpu': '1'},
                'nodeSelector': copy.deepcopy(selector)}
    return out
