"""Configured serving facts for measurement identity, without credential values.

These describe the launch specification, not an attestation of loaded weights.
Catalog resolution and lease descriptors use the same projection; the latter
projects the deployment actually assigned, including any coalesced capacity.
"""
from __future__ import annotations

import hashlib
import json
import re

from .endpoints import ExternalTarget
from .launch import VLLM_DEFAULTS

SCHEMA = 'infer-stack-serving-provenance/1'


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return 'sha256:' + hashlib.sha256(encoded.encode()).hexdigest()


def _envelope(payload, reasons):
    payload = {'schema': SCHEMA, **payload}
    return {'schema': SCHEMA, 'payload': payload, 'digest': _digest(payload),
            'immutable': not reasons, 'unknown_reasons': reasons}


def managed_provenance(engine, spec, served):
    """Project either a resolved request or its assigned deployment.

    Hash all launch options conservatively, including environment and custom
    commands, instead of publishing them in logs. Placement, reclaim policy,
    aliases, ports and credentials used by the gateway are not model identity.
    """
    runtime = {**(VLLM_DEFAULTS if engine == 'vllm' else {}),
               **spec.get('runtime', {})}
    revision = spec.get('revision')
    image = runtime.get('image')
    reasons = []
    if engine != 'vllm':
        reasons.append('serving engine has no immutable model provenance projection')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', str(revision or '')):
        reasons.append('serving model revision is not a pinned commit')
    if not re.search(r'@sha256:[0-9a-fA-F]{64}$', str(image or '')):
        reasons.append('serving image is not pinned by digest')
    if runtime.get('command') or runtime.get('mounts'):
        reasons.append('custom serving command or mounts may supply mutable artifacts')
    if runtime.get('chat_template') or runtime.get('lora_adapters'):
        reasons.append('serving template or adapter artifacts are not content-pinned')
    args = runtime.get('extra_args') or []
    if any(str(a).split('=')[0] in {
        '--model', '--revision', '--tokenizer', '--tokenizer-revision',
        '--chat-template', '--lora-modules', '--code-revision',
    } for a in args):
        reasons.append('extra serving arguments override model artifacts')
    payload = {
        'engine': engine,
        'model_ref': spec.get('hf_model_id'),
        'revision': revision,
        'quantization': spec.get('quantization'),
        'dtype': spec.get('dtype'),
        'image': image,
        'protocol': served.get('protocol', 'chat'),
        'runtime_digest': _digest(runtime),
    }
    return _envelope(payload, reasons)


def endpoint_provenance(endpoint):
    """Resolve facts without acquiring a lease or downloading model artifacts."""
    if isinstance(endpoint.target, ExternalTarget):
        return _envelope({'engine': 'external', 'model_ref': endpoint.target.model,
                          'protocol': endpoint.protocol},
                         ['external serving weights and runtime are not attested'])
    request = endpoint.to_request()
    return managed_provenance(request.engine, request.spec, request.served)
