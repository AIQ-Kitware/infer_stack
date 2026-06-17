"""The endpoint descriptor + env-file a lease hands to a job.

This is the standard "here is the endpoint and the models" artifact the
inference-endpoint-injection brainstorm asked for, aligned with the
``contracts.py`` serving-profile-contract (same ``base_url`` / ``api_key_env`` /
``models`` shape). ``acquire`` / ``run`` emit it so a pipeline node sources one
file and gets a ready OpenAI-compatible client, instead of every team inventing
its own env-var contract.

For now ``base_url`` is supplied by the caller (the dry-run/null backend has no
real front door). Once the Compose/KubeAI backends land they will report the
realized base URL, and this same descriptor will carry it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import DeploymentGroup, Lease

SESSION_ENV = 'INFER_STACK_SESSION_ID'


def _request_model_name(payload: dict[str, Any], endpoint: str) -> str:
    """The name a client must pass as ``model`` to reach this endpoint."""
    return (
        payload.get('served_model_name')
        or payload.get('model')
        or endpoint
    )


def _endpoint_var(endpoint: str) -> str:
    slug = re.sub(r'[^A-Z0-9]+', '_', endpoint.upper()).strip('_')
    return f'INFER_STACK_ENDPOINT_{slug}'


def build_descriptor(
    lease: Lease,
    groups: list[DeploymentGroup],
    *,
    base_url: str,
    api_key_env: str = 'LITELLM_MASTER_KEY',
    api_key: str | None = None,
    request_names: dict[str, str] | None = None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Build the endpoint descriptor for one lease.

    Only the endpoints this lease actually requested are included (a coalesced
    group may serve more). ``request_names`` overrides the model name a client
    must request per endpoint — e.g. behind a LiteLLM front door the client asks
    for the endpoint *alias*, not the upstream served name.
    """
    endpoints: dict[str, str] = {}
    for group in groups:
        for endpoint, payload in group.served.items():
            if endpoint in lease.endpoints:
                if request_names and endpoint in request_names:
                    endpoints[endpoint] = request_names[endpoint]
                else:
                    endpoints[endpoint] = _request_model_name(payload, endpoint)
    # preserve the lease's requested order where possible
    ordered = {ep: endpoints[ep] for ep in lease.endpoints if ep in endpoints}
    descriptor: dict[str, Any] = {
        'schema_version': 1,
        'kind': 'infer-stack-endpoint',
        'session_id': lease.id,
        'base_url': base_url,
        'api_key_env': api_key_env,
        'protocol': 'openai',
        'endpoints': ordered,
        'models': list(ordered.values()),
    }
    if api_key:
        descriptor['api_key'] = api_key
    if cuda_visible_devices is not None:
        descriptor['cuda_visible_devices'] = cuda_visible_devices
    return descriptor


def descriptor_env(descriptor: dict[str, Any]) -> dict[str, str]:
    """Flatten a descriptor into the env vars a job should see."""
    env: dict[str, str] = {SESSION_ENV: descriptor['session_id']}
    if descriptor.get('base_url'):
        env['OPENAI_BASE_URL'] = descriptor['base_url']
    if descriptor.get('api_key'):
        # The actual key, so `source <env-file>` configures an OpenAI client
        # outright (OPENAI_BASE_URL + OPENAI_API_KEY) — no manual export.
        env['OPENAI_API_KEY'] = descriptor['api_key']
    if descriptor.get('api_key_env'):
        env['INFER_STACK_API_KEY_ENV'] = descriptor['api_key_env']
    for endpoint, model in descriptor.get('endpoints', {}).items():
        env[_endpoint_var(endpoint)] = model
    if descriptor.get('models'):
        env['INFER_STACK_MODELS'] = ','.join(descriptor['models'])
    if descriptor.get('cuda_visible_devices') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(descriptor['cuda_visible_devices'])
    return env


def render_env_file(descriptor: dict[str, Any]) -> str:
    """Render the descriptor as a sourceable shell env-file.

    Example:
        >>> from infer_stack.leasing.models import Lease, DeploymentGroup, GroupState
        >>> lease = Lease('sess-1', 'me', 'active', 0.0, None, None, 0.0,
        ...     endpoints=['qwen-coder'])
        >>> grp = DeploymentGroup('g1', 'ck', 'vllm', 'shared-compatible', {},
        ...     {}, {'qwen-coder': {'served_model_name': 'qwen-coder'}},
        ...     GroupState.LIVE, 0.0, 0.0)
        >>> d = build_descriptor(lease, [grp], base_url='http://h:1/v1')
        >>> print(render_env_file(d), end='')
        export INFER_STACK_SESSION_ID=sess-1
        export OPENAI_BASE_URL=http://h:1/v1
        export INFER_STACK_API_KEY_ENV=LITELLM_MASTER_KEY
        export INFER_STACK_ENDPOINT_QWEN_CODER=qwen-coder
        export INFER_STACK_MODELS=qwen-coder
    """
    return ''.join(
        f'export {key}={value}\n'
        for key, value in descriptor_env(descriptor).items()
    )


def read_session_id(env_file: str | Path) -> str | None:
    """Recover the session id from a previously written env-file."""
    prefix = f'export {SESSION_ENV}='
    for line in Path(env_file).expanduser().read_text().splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip().strip('"').strip("'")
        if line.startswith(f'{SESSION_ENV}='):
            return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None
