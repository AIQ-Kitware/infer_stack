"""Generic launch fields for a vLLM endpoint's container.

A model whose image wraps vLLM in its own launcher is described in catalog
data, not in Python. The fields, all under ``runtime`` and all optional:

``command``
    The container command, replacing the stock ``MODEL --host --port <vLLM
    flags>``. With it set, infer-stack passes the engine nothing else: the
    image's launcher reads what it needs from ``command`` and ``env``.
``env``
    Container environment. Values are written as strings: numbers as
    written, booleans as ``true``/``false``. A ``$`` is literal (never Compose
    interpolation, so the managed ``.env`` secrets cannot leak in).
``mounts``
    ``{container path: subdirectory}`` persisted under infer-stack's runtime
    data directory, for an image that keeps prepared weights or caches
    outside vLLM's usual ``/root/.cache``. With ``command`` set they are the
    only mounts (such an image does not use vLLM's caches); on the stock
    command they are added to them.
``extra_args``
    Extra vLLM flags appended to the stock command (after infer-stack's own,
    so a repeated flag takes the extra value). Not allowed with ``command``.

``command`` and ``env`` values may use ``{max_model_len}``,
``{gpu_memory_utilization}``, ``{served_model_name}`` and ``{port}``. Those
are values infer-stack itself relies on -- capacity, routing, the port it
probes -- so a launcher that takes them through its own variables stays in
step when the endpoint is edited. Other braces are left alone.

All four are deployment identity: two endpoints that launch differently
never share a process.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

#: The fields this module owns, in the order they are documented.
LAUNCH_FIELDS = ('command', 'env', 'mounts', 'extra_args')

#: The port every engine container serves on (what routing and probes use).
ENGINE_PORT = 8000

#: Environment infer-stack sets or relies on itself. An endpoint may not set
#: these: the token is a managed secret, the attention backend has its own
#: field, and GPU visibility is decided by placement.
RESERVED_ENV = frozenset({
    'HF_TOKEN', 'VLLM_ATTENTION_BACKEND', 'CUDA_VISIBLE_DEVICES',
    'NVIDIA_VISIBLE_DEVICES',
})

#: vLLM flags infer-stack renders from fields it also acts on (routing, GPU
#: count, capacity). Repeating one in ``extra_args`` would make the process
#: disagree with the ledger, so it is refused rather than left to last-wins.
IDENTITY_FLAGS = frozenset({
    '--served-model-name', '--tensor-parallel-size', '-tp',
    '--pipeline-parallel-size', '-pp', '--data-parallel-size', '-dp',
    '--max-model-len',
})

_TEMPLATE = re.compile(r'\{(max_model_len|gpu_memory_utilization|served_model_name|port)\}')
_ENV_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

#: The one named recipe that existed before these fields, translated where a
#: catalog is read. Nothing new should produce ``serve_recipe``.
LEGACY_RECIPES = {
    'hyperqwen-3090-single': {
        'command': ['single'],
        'env': {
            'PORT': '{port}',
            'SPEC': 'dflash2',
            'CTX': 'fast',
            'MAX_LEN': '{max_model_len}',
            'GPU_UTIL': '{gpu_memory_utilization}',
            'EXTRA_ARGS': '--served-model-name={served_model_name}',
        },
        'mounts': {
            '/app/models': 'hyperqwen/qwen3.8-27b/models',
            '/cache': 'hyperqwen/qwen3.8-27b/cache',
        },
    },
}


def translate_legacy(runtime: dict[str, Any]) -> dict[str, Any]:
    """``runtime`` with a known ``serve_recipe`` replaced by the generic fields.

    Fields the catalog already sets win. ``PREFIX_CACHE`` follows
    ``enable_prefix_caching`` exactly as the recipe did. An unknown recipe is
    left in place for :func:`launch_errors` to report.
    """
    recipe = runtime.get('serve_recipe')
    if recipe not in LEGACY_RECIPES:
        return runtime
    out = {k: v for k, v in runtime.items() if k != 'serve_recipe'}
    for key, value in LEGACY_RECIPES[recipe].items():
        if key not in out:
            out[key] = dict(value) if isinstance(value, dict) else list(value)
    if 'env' in out and 'PREFIX_CACHE' not in out['env']:
        out['env'] = {**out['env'],
                      'PREFIX_CACHE': '1' if runtime.get('enable_prefix_caching') else '0'}
    return out


def launch_errors(name: str, engine: str, runtime: dict[str, Any]) -> list[str]:
    """Problems with an endpoint's launch fields, as catalog error strings."""
    where = f"endpoint '{name}'"
    errors: list[str] = []
    if 'serve_recipe' in runtime:
        errors.append(
            f"{where}: unknown runtime.serve_recipe {runtime['serve_recipe']!r}; "
            'describe the launch with runtime.command / env / mounts instead')
    present = [k for k in LAUNCH_FIELDS if runtime.get(k)]
    if present and engine != 'vllm':
        return errors + [f"{where}: runtime.{present[0]} is only supported on vllm "
                         f"endpoints (engine is '{engine}')"]
    command = runtime.get('command')
    if command is not None and (not isinstance(command, list) or not command
                                or not all(isinstance(c, (str, int, float)) for c in command)):
        errors.append(f'{where}: runtime.command must be a non-empty list of strings')
    env = runtime.get('env')
    if env is not None:
        if not isinstance(env, dict):
            errors.append(f'{where}: runtime.env must be a mapping of NAME: value')
        else:
            for key, value in env.items():
                if not _ENV_NAME.match(str(key)):
                    errors.append(f'{where}: runtime.env name {key!r} is not a valid '
                                  'environment variable name')
                elif key in RESERVED_ENV:
                    errors.append(f'{where}: runtime.env may not set {key} '
                                  '(infer-stack manages it)')
                if isinstance(value, (dict, list)) or value is None:
                    errors.append(f'{where}: runtime.env.{key} must be a string, '
                                  'number or boolean')
    mounts = runtime.get('mounts')
    if mounts is not None:
        if not isinstance(mounts, dict):
            errors.append(f'{where}: runtime.mounts must map a container path to '
                          'a subdirectory')
        else:
            for target, sub in mounts.items():
                sub_path = PurePosixPath(str(sub))
                if not str(target).startswith('/'):
                    errors.append(f'{where}: runtime.mounts container path {target!r} '
                                  'must be absolute')
                if sub_path.is_absolute() or '..' in sub_path.parts or not str(sub):
                    errors.append(f'{where}: runtime.mounts.{target} must be a '
                                  'subdirectory of the runtime data dir, not '
                                  f'{sub!r}')
    extra = runtime.get('extra_args') or []
    if extra and command:
        errors.append(f'{where}: runtime.extra_args apply to the stock vLLM command; '
                      "with runtime.command, pass flags through the image's launcher")
    for arg in extra if isinstance(extra, list) else []:
        flag = str(arg).split('=', 1)[0]
        if flag in IDENTITY_FLAGS:
            errors.append(f'{where}: runtime.extra_args repeats {flag}, which '
                          'infer-stack sets from its own field (use that field)')
    return errors


def launch_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    """The launch fields that are set, for the deployment's structural key.

    Unset fields are omitted, so an endpoint using none of them keeps the
    compatibility key it had before they existed.
    """
    out: dict[str, Any] = {}
    for key in LAUNCH_FIELDS:
        value = runtime.get(key)
        if value:
            out[key] = ({str(k): env_string(v) for k, v in value.items()}
                        if isinstance(value, dict) else [str(v) for v in value])
    return out


def env_string(value: Any) -> str:
    """How an env value is written: ``true``/``false`` for booleans, else ``str``."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def fill(value: str, service: dict[str, Any]) -> str:
    """Substitute the documented ``{...}`` names in ``value``.

    Example:
        >>> fill('--served-model-name={served_model_name} {x}', {'served_model_name': 'm'})
        '--served-model-name=m {x}'
    """
    values = {
        'max_model_len': service.get('max_model_len'),
        'gpu_memory_utilization': service.get('gpu_memory_utilization'),
        'served_model_name': service.get('served_model_name'),
        'port': ENGINE_PORT,
    }
    return _TEMPLATE.sub(lambda m: str(values[m.group(1)]), str(value))
