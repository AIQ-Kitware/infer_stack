from __future__ import annotations

from typing import Any

#: ``runtime.simulator.kind`` values we know how to render a command for.
SIMULATOR_KINDS = ('llm-d-sim',)

#: Keys of ``runtime.simulator`` that select/describe the simulator rather than
#: naming one of its flags.
_SIM_META_KEYS = ('kind', 'model', 'extra_args')


def simulator_args(service: dict[str, Any]) -> list[str]:
    """Render the argv for a simulator image standing in for vLLM.

    ``llm-d-inference-sim`` speaks the OpenAI/vLLM *API* but not vLLM's *CLI*:
    it has no positional model argument, no ``--host``, and no
    ``--tensor-parallel-size`` / ``--gpu-memory-utilization`` (it rejects
    unknown flags outright).  So a simulator deployment cannot reuse
    :func:`vllm_args`; it gets its own renderer, driven by the same endpoint
    fields so the catalog entry still reads like the real one.

    ``--model`` deliberately defaults to the *served* name, not the HF repo id.
    The simulator treats a real repo id as a request for real HuggingFace
    tokenization, which it delegates to a separate render service and dies
    without.  A name that is not a repo id makes it use its built-in simulated
    tokenizer, which is the whole point of running it GPU-less.

    Args:
        service: the dict :func:`vllm_args` consumes, with a ``simulator``
            block (see :func:`~infer_stack.leasing.compose.vllm_service_dict`).

    Returns:
        The container command, minus the image's own entrypoint.

    Example:
        >>> print('\\n'.join(simulator_args({
        ...     'served_model_name': 'mock-smol',
        ...     'max_model_len': 2048,
        ...     'max_num_seqs': 4,
        ...     'simulator': {'kind': 'llm-d-sim', 'mode': 'random',
        ...                   'startup_duration': '10s'},
        ... })))
        --model
        mock-smol
        --port
        8000
        --served-model-name=mock-smol
        --max-model-len=2048
        --max-num-seqs=4
        --mode
        random
        --startup-duration
        10s
    """
    sim = dict(service.get('simulator') or {})
    kind = sim.get('kind', 'llm-d-sim')
    if kind not in SIMULATOR_KINDS:
        raise ValueError(
            f'unknown runtime.simulator.kind {kind!r}; '
            f'known kinds: {", ".join(SIMULATOR_KINDS)}'
        )
    served = service['served_model_name']
    args = [
        '--model', str(sim.get('model') or served),
        '--port', '8000',
        f'--served-model-name={served}',
    ]
    if service.get('max_model_len') is not None:
        args.append(f'--max-model-len={service["max_model_len"]}')
    if service.get('max_num_seqs') is not None:
        args.append(f'--max-num-seqs={service["max_num_seqs"]}')
    # Everything else in the block is a simulator flag verbatim, so any knob
    # the simulator grows (latency profiles, failure injection, LoRA lifecycle)
    # is reachable from the catalog without a change here.  snake_case is
    # accepted because the rest of the runtime block uses it.
    for key in sorted(k for k in sim if k not in _SIM_META_KEYS):
        value = sim[key]
        flag = '--' + str(key).replace('_', '-')
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.extend([flag, *(str(v) for v in value)])
        elif value is not None:
            args.extend([flag, str(value)])
    args.extend(str(a) for a in (sim.get('extra_args') or []))
    return args


def vllm_args(service: dict[str, Any]) -> list[str]:
    args = [
        f'--served-model-name={service["served_model_name"]}',
        f'--tensor-parallel-size={service["tensor_parallel_size"]}',
        # .get: older KubeAI lock data predates these keys.
        f'--pipeline-parallel-size={service.get("pipeline_parallel_size", 1)}',
        f'--data-parallel-size={service["data_parallel_size"]}',
        f'--max-model-len={service["max_model_len"]}',
        f'--gpu-memory-utilization={service["gpu_memory_utilization"]}',
        f'--max-num-batched-tokens={service["max_num_batched_tokens"]}',
        f'--max-num-seqs={service["max_num_seqs"]}',
    ]
    # Optional model-serving knobs: emitted only when set, so a knob-less
    # service keeps vLLM's own defaults (revision=main, dtype=auto, ...).
    if service.get('revision'):
        args.append(f'--revision={service["revision"]}')
    if service.get('quantization'):
        args.append(f'--quantization={service["quantization"]}')
    if service.get('dtype'):
        args.append(f'--dtype={service["dtype"]}')
    if service.get('chat_template'):
        args.append(f'--chat-template={service["chat_template"]}')
    if service.get('trust_remote_code'):
        args.append('--trust-remote-code')
    if service.get('enable_prefix_caching'):
        args.append('--enable-prefix-caching')
    if service.get('enable_auto_tool_choice'):
        args.append('--enable-auto-tool-choice')
        if service.get('tool_call_parser'):
            args.append(f'--tool-call-parser={service["tool_call_parser"]}')
    args.extend(service.get('extra_args', []))
    return args


def default_base_url(
    deployment: dict[str, Any], *, explicit: str | None = None
) -> str:
    if explicit:
        return explicit.rstrip('/')
    access = deployment.get('access', {}).get('default') or {}
    if access.get('base_url'):
        return str(access['base_url']).rstrip('/')
    backend = deployment.get('backend', 'compose')
    if backend == 'kubeai':
        ingress = deployment.get('cluster', {}).get('ingress', {}) or {}
        host = ingress.get('host', '')
        if ingress.get('enabled') and host:
            return f'http://{host}/openai/v1'
        return 'http://127.0.0.1:8000/openai/v1'
    if (deployment.get('gateways', {}).get('litellm') or {}).get('enabled'):
        return f'http://127.0.0.1:{deployment.get("ports", {}).get("litellm", 14042)}/v1'
    if (deployment.get('providers', {}).get('ollama') or {}).get('enabled'):
        return f'http://127.0.0.1:{deployment.get("ports", {}).get("ollama", 11434)}'
    return f'http://127.0.0.1:{deployment.get("ports", {}).get("litellm", 14042)}/v1'
