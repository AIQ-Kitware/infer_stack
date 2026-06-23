from __future__ import annotations

from typing import Any


def vllm_args(service: dict[str, Any]) -> list[str]:
    args = [
        f'--served-model-name={service["served_model_name"]}',
        f'--tensor-parallel-size={service["tensor_parallel_size"]}',
        f'--data-parallel-size={service["data_parallel_size"]}',
        f'--max-model-len={service["max_model_len"]}',
        f'--gpu-memory-utilization={service["gpu_memory_utilization"]}',
        f'--max-num-batched-tokens={service["max_num_batched_tokens"]}',
        f'--max-num-seqs={service["max_num_seqs"]}',
    ]
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
