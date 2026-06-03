from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Pattern, Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore


GiB = 1024**3


# --- model presets ---------------------------------------------------------
#
# Keep the presets we've been interested in, but use if 0 / if 1 gates so it
# is obvious which ones are active by default right now.

DEFAULT_PRESET_REPO_IDS: list[str] = []

if 1:
    DEFAULT_PRESET_REPO_IDS += [
        'Qwen/Qwen3.5-122B-A10B',
        'Qwen/Qwen3.5-122B-A10B-FP8',
        # Official integer-quantized preset currently available on HF.
        'Qwen/Qwen3.5-122B-A10B-GPTQ-Int4',
    ]

if 0:
    DEFAULT_PRESET_REPO_IDS += [
        'Qwen/Qwen3.5-2B',
        'Qwen/Qwen3.5-9B',
        'Qwen/Qwen3.5-27B',
        'Qwen/Qwen3.5-35B-A3B',
    ]

if 0:
    DEFAULT_PRESET_REPO_IDS += [
        'Qwen/Qwen3.6-30B-A3B-Thinking-2507',
        'Qwen/Qwen3.6-235B-A22B-Thinking-2507',
        'google/gemma-4-27b-it',
        'google/gemma-4-9b-it',
    ]


@dataclass(frozen=True)
class ModelDiscoveryRule:
    author: str
    search: str
    include: Pattern[str]


# Keep discovery rules around for optional no-hardcoded discovery workflows.
DEFAULT_MODEL_DISCOVERY_RULES: Final[tuple[ModelDiscoveryRule, ...]] = (
    ModelDiscoveryRule(
        author='Qwen',
        search='Qwen3.5',
        include=re.compile(r'^Qwen3\.5-.*$', re.IGNORECASE),
    ),
    ModelDiscoveryRule(
        author='Qwen',
        search='Qwen3.6',
        include=re.compile(r'^Qwen3\.6-.*$', re.IGNORECASE),
    ),
    ModelDiscoveryRule(
        author='google',
        search='gemma-4',
        include=re.compile(r'^gemma-4-.*$', re.IGNORECASE),
    ),
)
DEFAULT_MODEL_DISCOVERY_LIMIT = 100


FIT_STYLES: Final[dict[str, str]] = {
    'yes': 'bold green',
    'no': 'bold red',
    'maybe': 'bold yellow',
}


def fit_status_markup(status: Literal['yes', 'no', 'maybe']) -> str:
    style = FIT_STYLES[status]
    return f'[{style}]{status}[/{style}]'


def gib(x: float) -> float:
    return x / GiB


def gib_text(x: float | None) -> str:
    if x is None:
        return '-'
    return f'{gib(x):.2f}'


def range_text(lo: float, hi: float) -> str:
    return f'{gib(lo):.2f}–{gib(hi):.2f}'


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    bytes_per_element: float
    notes: str = ''


DTYPE_F16 = DTypeSpec('fp16/bf16', 2.0, 'Standard half precision')
DTYPE_FP8 = DTypeSpec('fp8', 1.0, 'Approximate FP8 storage')
DTYPE_F32 = DTypeSpec('fp32', 4.0, 'Single precision')


@dataclass(frozen=True)
class WeightFootprint:
    total_bytes: int | None
    source: str
    notes: str = ''


@dataclass(frozen=True)
class CacheGroupSpec:
    name: str
    kind: Literal['full_kv', 'sliding_kv', 'linear_recurrent', 'linear_conv']
    layer_type: str
    total_layers: int
    unique_cache_layers: int
    num_heads: int | None = None
    head_dim: int | None = None
    seq_len_mode: Literal['full', 'sliding', 'fixed'] = 'full'
    sliding_window: int | None = None
    fixed_elements_per_sequence: int | None = None
    kv_copies: int = 2
    dtype_source: Literal['kv', 'linear_state'] = 'kv'
    notes: str = ''

    def request_floor_elements_total(
        self, deployment: 'DeploymentSpec'
    ) -> float:
        batch = deployment.concurrent_sequences
        if self.kind in {'full_kv', 'sliding_kv'}:
            if self.num_heads is None or self.head_dim is None:
                raise ValueError(
                    f'Missing num_heads/head_dim for cache group {self.name}'
                )
            seq_len = deployment.total_sequence_tokens
            if self.seq_len_mode == 'sliding':
                if self.sliding_window is None:
                    raise ValueError(f'Sliding window not set for {self.name}')
                seq_len = min(seq_len, self.sliding_window)
            return (
                self.kv_copies
                * self.unique_cache_layers
                * batch
                * seq_len
                * self.num_heads
                * self.head_dim
            )
        if self.fixed_elements_per_sequence is None:
            raise ValueError(
                f'Missing fixed_elements_per_sequence for cache group {self.name}'
            )
        return (
            self.unique_cache_layers * batch * self.fixed_elements_per_sequence
        )

    def bytes_per_request_cluster(self, deployment: 'DeploymentSpec') -> float:
        dtype = (
            deployment.kv_cache_dtype
            if self.dtype_source == 'kv'
            else deployment.linear_state_dtype
        )
        return (
            self.request_floor_elements_total(deployment)
            * dtype.bytes_per_element
        )

    def single_sequence_token_slope_cluster(
        self, deployment: 'DeploymentSpec'
    ) -> float:
        dtype = (
            deployment.kv_cache_dtype
            if self.dtype_source == 'kv'
            else deployment.linear_state_dtype
        )
        if self.kind in {'full_kv', 'sliding_kv'}:
            if self.num_heads is None or self.head_dim is None:
                raise ValueError(
                    f'Missing num_heads/head_dim for cache group {self.name}'
                )
            return (
                self.kv_copies
                * self.unique_cache_layers
                * self.num_heads
                * self.head_dim
                * dtype.bytes_per_element
            )
        return 0.0

    def single_sequence_fixed_bytes_cluster(
        self, deployment: 'DeploymentSpec'
    ) -> float:
        dtype = (
            deployment.kv_cache_dtype
            if self.dtype_source == 'kv'
            else deployment.linear_state_dtype
        )
        if self.kind in {'full_kv', 'sliding_kv'}:
            return 0.0
        if self.fixed_elements_per_sequence is None:
            raise ValueError(
                f'Missing fixed_elements_per_sequence for cache group {self.name}'
            )
        return (
            self.unique_cache_layers
            * self.fixed_elements_per_sequence
            * dtype.bytes_per_element
        )


@dataclass(frozen=True)
class ModelMemorySpec:
    repo_id: str
    family: Literal['qwen3.5', 'qwen3.6', 'gemma4']
    architecture: str
    max_position_embeddings: int
    text_hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    weight_footprint: WeightFootprint
    cache_groups: tuple[CacheGroupSpec, ...]
    has_vision: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StartupOverheadInterval:
    low_bytes_per_gpu: float
    high_bytes_per_gpu: float
    notes: str = ''


@dataclass(frozen=True)
class StartupOverheadPolicy:
    base_runtime_gib_low: float = 2.25
    base_runtime_gib_high: float = 3.25
    compile_graph_gib_low: float = 1.00
    compile_graph_gib_high: float = 1.75
    warmup_workspace_gib_low: float = 0.75
    warmup_workspace_gib_high: float = 1.25
    hidden_size_scale_low_gib_per_8k: float = 0.25
    hidden_size_scale_high_gib_per_8k: float = 0.50
    multimodal_margin_gib_low: float = 1.50
    multimodal_margin_gib_high: float = 3.50

    def estimate(
        self, model: ModelMemorySpec, deployment: 'DeploymentSpec'
    ) -> StartupOverheadInterval:
        size_ratio = model.text_hidden_size / 8192.0
        low = (
            self.base_runtime_gib_low
            + self.compile_graph_gib_low
            + self.warmup_workspace_gib_low
            + self.hidden_size_scale_low_gib_per_8k * size_ratio
        ) * GiB
        high = (
            self.base_runtime_gib_high
            + self.compile_graph_gib_high
            + self.warmup_workspace_gib_high
            + self.hidden_size_scale_high_gib_per_8k * size_ratio
        ) * GiB
        notes = [
            'Deterministic startup-overhead interval covering runtime resident allocations, compile/cudagraph setup, and warmup workspaces.'
        ]
        if model.has_vision and not deployment.language_model_only:
            low += self.multimodal_margin_gib_low * GiB
            high += self.multimodal_margin_gib_high * GiB
            notes.append(
                'Includes multimodal resident-memory margin because language_model_only is disabled.'
            )
        elif model.has_vision and deployment.language_model_only:
            notes.append(
                'Multimodal resident-memory margin omitted because language_model_only is enabled.'
            )
        return StartupOverheadInterval(low, high, ' '.join(notes))


DEFAULT_STARTUP_OVERHEAD_POLICY = StartupOverheadPolicy()


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    tensor_parallel_size: int
    data_parallel_size: int = 1
    concurrent_sequences: int = 1
    prompt_text_tokens: int = 8192
    media_soft_tokens: int = 0
    max_new_tokens: int = 0
    kv_cache_dtype: DTypeSpec = DTYPE_F16
    linear_state_dtype: DTypeSpec = DTYPE_F16
    gpu_memory_bytes: int | None = None
    gpu_memory_utilization: float = 0.95
    language_model_only: bool = True
    startup_overhead_policy: StartupOverheadPolicy = (
        DEFAULT_STARTUP_OVERHEAD_POLICY
    )

    @property
    def total_sequence_tokens(self) -> int:
        return (
            self.prompt_text_tokens
            + self.media_soft_tokens
            + self.max_new_tokens
        )

    @property
    def managed_budget_bytes_per_gpu(self) -> float | None:
        if self.gpu_memory_bytes is None:
            return None
        return self.gpu_memory_bytes * self.gpu_memory_utilization

    @property
    def gpu_memory_gib(self) -> float | None:
        if self.gpu_memory_bytes is None:
            return None
        return self.gpu_memory_bytes / GiB


@dataclass(frozen=True)
class RequestFloor:
    total_bytes_per_gpu: float
    token_slope_bytes_per_gpu: float
    fixed_bytes_per_gpu: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StartupFit:
    weight_bytes_per_gpu: float | None
    overhead_low_bytes_per_gpu: float
    overhead_high_bytes_per_gpu: float
    used_low_bytes_per_gpu: float | None
    used_high_bytes_per_gpu: float | None
    margin_low_bytes_per_gpu: float | None
    margin_high_bytes_per_gpu: float | None
    status: Literal['yes', 'no', 'maybe']


@dataclass(frozen=True)
class CapacityEstimate:
    kv_budget_low_bytes_per_gpu: float | None
    kv_budget_high_bytes_per_gpu: float | None
    kv_tokens_low_cluster: float | None
    kv_tokens_high_cluster: float | None
    max_concurrency_low: float | None
    max_concurrency_high: float | None
    fits_target_low: bool | None
    fits_target_high: bool | None


@dataclass(frozen=True)
class MemoryEstimate:
    model: ModelMemorySpec
    deployment: DeploymentSpec
    request_floor: RequestFloor
    startup_fit: StartupFit
    capacity: CapacityEstimate


def _load_json(
    repo_id: str, filename: str, token: str | None = None
) -> dict[str, Any]:
    if hf_hub_download is None:
        raise RuntimeError('huggingface_hub is required to fetch configs')
    path = hf_hub_download(repo_id, filename=filename, token=token)
    return json.loads(Path(path).read_text())


def _model_info(
    repo_id: str, token: str | None = None
) -> tuple[Any | None, Any | None]:
    if HfApi is None:
        return None, None
    api = HfApi(token=token)
    info_expand = None
    info_files = None
    try:
        info_expand = api.model_info(
            repo_id,
            expand=[
                'config',
                'safetensors',
                'siblings',
                'tags',
                'pipeline_tag',
                'createdAt',
                'lastModified',
            ],
        )
    except Exception:
        try:
            info_expand = api.model_info(repo_id)
        except Exception:
            info_expand = None
    try:
        info_files = api.model_info(repo_id, files_metadata=True)
    except Exception:
        info_files = None
    return info_expand, info_files


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    data = getattr(obj, '__dict__', None)
    if isinstance(data, dict):
        return dict(data)
    out: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith('_'):
            continue
        value = getattr(obj, name)
        if callable(value):
            continue
        out[name] = value
    return out


def _listify(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _weight_footprint_from_hf(
    repo_id: str, token: str | None = None
) -> WeightFootprint:
    info_expand, info_files = _model_info(repo_id, token=token)
    expand_dict = _to_dict(info_expand)
    candidates: list[tuple[int, str, str]] = []

    safetensors_info = expand_dict.get('safetensors')
    if isinstance(safetensors_info, dict):
        total = safetensors_info.get('total') or safetensors_info.get(
            'total_size'
        )
        if isinstance(total, (int, float)):
            candidates.append(
                (
                    int(total),
                    'huggingface model_info.safetensors.total',
                    'Checkpoint storage bytes reported by HF safetensors metadata.',
                )
            )

    if info_files is not None:
        siblings = _listify(getattr(info_files, 'siblings', None))
        total = 0
        used = False
        for sibling in siblings:
            sdict = _to_dict(sibling)
            name = str(sdict.get('rfilename') or sdict.get('path') or '')
            size = sdict.get('size')
            if name.endswith(('.safetensors', '.bin', '.pt')) and isinstance(
                size, (int, float)
            ):
                total += int(size)
                used = True
        if used:
            candidates.append(
                (
                    total,
                    'sum(model_info(files_metadata=True).siblings[*].size)',
                    'Checkpoint storage bytes reconstructed from HF file metadata.',
                )
            )

    if not candidates:
        return WeightFootprint(
            total_bytes=None,
            source='unavailable',
            notes='Could not recover checkpoint byte size from HF metadata.',
        )

    best_total, best_source, best_note = max(candidates, key=lambda x: x[0])
    notes = [best_note]
    if len(candidates) > 1:
        rendered = ', '.join(
            f'{src_name}={total / GiB:.2f} GiB'
            for total, src_name, _ in candidates
        )
        notes.append(
            'Multiple HF byte signals were available; using the maximum to avoid undercounting. '
            f'Candidates: {rendered}.'
        )

    return WeightFootprint(
        total_bytes=best_total, source=best_source, notes=' '.join(notes)
    )


def _default_qwen_layer_types(
    num_hidden_layers: int, full_attention_interval: int = 4
) -> list[str]:
    return [
        'linear_attention'
        if bool((i + 1) % full_attention_interval)
        else 'full_attention'
        for i in range(num_hidden_layers)
    ]


def _default_gemma_layer_types(
    num_hidden_layers: int, sliding_pattern: int = 6
) -> list[str]:
    layer_types = [
        'sliding_attention'
        if bool((i + 1) % sliding_pattern)
        else 'full_attention'
        for i in range(num_hidden_layers)
    ]
    if layer_types and layer_types[-1] != 'full_attention':
        layer_types[-1] = 'full_attention'
    return layer_types


def _build_qwen_spec(
    repo_id: str,
    raw_config: dict[str, Any],
    weight_footprint: WeightFootprint,
    family: Literal['qwen3.5', 'qwen3.6'],
) -> ModelMemorySpec:
    text = raw_config.get('text_config', raw_config)
    vision = raw_config.get('vision_config', {})
    num_hidden_layers = int(text['num_hidden_layers'])
    layer_types = tuple(
        text.get('layer_types') or _default_qwen_layer_types(num_hidden_layers)
    )
    num_full = sum(1 for t in layer_types if t == 'full_attention')
    num_linear = sum(1 for t in layer_types if t == 'linear_attention')

    full_group = CacheGroupSpec(
        name='full_attention_kv',
        kind='full_kv',
        layer_type='full_attention',
        total_layers=num_full,
        unique_cache_layers=num_full,
        num_heads=int(text['num_key_value_heads']),
        head_dim=int(
            text.get('head_dim')
            or (text['hidden_size'] // text['num_attention_heads'])
        ),
        seq_len_mode='full',
        kv_copies=2,
        dtype_source='kv',
        notes='Standard KV cache for the full-attention layers only.',
    )

    linear_num_key_heads = int(text['linear_num_key_heads'])
    linear_num_value_heads = int(text['linear_num_value_heads'])
    linear_key_head_dim = int(text['linear_key_head_dim'])
    linear_value_head_dim = int(text['linear_value_head_dim'])
    conv_kernel = int(text['linear_conv_kernel_dim'])
    conv_width = (
        2 * linear_num_key_heads * linear_key_head_dim
        + linear_num_value_heads * linear_value_head_dim
    )

    linear_recurrent = CacheGroupSpec(
        name='linear_attention_recurrent_state',
        kind='linear_recurrent',
        layer_type='linear_attention',
        total_layers=num_linear,
        unique_cache_layers=num_linear,
        fixed_elements_per_sequence=linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim,
        seq_len_mode='fixed',
        dtype_source='linear_state',
        notes='Fixed-size recurrent state for Qwen Gated DeltaNet layers.',
    )
    linear_conv = CacheGroupSpec(
        name='linear_attention_conv_state',
        kind='linear_conv',
        layer_type='linear_attention',
        total_layers=num_linear,
        unique_cache_layers=num_linear,
        fixed_elements_per_sequence=conv_width * conv_kernel,
        seq_len_mode='fixed',
        dtype_source='linear_state',
        notes='Fixed-size causal-convolution state for Qwen Gated DeltaNet layers.',
    )

    notes = [
        f'Parsed as {family} hybrid text+vision model.',
        f'Layer mix: {num_linear} linear-attention layers and {num_full} full-attention layers.',
    ]

    has_vision = isinstance(vision, dict) and bool(vision)

    return ModelMemorySpec(
        repo_id=repo_id,
        family=family,
        architecture='hybrid_qwen_linear_plus_full_attention',
        max_position_embeddings=int(text['max_position_embeddings']),
        text_hidden_size=int(text['hidden_size']),
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        weight_footprint=weight_footprint,
        cache_groups=(full_group, linear_recurrent, linear_conv),
        has_vision=has_vision,
        notes=tuple(notes),
    )


def _build_gemma4_spec(
    repo_id: str, raw_config: dict[str, Any], weight_footprint: WeightFootprint
) -> ModelMemorySpec:
    text = raw_config.get('text_config', raw_config)
    layer_types = tuple(
        text.get('layer_types')
        or _default_gemma_layer_types(int(text['num_hidden_layers']))
    )
    shared_tail = int(text.get('num_kv_shared_layers', 0))
    shared_mask = [False] * len(layer_types)
    for i in range(max(0, len(layer_types) - shared_tail), len(layer_types)):
        shared_mask[i] = True

    num_heads_sliding = int(text['num_key_value_heads'])
    head_dim_sliding = int(
        text.get('head_dim')
        or (text['hidden_size'] // text['num_attention_heads'])
    )
    num_heads_full = int(
        text.get('num_global_key_value_heads')
        or text.get('num_key_value_heads')
    )
    head_dim_full = int(
        text.get('global_head_dim')
        or text.get('head_dim')
        or (text['hidden_size'] // text['num_attention_heads'])
    )
    kv_copies = 1 if bool(text.get('attention_k_eq_v', False)) else 2

    sliding_total = sum(1 for t in layer_types if t == 'sliding_attention')
    full_total = sum(1 for t in layer_types if t == 'full_attention')
    sliding_unique = sum(
        1
        for i, t in enumerate(layer_types)
        if t == 'sliding_attention' and not shared_mask[i]
    )
    full_unique = sum(
        1
        for i, t in enumerate(layer_types)
        if t == 'full_attention' and not shared_mask[i]
    )

    sliding_group = CacheGroupSpec(
        name='sliding_attention_kv',
        kind='sliding_kv',
        layer_type='sliding_attention',
        total_layers=sliding_total,
        unique_cache_layers=sliding_unique,
        num_heads=num_heads_sliding,
        head_dim=head_dim_sliding,
        seq_len_mode='sliding',
        sliding_window=int(text['sliding_window']),
        kv_copies=kv_copies,
        dtype_source='kv',
        notes='Sliding-window KV cache.',
    )
    full_group = CacheGroupSpec(
        name='full_attention_kv',
        kind='full_kv',
        layer_type='full_attention',
        total_layers=full_total,
        unique_cache_layers=full_unique,
        num_heads=num_heads_full,
        head_dim=head_dim_full,
        seq_len_mode='full',
        kv_copies=kv_copies,
        dtype_source='kv',
        notes='Global/full-attention KV cache.',
    )

    vision = raw_config.get('vision_config', {})
    has_vision = isinstance(vision, dict) and bool(vision)

    return ModelMemorySpec(
        repo_id=repo_id,
        family='gemma4',
        architecture='hybrid_gemma4_sliding_plus_full_attention',
        max_position_embeddings=int(text['max_position_embeddings']),
        text_hidden_size=int(text['hidden_size']),
        num_hidden_layers=int(text['num_hidden_layers']),
        layer_types=layer_types,
        weight_footprint=weight_footprint,
        cache_groups=(sliding_group, full_group),
        has_vision=has_vision,
        notes=(
            'Parsed as Gemma 4 hybrid sliding/full-attention model.',
            f'Shared KV tail layers: {shared_tail}.',
        ),
    )


def load_model_spec(repo_id: str, token: str | None = None) -> ModelMemorySpec:
    raw_config = _load_json(repo_id, 'config.json', token=token)
    weight_footprint = _weight_footprint_from_hf(repo_id, token=token)

    model_type = str(raw_config.get('model_type') or '')
    text_config = raw_config.get('text_config') or raw_config
    text_model_type = str(text_config.get('model_type') or model_type)

    if text_model_type in {'qwen3_5_text', 'qwen3_5'}:
        return _build_qwen_spec(
            repo_id, raw_config, weight_footprint, family='qwen3.5'
        )
    if text_model_type in {'qwen3_5_moe_text', 'qwen3_5_moe'}:
        return _build_qwen_spec(
            repo_id, raw_config, weight_footprint, family='qwen3.6'
        )
    if text_model_type in {'gemma4_text', 'gemma4'}:
        return _build_gemma4_spec(repo_id, raw_config, weight_footprint)

    raise ValueError(
        f'Unsupported model_type/text_model_type for this estimator: {model_type} / {text_model_type}'
    )


def _request_floor(
    model: ModelMemorySpec, deployment: DeploymentSpec
) -> RequestFloor:
    token_slope_cluster = 0.0
    fixed_cluster = 0.0
    notes: list[str] = []
    for group in model.cache_groups:
        token_slope_cluster += group.single_sequence_token_slope_cluster(
            deployment
        )
        fixed_cluster += group.single_sequence_fixed_bytes_cluster(deployment)
        notes.append(group.notes)
    total_cluster = deployment.concurrent_sequences * (
        token_slope_cluster * deployment.total_sequence_tokens + fixed_cluster
    )
    per_gpu_total = total_cluster / deployment.tensor_parallel_size
    return RequestFloor(
        total_bytes_per_gpu=per_gpu_total,
        token_slope_bytes_per_gpu=(
            token_slope_cluster / deployment.tensor_parallel_size
        ),
        fixed_bytes_per_gpu=(
            fixed_cluster
            * deployment.concurrent_sequences
            / deployment.tensor_parallel_size
        ),
        notes=tuple(notes),
    )


def _startup_fit(
    model: ModelMemorySpec, deployment: DeploymentSpec
) -> StartupFit:
    weight_per_gpu: float | None = None
    if model.weight_footprint.total_bytes is not None:
        weight_per_gpu = (
            model.weight_footprint.total_bytes / deployment.tensor_parallel_size
        )
    overhead = deployment.startup_overhead_policy.estimate(model, deployment)
    if (
        weight_per_gpu is None
        or deployment.managed_budget_bytes_per_gpu is None
    ):
        return StartupFit(
            weight_bytes_per_gpu=weight_per_gpu,
            overhead_low_bytes_per_gpu=overhead.low_bytes_per_gpu,
            overhead_high_bytes_per_gpu=overhead.high_bytes_per_gpu,
            used_low_bytes_per_gpu=None,
            used_high_bytes_per_gpu=None,
            margin_low_bytes_per_gpu=None,
            margin_high_bytes_per_gpu=None,
            status='maybe',
        )

    used_low = weight_per_gpu + overhead.low_bytes_per_gpu
    used_high = weight_per_gpu + overhead.high_bytes_per_gpu
    margin_high = deployment.managed_budget_bytes_per_gpu - used_low
    margin_low = deployment.managed_budget_bytes_per_gpu - used_high

    if margin_low >= 0:
        status: Literal['yes', 'no', 'maybe'] = 'yes'
    elif margin_high < 0:
        status = 'no'
    else:
        status = 'maybe'

    return StartupFit(
        weight_bytes_per_gpu=weight_per_gpu,
        overhead_low_bytes_per_gpu=overhead.low_bytes_per_gpu,
        overhead_high_bytes_per_gpu=overhead.high_bytes_per_gpu,
        used_low_bytes_per_gpu=used_low,
        used_high_bytes_per_gpu=used_high,
        margin_low_bytes_per_gpu=margin_low,
        margin_high_bytes_per_gpu=margin_high,
        status=status,
    )


def _steady_state_capacity(
    model: ModelMemorySpec,
    deployment: DeploymentSpec,
    startup_fit: StartupFit,
    request_floor: RequestFloor,
) -> CapacityEstimate:
    if (
        deployment.managed_budget_bytes_per_gpu is None
        or startup_fit.used_low_bytes_per_gpu is None
        or startup_fit.used_high_bytes_per_gpu is None
    ):
        return CapacityEstimate(None, None, None, None, None, None, None, None)

    kv_budget_low = (
        deployment.managed_budget_bytes_per_gpu
        - startup_fit.used_high_bytes_per_gpu
    )
    kv_budget_high = (
        deployment.managed_budget_bytes_per_gpu
        - startup_fit.used_low_bytes_per_gpu
    )

    if kv_budget_high <= 0:
        return CapacityEstimate(
            kv_budget_low, kv_budget_high, 0.0, 0.0, 0.0, 0.0, False, False
        )

    per_request = request_floor.total_bytes_per_gpu
    if per_request <= 0:
        max_conc_low = math.inf
        max_conc_high = math.inf
    else:
        max_conc_low = max(0.0, kv_budget_low / per_request)
        max_conc_high = max(0.0, kv_budget_high / per_request)

    per_token_per_gpu = request_floor.token_slope_bytes_per_gpu
    if per_token_per_gpu > 0:
        kv_tokens_low_cluster = max(0.0, kv_budget_low / per_token_per_gpu)
        kv_tokens_high_cluster = max(0.0, kv_budget_high / per_token_per_gpu)
    else:
        kv_tokens_low_cluster = None
        kv_tokens_high_cluster = None

    target = deployment.concurrent_sequences
    fits_low = max_conc_low >= target
    fits_high = max_conc_high >= target

    return CapacityEstimate(
        kv_budget_low_bytes_per_gpu=kv_budget_low,
        kv_budget_high_bytes_per_gpu=kv_budget_high,
        kv_tokens_low_cluster=kv_tokens_low_cluster,
        kv_tokens_high_cluster=kv_tokens_high_cluster,
        max_concurrency_low=max_conc_low,
        max_concurrency_high=max_conc_high,
        fits_target_low=fits_low,
        fits_target_high=fits_high,
    )


def estimate_memory(
    model: ModelMemorySpec, deployment: DeploymentSpec
) -> MemoryEstimate:
    request_floor = _request_floor(model, deployment)
    startup_fit = _startup_fit(model, deployment)
    capacity = _steady_state_capacity(
        model, deployment, startup_fit, request_floor
    )
    return MemoryEstimate(
        model=model,
        deployment=deployment,
        request_floor=request_floor,
        startup_fit=startup_fit,
        capacity=capacity,
    )


def standard_deployments(
    model: ModelMemorySpec,
    *,
    gpu_gib: int | None = 96,
    gpu_memory_utilization: float = 0.95,
    language_model_only: bool = True,
) -> list[DeploymentSpec]:
    native_tokens = model.max_position_embeddings
    gpu_bytes = (gpu_gib * GiB) if gpu_gib is not None else None
    return [
        DeploymentSpec(
            name='tp4_ctx_98k',
            tensor_parallel_size=4,
            prompt_text_tokens=min(98304, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
        DeploymentSpec(
            name='tp4_ctx_128k',
            tensor_parallel_size=4,
            prompt_text_tokens=min(131072, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
        DeploymentSpec(
            name='tp4_ctx_256k',
            tensor_parallel_size=4,
            prompt_text_tokens=min(262144, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
    ]


def _fmt_fit_interval(low: bool | None, high: bool | None) -> str:
    if low is True and high is True:
        return fit_status_markup('yes')
    if low is False and high is False:
        return fit_status_markup('no')
    return fit_status_markup('maybe')


def render_summary_matrix(
    estimates: Sequence[MemoryEstimate], console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(
        title='Deterministic startup + capacity estimate',
        box=box.MINIMAL_DOUBLE_HEAD,
        row_styles=['', 'dim'],
    )
    table.add_column('model')
    table.add_column('deployment')
    table.add_column('gpu GiB', justify='right')
    table.add_column('managed', justify='right')
    table.add_column('weights', justify='right')
    table.add_column('startup ovhd', justify='right')
    table.add_column('startup', justify='center')
    table.add_column('req-cache', justify='right')
    table.add_column('kv budget', justify='right')
    table.add_column('fit@ctx', justify='center')
    table.add_column('max conc', justify='right')

    for est in estimates:
        dep = est.deployment
        sf = est.startup_fit
        cap = est.capacity
        row_style = None
        if sf.status == 'no':
            row_style = 'red'
        table.add_row(
            est.model.repo_id,
            dep.name,
            f'{dep.gpu_memory_gib:.1f}'
            if dep.gpu_memory_gib is not None
            else '-',
            f'{gib(dep.managed_budget_bytes_per_gpu):.2f}'
            if dep.managed_budget_bytes_per_gpu is not None
            else '-',
            gib_text(sf.weight_bytes_per_gpu),
            range_text(
                sf.overhead_low_bytes_per_gpu, sf.overhead_high_bytes_per_gpu
            ),
            fit_status_markup(sf.status),
            gib_text(est.request_floor.total_bytes_per_gpu),
            range_text(
                cap.kv_budget_low_bytes_per_gpu or 0.0,
                cap.kv_budget_high_bytes_per_gpu or 0.0,
            )
            if cap.kv_budget_low_bytes_per_gpu is not None
            else '-',
            _fmt_fit_interval(cap.fits_target_low, cap.fits_target_high),
            f'{cap.max_concurrency_low:.2f}–{cap.max_concurrency_high:.2f}'
            if cap.max_concurrency_low is not None
            else '-',
            style=row_style,
        )
    console.print(table)


def render_detailed_tables(
    estimates: Sequence[MemoryEstimate], console: Console | None = None
) -> None:
    console = console or Console()
    for est in estimates:
        dep = est.deployment
        sf = est.startup_fit
        cap = est.capacity
        header = (
            f'family={est.model.family} | arch={est.model.architecture} | '
            f'seq={dep.total_sequence_tokens:,} | tp={dep.tensor_parallel_size} | '
            f'lm_only={dep.language_model_only} | kv_dtype={dep.kv_cache_dtype.name}'
        )
        console.print(
            Panel(
                header, title=f'{est.model.repo_id} — {dep.name}', expand=False
            )
        )

        t = Table(box=box.SIMPLE_HEAVY)
        t.add_column('metric')
        t.add_column('value', justify='right')
        t.add_column('notes')
        t.add_row(
            'gpu_budget/GPU',
            f'{dep.gpu_memory_gib:.2f}'
            if dep.gpu_memory_gib is not None
            else '-',
            'Physical GPU memory',
        )
        t.add_row(
            'managed_budget/GPU',
            gib_text(dep.managed_budget_bytes_per_gpu),
            'gpu_memory_utilization * gpu_budget',
        )
        t.add_row(
            'weights/GPU',
            gib_text(sf.weight_bytes_per_gpu),
            est.model.weight_footprint.notes,
        )
        t.add_row(
            'startup_overhead/GPU',
            range_text(
                sf.overhead_low_bytes_per_gpu, sf.overhead_high_bytes_per_gpu
            ),
            dep.startup_overhead_policy.estimate(est.model, dep).notes,
        )
        t.add_row(
            'startup_fit',
            fit_status_markup(sf.status),
            'Compares weights + startup overhead against managed budget',
        )
        t.add_row(
            'request_floor_cache/GPU',
            gib_text(est.request_floor.total_bytes_per_gpu),
            'Deterministic bytes required for one configured request at this context',
        )
        t.add_row(
            'token_slope/GPU',
            gib_text(est.request_floor.token_slope_bytes_per_gpu),
            'Per-token request-floor cache coefficient',
        )
        t.add_row(
            'fixed_cache/GPU',
            gib_text(est.request_floor.fixed_bytes_per_gpu),
            'Per-sequence fixed recurrent/conv state',
        )
        if cap.kv_budget_low_bytes_per_gpu is not None:
            t.add_row(
                'kv_budget/GPU',
                range_text(
                    cap.kv_budget_low_bytes_per_gpu,
                    cap.kv_budget_high_bytes_per_gpu,
                ),
                'Managed budget minus startup-used interval',
            )
        if cap.kv_tokens_low_cluster is not None:
            t.add_row(
                'kv_tokens/cluster',
                f'{cap.kv_tokens_low_cluster:,.0f}–{cap.kv_tokens_high_cluster:,.0f}',
                'Cluster-wide token capacity implied by per-GPU KV budget and token slope',
            )
        if cap.max_concurrency_low is not None:
            t.add_row(
                'max_concurrency',
                f'{cap.max_concurrency_low:.2f}–{cap.max_concurrency_high:.2f}',
                f'At ctx={dep.total_sequence_tokens:,} and concurrent_sequences target={dep.concurrent_sequences}',
            )
        console.print(t)
        console.print()


def parse_deployment_arg(
    spec: str, *, default_gpu_mem_util: float, default_language_model_only: bool
) -> DeploymentSpec:
    # Format: name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util][,seqs]
    parts = spec.split(',')
    if len(parts) < 4:
        raise ValueError(
            'Deployment spec must be name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util][,seqs]'
        )
    name = parts[0]
    tp = int(parts[1])
    prompt = int(parts[2])
    max_new = int(parts[3])

    gpu_gib: int | None = None
    media = 0
    kv_dtype = DTYPE_F16
    gpu_mem_util = default_gpu_mem_util
    seqs = 1

    if len(parts) >= 5 and parts[4]:
        gpu_gib = int(parts[4])
    if len(parts) >= 6 and parts[5]:
        media = int(parts[5])
    if len(parts) >= 7 and parts[6]:
        kv_name = parts[6].lower()
        if kv_name == 'fp8':
            kv_dtype = DTYPE_FP8
        elif kv_name == 'fp32':
            kv_dtype = DTYPE_F32
    if len(parts) >= 8 and parts[7]:
        gpu_mem_util = float(parts[7])
    if len(parts) >= 9 and parts[8]:
        seqs = int(parts[8])

    return DeploymentSpec(
        name=name,
        tensor_parallel_size=tp,
        prompt_text_tokens=prompt,
        max_new_tokens=max_new,
        media_soft_tokens=media,
        concurrent_sequences=seqs,
        gpu_memory_bytes=(gpu_gib * GiB) if gpu_gib is not None else None,
        gpu_memory_utilization=gpu_mem_util,
        kv_cache_dtype=kv_dtype,
        language_model_only=default_language_model_only,
    )


def discover_default_repo_ids(token: str | None = None) -> list[str]:
    return list(DEFAULT_PRESET_REPO_IDS)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Deterministic startup-fit + serving-capacity estimator for recent Qwen/Gemma models.'
    )
    p.add_argument(
        'repo_ids',
        nargs='*',
        help='HF repo IDs. If omitted, use the if-guarded preset list in the source.',
    )
    p.add_argument(
        '--list-default-models',
        action='store_true',
        help='Print the default preset repo IDs and exit.',
    )
    p.add_argument('--token', default=None, help='HF token if needed')
    p.add_argument(
        '--standard',
        action='store_true',
        help='Run the built-in standardized deployment set',
    )
    p.add_argument(
        '--deployment',
        action='append',
        default=[],
        help='Custom deployment spec: name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util][,seqs]',
    )
    p.add_argument(
        '--details',
        action='store_true',
        help='Show detailed per-deployment tables',
    )
    p.add_argument(
        '--gpu-gib',
        type=int,
        default=96,
        help='GPU size for standard deployments',
    )
    p.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.95,
        help='Managed memory fraction for standard deployments',
    )
    p.add_argument(
        '--language-model-only',
        action='store_true',
        default=True,
        help='Assume text-only serving mode',
    )
    p.add_argument(
        '--multimodal',
        dest='language_model_only',
        action='store_false',
        help='Enable multimodal resident-memory margins',
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    console = Console()

    repo_ids = list(args.repo_ids)
    if not repo_ids or args.list_default_models:
        repo_ids = discover_default_repo_ids(token=args.token)
        if args.list_default_models:
            for repo_id in repo_ids:
                print(repo_id)
            return 0
        console.print(
            Panel(
                '\n'.join(repo_ids) if repo_ids else '(no models enabled)',
                title='Default preset model set',
                expand=False,
            )
        )

    explicit_deployments = [
        parse_deployment_arg(
            item,
            default_gpu_mem_util=args.gpu_memory_utilization,
            default_language_model_only=args.language_model_only,
        )
        for item in args.deployment
    ]

    estimates: list[MemoryEstimate] = []
    for repo_id in repo_ids:
        model = load_model_spec(repo_id, token=args.token)
        model_deployments: list[DeploymentSpec] = []
        if args.standard or not explicit_deployments:
            model_deployments.extend(
                standard_deployments(
                    model,
                    gpu_gib=args.gpu_gib,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    language_model_only=args.language_model_only,
                )
            )
        model_deployments.extend(explicit_deployments)
        for dep in model_deployments:
            estimates.append(estimate_memory(model, dep))

    render_summary_matrix(estimates, console=console)
    if args.details:
        console.print()
        render_detailed_tables(estimates, console=console)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
