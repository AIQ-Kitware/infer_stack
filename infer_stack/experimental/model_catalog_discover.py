#!/usr/bin/env python3
"""Standalone CLI to discover Hugging Face model metadata and append new model
entries into a infer_stack-style models.yaml manifest.

Highlights
----------
- no-arg mode defaults to `refresh`
- uses the existing models.yaml as a cache of already-known hf_model_id values
  and as a source of stems to expand latest-family siblings
- appends only new entries under top-level `models:`
- shows a unified diff and asks for confirmation unless `--yes` is passed
- refresh unions multiple ranking buckets (newest, updated, popular, downloads)
- quantized and unquantized are separate final buckets, and both are kept
- expands sibling variants for the latest stems so a hit like
  `Qwen/Qwen3.6-35B-A3B-FP8` also pulls `Qwen/Qwen3.6-35B-A3B`
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from huggingface_hub import HfApi, hf_hub_download


PROGRAM_NAME = 'model_catalog_discover'
PROGRAM_VERSION = '0.6.2'

console = Console()
yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.default_flow_style = False
yaml_rt.width = 1000

DEFAULT_AUTHORS = [
    'Qwen',
    # "meta-llama",
    # "mistralai",
    # "google",
    # "microsoft",
    # "ibm-granite",
]
DEFAULT_LIMIT = 0
DEFAULT_PER_CATEGORY_FETCH = 24
DEFAULT_FAMILY_EXPANSION_FETCH = 60
DEFAULT_PER_FAMILY_LIMIT = 0

TEXTY_PIPELINES = {
    'text-generation',
    'image-text-to-text',
    'text2text-generation',
    'conversational',
    '',
    None,
}
REJECT_NAME_PATTERNS = [
    r'(^|[-_/])gguf($|[-_/])',
    r'(^|[-_/])mlx($|[-_/])',
    r'(^|[-_/])exl2($|[-_/])',
    r'(^|[-_/])lora($|[-_/])',
    r'(^|[-_/])adapter($|[-_/])',
    r'(^|[-_/])merged?($|[-_/])',
]
PREFER_NAME_PATTERNS = [
    r'instruct',
    r'chat',
    r'assistant',
    r'reason',
    r'coder',
    r'vl',
]
QUANT_TOKENS = [
    'fp8',
    'awq',
    'gptq',
    'int8',
    'int4',
    '4bit',
    '8bit',
    'bnb',
    'bitsandbytes',
    'quantized',
]
CATEGORY_SPECS = [
    ('newest', 'created_at'),
    ('updated', 'last_modified'),
    ('popular', 'likes'),
    ('downloads', 'downloads'),
]


@dataclass
class Candidate:
    repo_id: str
    author: str | None
    downloads: int | None
    likes: int | None
    last_modified: str | None
    created_at: str | None
    tags: list[str]
    pipeline_tag: str | None
    score: float
    reasons: list[str]
    family: str
    quantization: str
    variant_stem: str
    bucket_hits: list[str] = field(default_factory=list)
    rejected_reason: str | None = None


@dataclass
class DiscoverResult:
    repo_id: str
    model_key: str
    entry: dict[str, Any]
    facts: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class HubInspector:
    def __init__(self, token: str | None = None) -> None:
        self.api = HfApi(token=token)
        self.token = token

    def list_models(
        self,
        *,
        author: str | None = None,
        search: str | None = None,
        sort: str | None = None,
        direction: int = -1,
        limit: int = 20,
    ) -> list[Any]:
        kwargs: dict[str, Any] = {'limit': limit}
        if author:
            kwargs['author'] = author
        if search:
            kwargs['search'] = search
        if sort:
            kwargs['sort'] = sort
            kwargs['direction'] = direction
        try:
            return list(self.api.list_models(**kwargs))
        except TypeError:
            fallback = {
                k: v
                for k, v in kwargs.items()
                if k not in {'sort', 'direction'}
            }
            return list(self.api.list_models(**fallback))

    def model_info(self, repo_id: str) -> Any:
        attempts = [
            {
                'expand': [
                    'cardData',
                    'config',
                    'transformersInfo',
                    'siblings',
                    'tags',
                    'pipeline_tag',
                    'downloads',
                    'likes',
                    'createdAt',
                    'lastModified',
                    'safetensors',
                ]
            },
            {'expand': ['cardData', 'config', 'transformersInfo', 'siblings']},
            {'files_metadata': True},
            {},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self.api.model_info(repo_id, **kwargs)
            except (TypeError, ValueError) as ex:
                last_error = ex
                continue
            except Exception as ex:
                last_error = ex
                continue
        if last_error:
            raise last_error
        raise RuntimeError(f'Unable to query model info for {repo_id}')

    def safetensors_metadata(self, repo_id: str) -> Any | None:
        getter = getattr(self.api, 'get_safetensors_metadata', None)
        if getter is None:
            return None
        try:
            return getter(repo_id)
        except Exception:
            return None

    def download_json(
        self, repo_id: str, filename: str
    ) -> dict[str, Any] | None:
        try:
            path = hf_hub_download(repo_id, filename=filename, token=self.token)
        except Exception:
            return None
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception:
            return None


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    data = getattr(obj, '__dict__', None)
    if isinstance(data, dict):
        return dict(data)
    return {
        name: getattr(obj, name)
        for name in dir(obj)
        if not name.startswith('_') and not callable(getattr(obj, name))
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return re.sub(r'-+', '-', text).strip('-')


def _strip_known_quant_suffix(model_name: str) -> str:
    out = model_name
    changed = True
    while changed:
        changed = False
        for token in QUANT_TOKENS:
            pattern = re.compile(rf'([._-]){re.escape(token)}$', flags=re.I)
            newer = pattern.sub('', out)
            if newer != out:
                out = newer
                changed = True
    return out


def _is_quantized(repo_id: str, tags: list[str]) -> bool:
    lowered = repo_id.lower()
    tagset = {str(t).lower() for t in tags}
    return any(token in lowered for token in QUANT_TOKENS) or any(
        token in tagset for token in QUANT_TOKENS
    )


def _infer_family(repo_id: str, tags: list[str], config: dict[str, Any]) -> str:
    lowered = repo_id.lower()
    patterns = [
        r'(qwen\d+(?:\.\d+)?)',
        r'(gemma\d+)',
        r'(llama[- ]?\d+(?:\.\d+)?)',
        r'(mistral)',
        r'(granite)',
        r'(gpt-oss)',
        r'(phi[- ]?\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            value = match.group(1)
            return (
                value.replace(' ', '').replace('-', '')
                if 'llama' in value
                else value
            )
    model_type = str(config.get('model_type') or '').strip().lower()
    if model_type:
        return model_type
    return lowered.split('/')[-1].split('-')[0]


def _manifest_has_model(
    doc: dict[str, Any], model_key: str, hf_model_id: str
) -> bool:
    models = doc.get('models') or {}
    if model_key in models:
        return True
    for _, value in models.items():
        if (
            isinstance(value, dict)
            and str(value.get('hf_model_id') or '') == hf_model_id
        ):
            return True
    return False


def _load_manifest(path: Path) -> CommentedMap:
    if path.exists():
        data = yaml_rt.load(path.read_text(encoding='utf-8'))
        if data is None:
            data = CommentedMap()
        if not isinstance(data, CommentedMap):
            data = CommentedMap(data)
    else:
        data = CommentedMap()
    if 'models' not in data or data['models'] is None:
        data['models'] = CommentedMap()
    if not isinstance(data['models'], CommentedMap):
        data['models'] = CommentedMap(data['models'])
    return data


def _manifest_repo_ids(doc: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for _, value in (doc.get('models') or {}).items():
        if isinstance(value, dict):
            repo_id = str(value.get('hf_model_id') or '')
            if repo_id:
                found.add(repo_id)
    return found


def _manifest_variant_stems(doc: dict[str, Any]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for _, value in (doc.get('models') or {}).items():
        if not isinstance(value, dict):
            continue
        repo_id = str(value.get('hf_model_id') or '')
        if '/' not in repo_id:
            continue
        author, name = repo_id.split('/', 1)
        found.add((author, _strip_known_quant_suffix(name)))
    return found


def _render_yaml(doc: CommentedMap) -> str:
    buf = io.StringIO()
    yaml_rt.dump(doc, buf)
    return buf.getvalue()


def _render_diff(before: str, after: str, manifest_path: Path) -> str:
    return ''.join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f'{manifest_path}.before',
            tofile=str(manifest_path),
        )
    )


def _apply_with_confirmation(
    *,
    manifest_path: Path,
    original_doc: CommentedMap,
    updated_doc: CommentedMap,
    yes: bool,
) -> bool:
    before = _render_yaml(original_doc)
    after = _render_yaml(updated_doc)
    if before == after:
        console.print(
            Panel('No changes to apply.', title='Diff', border_style='yellow')
        )
        return False
    diff = _render_diff(before, after, manifest_path)
    console.print(Syntax(diff, 'diff', theme='ansi_dark', word_wrap=False))
    if not yes and not Confirm.ask('Apply this diff?'):
        console.print(
            Panel(
                'Aborted. Manifest not modified.',
                title='Apply',
                border_style='yellow',
            )
        )
        return False
    manifest_path.write_text(after, encoding='utf-8')
    console.print(
        Panel(f'Wrote {manifest_path}', title='Apply', border_style='green')
    )
    return True


def _score_candidate(
    model: dict[str, Any], mode: str
) -> tuple[float, list[str], str | None]:
    repo_id = str(model.get('id') or model.get('modelId') or '')
    lowered = repo_id.lower()
    tags = [str(t).lower() for t in _as_list(model.get('tags'))]
    pipeline_tag = model.get('pipeline_tag') or model.get('pipelineTag')
    author = str(model.get('author') or '')
    downloads = int(model.get('downloads') or 0)
    likes = int(model.get('likes') or 0)

    if pipeline_tag not in TEXTY_PIPELINES:
        return (
            0.0,
            [],
            f"pipeline_tag '{pipeline_tag}' is outside default text-focused policy",
        )

    for pattern in REJECT_NAME_PATTERNS:
        if re.search(pattern, lowered):
            return (
                0.0,
                [],
                f"repo name matches excluded artifact pattern '{pattern}'",
            )

    reasons: list[str] = []
    score = 0.0
    if author:
        reasons.append(f'author={author}')
    if downloads:
        score += min(downloads, 2_000_000) / 20_000.0
        reasons.append(f'downloads={downloads}')
    if likes:
        score += min(likes, 100_000) / 1000.0
        reasons.append(f'likes={likes}')
    if pipeline_tag:
        score += 10.0
    if mode == 'frontier':
        freshness = str(
            model.get('created_at')
            or model.get('createdAt')
            or model.get('last_modified')
            or model.get('lastModified')
            or ''
        )
        if freshness:
            score += 15.0
            reasons.append('frontier+freshness')
    for pattern in PREFER_NAME_PATTERNS:
        if re.search(pattern, lowered):
            score += 5.0
            reasons.append(f'name~/{pattern}/')
    if _is_quantized(repo_id, tags):
        score += 1.0
        reasons.append('quantized')
    else:
        score += 2.0
        reasons.append('unquantized')
    return score, reasons, None


def _candidate_from_model(
    model: Any, mode: str, bucket_name: str | None = None
) -> Candidate:
    mdict = _to_dict(model)
    repo_id = str(mdict.get('id') or mdict.get('modelId') or '')
    tags = [str(t) for t in _as_list(mdict.get('tags'))]
    _raw_config = mdict.get('config')
    config: dict[str, Any] = _raw_config if isinstance(_raw_config, dict) else {}
    score, reasons, rejected_reason = _score_candidate(mdict, mode)
    author = str(mdict.get('author') or '') or None
    family = _infer_family(repo_id, tags, config)
    model_name = repo_id.split('/', 1)[-1]
    return Candidate(
        repo_id=repo_id,
        author=author,
        downloads=mdict.get('downloads'),
        likes=mdict.get('likes'),
        last_modified=str(
            mdict.get('last_modified') or mdict.get('lastModified') or ''
        )
        or None,
        created_at=str(mdict.get('created_at') or mdict.get('createdAt') or '')
        or None,
        tags=tags,
        pipeline_tag=mdict.get('pipeline_tag') or mdict.get('pipelineTag'),
        score=score,
        reasons=reasons,
        family=family,
        quantization='quantized'
        if _is_quantized(repo_id, tags)
        else 'unquantized',
        variant_stem=_strip_known_quant_suffix(model_name),
        bucket_hits=[bucket_name] if bucket_name else [],
        rejected_reason=rejected_reason,
    )


def _merge_bucket_hits(into: Candidate, other: Candidate) -> None:
    for hit in other.bucket_hits:
        if hit not in into.bucket_hits:
            into.bucket_hits.append(hit)
    into.score = max(into.score, other.score)


def _pick_refresh_candidates(
    inspector: HubInspector,
    *,
    authors: list[str],
    limit: int,
    mode: str,
    per_category_fetch: int,
    per_family_limit: int,
    existing_repo_ids: set[str],
) -> tuple[list[Candidate], list[Candidate]]:
    accepted_by_repo: dict[str, Candidate] = {}
    rejected: list[Candidate] = []
    family_counts: dict[tuple[str | None, str, str], int] = {}

    for bucket_name, sort_key in CATEGORY_SPECS:
        for author in authors:
            try:
                models = inspector.list_models(
                    author=author,
                    sort=sort_key,
                    direction=-1,
                    limit=per_category_fetch,
                )
            except Exception as ex:
                rejected.append(
                    Candidate(
                        repo_id=f'<search:{author}:{bucket_name}>',
                        author=author,
                        downloads=None,
                        likes=None,
                        last_modified=None,
                        created_at=None,
                        tags=[],
                        pipeline_tag=None,
                        score=0.0,
                        reasons=[],
                        family='',
                        quantization='unknown',
                        variant_stem='',
                        bucket_hits=[bucket_name],
                        rejected_reason=str(ex),
                    )
                )
                continue
            for model in models:
                cand = _candidate_from_model(model, mode, bucket_name)
                if not cand.repo_id or cand.repo_id in existing_repo_ids:
                    continue
                if cand.rejected_reason is not None:
                    rejected.append(cand)
                    continue
                key = (cand.author, cand.family, cand.quantization)
                if (
                    per_family_limit
                    and per_family_limit > 0
                    and family_counts.get(key, 0) >= per_family_limit
                    and cand.repo_id not in accepted_by_repo
                ):
                    continue
                if cand.repo_id in accepted_by_repo:
                    _merge_bucket_hits(accepted_by_repo[cand.repo_id], cand)
                else:
                    accepted_by_repo[cand.repo_id] = cand
                    family_counts[key] = family_counts.get(key, 0) + 1

    ordered = sorted(
        accepted_by_repo.values(),
        key=lambda c: (
            len(c.bucket_hits),
            c.score,
            c.downloads or 0,
            c.likes or 0,
            c.created_at or '',
            c.repo_id,
        ),
        reverse=True,
    )
    if limit and limit > 0:
        by_lane: dict[str, list[Candidate]] = {
            'unquantized': [],
            'quantized': [],
        }
        for cand in ordered:
            lane = by_lane.setdefault(cand.quantization, [])
            if len(lane) < limit:
                lane.append(cand)
        accepted = by_lane.get('unquantized', []) + by_lane.get('quantized', [])
    else:
        accepted = ordered
    return accepted, rejected


def _expand_family_variants(
    inspector: HubInspector,
    *,
    seeds: list[Candidate],
    cached_stems: set[tuple[str, str]],
    existing_repo_ids: set[str],
    mode: str,
    family_expansion_fetch: int,
) -> list[Candidate]:
    expanded: dict[str, Candidate] = {c.repo_id: c for c in seeds}
    search_jobs: set[tuple[str, str, str]] = set()

    for cand in seeds:
        if cand.author:
            search_jobs.add((cand.author, cand.family, cand.variant_stem))

    for author, stem in sorted(cached_stems):
        inferred_family = _infer_family(f'{author}/{stem}', [], {})
        search_jobs.add((author, inferred_family, stem))

    for author, family, stem in sorted(search_jobs):
        search_terms = []
        for term in (stem, family):
            term = (term or '').strip()
            if term and term not in search_terms:
                search_terms.append(term)

        for term in search_terms:
            try:
                models = inspector.list_models(
                    author=author, search=term, limit=family_expansion_fetch
                )
            except Exception:
                continue
            for model in models:
                cand = _candidate_from_model(
                    model, mode, bucket_name='family-expand'
                )
                if (
                    cand.repo_id in existing_repo_ids
                    or cand.rejected_reason is not None
                    or cand.author != author
                ):
                    continue

                repo_name = cand.repo_id.split('/', 1)[-1]
                same_stem = (
                    cand.variant_stem == stem
                    or repo_name == stem
                    or repo_name.startswith(stem + '-')
                    or repo_name.startswith(stem + '.')
                )
                if cand.family != family or not same_stem:
                    continue

                if cand.repo_id in expanded:
                    _merge_bucket_hits(expanded[cand.repo_id], cand)
                else:
                    expanded[cand.repo_id] = cand

    return sorted(
        expanded.values(),
        key=lambda c: (
            c.author or '',
            c.variant_stem,
            0 if c.quantization == 'unquantized' else 1,
            -len(c.bucket_hits),
            -(c.downloads or 0),
            c.repo_id,
        ),
    )


def _print_candidate_table(candidates: list[Candidate], title: str) -> None:
    table = Table(title=title)
    table.add_column('repo')
    table.add_column('family')
    table.add_column('quant')
    table.add_column('buckets')
    table.add_column('downloads', justify='right')
    table.add_column('likes', justify='right')
    for item in candidates:
        table.add_row(
            item.repo_id,
            item.family,
            item.quantization,
            ','.join(item.bucket_hits),
            str(item.downloads or ''),
            str(item.likes or ''),
        )
    console.print(table)


def _collect_numeric_candidates(
    obj: Any, keys: set[str], out: list[int]
) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys and isinstance(v, (int, float)):
                out.append(int(v))
            else:
                _collect_numeric_candidates(v, keys, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_numeric_candidates(item, keys, out)


def _infer_context_window(
    config: dict[str, Any],
    transformers_info: dict[str, Any],
    generation_config: dict[str, Any],
) -> int | None:
    candidates: list[int] = []
    _collect_numeric_candidates(
        config,
        {
            'max_position_embeddings',
            'n_positions',
            'seq_length',
            'context_length',
            'max_seq_len',
            'model_max_length',
        },
        candidates,
    )
    _collect_numeric_candidates(
        transformers_info,
        {'max_position_embeddings', 'model_max_length'},
        candidates,
    )
    _collect_numeric_candidates(generation_config, {'max_length'}, candidates)
    candidates = [x for x in candidates if x and x > 0]
    return max(candidates) if candidates else None


def _infer_modalities(
    *,
    tags: list[str],
    pipeline_tag: str | None,
    config: dict[str, Any],
    siblings: list[str],
) -> list[str]:
    tagset = {t.lower() for t in tags}
    names = ' '.join(
        [
            str(pipeline_tag or '').lower(),
            *sorted(tagset),
            *(s.lower() for s in siblings),
        ]
    )
    arch_names = ' '.join(
        str(x).lower() for x in _as_list(config.get('architectures'))
    )
    modalities: list[str] = ['text']
    if (
        any(
            tok in names
            for tok in [
                'image-text-to-text',
                'vision',
                'vl',
                'multimodal',
                'image_processor',
                'processor_config.json',
            ]
        )
        or 'vision' in arch_names
    ):
        modalities.insert(0, 'image')
    return modalities


def _estimate_memory_hints(metadata: Any | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    mdict = _to_dict(metadata)
    hints: dict[str, Any] = {}
    total = mdict.get('total') or mdict.get('total_size')
    if isinstance(total, (int, float)) and total > 0:
        gib = max(1, round(float(total) / (1024**3)))
        hints['min_vram_gib_per_replica'] = gib
        hints['memory_class_gib'] = gib
    return hints


def _discover_repo(
    inspector: HubInspector,
    repo_id: str,
    *,
    include_memory_hints: bool = False,
    model_key_override: str | None = None,
) -> DiscoverResult:
    info = _to_dict(inspector.model_info(repo_id))
    tags = [str(t) for t in _as_list(info.get('tags'))]
    _raw_config = info.get('config')
    config: dict[str, Any] = _raw_config if isinstance(_raw_config, dict) else {}
    _raw_tinfo = info.get('transformersInfo')
    transformers_info: dict[str, Any] = _raw_tinfo if isinstance(_raw_tinfo, dict) else {}
    pipeline_tag = info.get('pipeline_tag') or info.get('pipelineTag')
    siblings_raw = _as_list(info.get('siblings'))
    siblings = []
    for item in siblings_raw:
        if isinstance(item, dict):
            siblings.append(
                str(item.get('rfilename') or item.get('path') or '')
            )
        else:
            siblings.append(
                str(
                    getattr(item, 'rfilename', None)
                    or getattr(item, 'path', None)
                    or item
                )
            )
    generation_config = (
        inspector.download_json(repo_id, 'generation_config.json') or {}
    )
    tokenizer_config = (
        inspector.download_json(repo_id, 'tokenizer_config.json') or {}
    )
    safetensors = inspector.safetensors_metadata(repo_id)

    family = _infer_family(repo_id, tags, config)
    modalities = _infer_modalities(
        tags=tags, pipeline_tag=pipeline_tag, config=config, siblings=siblings
    )
    context_window = _infer_context_window(
        config, transformers_info, generation_config
    )

    author, name = repo_id.split('/', 1)
    model_key = model_key_override or _slugify(name)
    logical_model_name = f'{author.lower()}/{model_key}'
    entry: dict[str, Any] = {
        'hf_model_id': repo_id,
        'url': f'hf://{repo_id}',
        'family': family,
        'modalities': modalities,
        'tokenizer_name': tokenizer_config.get('name_or_path') or repo_id,
        'logical_model_name': logical_model_name,
        'served_model_name': logical_model_name,
        'defaults': {
            'gpu_memory_utilization': 0.9,
            'enable_prefix_caching': True,
        },
        'notes': [
            f'Generated by {PROGRAM_NAME} {PROGRAM_VERSION} from Hugging Face metadata.',
            'Review local serving policy separately: resource profile, topology, VRAM, and runtime concurrency stay operator-owned.',
        ],
    }
    license_hint = next(
        (t for t in tags if t.lower().startswith('license:')), None
    ) or config.get('license')
    if license_hint:
        entry['notes'].append(
            f'Upstream license hint: {str(license_hint).split(":", 1)[-1]}.'
        )
    if context_window:
        entry['context_window'] = context_window
        if context_window >= 262144:
            entry['defaults']['max_model_len'] = 65536
            entry['defaults']['max_num_batched_tokens'] = 4096
            entry['defaults']['max_num_seqs'] = 4
        elif context_window >= 65536:
            entry['defaults']['max_model_len'] = 65536
            entry['defaults']['max_num_batched_tokens'] = 4096
            entry['defaults']['max_num_seqs'] = 8
        else:
            entry['defaults']['max_model_len'] = min(context_window, 32768)
            entry['defaults']['max_num_batched_tokens'] = 8192
            entry['defaults']['max_num_seqs'] = 16
    if include_memory_hints:
        for k, v in _estimate_memory_hints(safetensors).items():
            entry[k] = v

    return DiscoverResult(
        repo_id=repo_id,
        model_key=model_key,
        entry=entry,
        facts={
            'repo_id': repo_id,
            'tags': tags,
            'pipeline_tag': pipeline_tag,
            'context_window': context_window,
            'siblings': siblings,
        },
    )


def _run_add(
    *,
    manifest_path: Path,
    repo_ids: list[str],
    token: str | None,
    include_memory_hints: bool,
    yes: bool,
    model_key: str | None = None,
) -> int:
    inspector = HubInspector(token=token)
    original_doc = _load_manifest(manifest_path)
    updated_doc = _load_manifest(manifest_path)
    added = 0
    skipped = 0
    for repo_id in repo_ids:
        result = _discover_repo(
            inspector,
            repo_id,
            include_memory_hints=include_memory_hints,
            model_key_override=(model_key if len(repo_ids) == 1 else None),
        )
        if _manifest_has_model(updated_doc, result.model_key, result.repo_id):
            console.print(f'[yellow]skip[/yellow] {repo_id} (already present)')
            skipped += 1
            continue
        updated_doc['models'][result.model_key] = CommentedMap(result.entry)
        added += 1
    if added == 0:
        console.print(
            Panel(
                f'No new models to add. Skipped {skipped}.',
                title='Add',
                border_style='yellow',
            )
        )
        return 0
    _apply_with_confirmation(
        manifest_path=manifest_path,
        original_doc=original_doc,
        updated_doc=updated_doc,
        yes=yes,
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    token = getattr(args, 'token', None) or os.environ.get('HF_TOKEN')
    inspector = HubInspector(token=token)
    models = inspector.list_models(
        author=args.author, search=args.search, limit=args.limit
    )
    table = Table(title='Hugging Face model search')
    table.add_column('repo')
    table.add_column('downloads', justify='right')
    table.add_column('likes', justify='right')
    for model in models:
        mdict = _to_dict(model)
        table.add_row(
            str(mdict.get('id') or mdict.get('modelId') or ''),
            str(mdict.get('downloads') or ''),
            str(mdict.get('likes') or ''),
        )
    console.print(table)
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    token = getattr(args, 'token', None) or os.environ.get('HF_TOKEN')
    return _run_add(
        manifest_path=Path(args.manifest),
        repo_ids=args.repo_ids,
        token=token,
        include_memory_hints=args.include_memory_hints,
        yes=args.yes,
        model_key=args.model_key,
    )


def cmd_refresh(args: argparse.Namespace) -> int:
    token = getattr(args, 'token', None) or os.environ.get('HF_TOKEN')
    inspector = HubInspector(token=token)
    manifest_path = Path(getattr(args, 'manifest', 'models.yaml'))
    manifest_doc = _load_manifest(manifest_path)
    existing_repo_ids = _manifest_repo_ids(manifest_doc)
    cached_stems = _manifest_variant_stems(manifest_doc)

    accepted, rejected = _pick_refresh_candidates(
        inspector,
        authors=list(getattr(args, 'authors', None) or DEFAULT_AUTHORS),
        limit=getattr(args, 'limit', DEFAULT_LIMIT),
        mode='frontier' if getattr(args, 'frontier', False) else 'stable',
        per_category_fetch=getattr(
            args, 'per_category_fetch', DEFAULT_PER_CATEGORY_FETCH
        ),
        per_family_limit=getattr(
            args, 'per_family_limit', DEFAULT_PER_FAMILY_LIMIT
        ),
        existing_repo_ids=existing_repo_ids,
    )
    expanded = _expand_family_variants(
        inspector,
        seeds=accepted,
        cached_stems=cached_stems,
        existing_repo_ids=existing_repo_ids,
        mode='frontier' if getattr(args, 'frontier', False) else 'stable',
        family_expansion_fetch=getattr(
            args, 'family_expansion_fetch', DEFAULT_FAMILY_EXPANSION_FETCH
        ),
    )
    if not expanded:
        console.print(
            Panel(
                'No candidates passed the default refresh policy.',
                title='Refresh',
                border_style='yellow',
            )
        )
        return 0
    _print_candidate_table(expanded, 'Refresh candidates')
    if getattr(args, 'show_rejected', False) and rejected:
        _print_candidate_table(rejected[:20], 'Rejected candidates')
    return _run_add(
        manifest_path=manifest_path,
        repo_ids=[c.repo_id for c in expanded],
        token=token,
        include_memory_hints=args.include_memory_hints,
        yes=args.yes,
        model_key=None,
    )


def _add_refresh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--manifest',
        default='models.yaml',
        help='Repo manifest to append to. Default: ./models.yaml',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_LIMIT,
        help=f'Per quantization lane cap after ranking. Default: {DEFAULT_LIMIT} (0 means unlimited)',
    )
    parser.add_argument(
        '--authors',
        nargs='*',
        default=None,
        help='Optional allowlisted authors. Defaults to a curated built-in set.',
    )
    parser.add_argument(
        '--frontier',
        action='store_true',
        help='Bias candidate ranking toward newer releases.',
    )
    parser.add_argument(
        '--include-memory-hints',
        action='store_true',
        help='Emit rough VRAM-related hints from safetensors metadata.',
    )
    parser.add_argument(
        '--show-rejected',
        action='store_true',
        help='Also print a table of rejected refresh candidates.',
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Apply the diff without interactive confirmation.',
    )
    parser.add_argument(
        '--token', default=None, help='HF token; defaults to HF_TOKEN env var.'
    )
    parser.add_argument(
        '--per-category-fetch',
        type=int,
        default=DEFAULT_PER_CATEGORY_FETCH,
        help=f'How many repos to fetch per ranking category per author. Default: {DEFAULT_PER_CATEGORY_FETCH}',
    )
    parser.add_argument(
        '--per-family-limit',
        type=int,
        default=DEFAULT_PER_FAMILY_LIMIT,
        help=f'Limit initial seed models per (author,family,quantization). Default: {DEFAULT_PER_FAMILY_LIMIT} (0 means unlimited)',
    )
    parser.add_argument(
        '--family-expansion-fetch',
        type=int,
        default=DEFAULT_FAMILY_EXPANSION_FETCH,
        help=f'How many search results to inspect when expanding sibling variants for a stem. Default: {DEFAULT_FAMILY_EXPANSION_FETCH}',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Discover Hugging Face model metadata and append conservative infer_stack-compatible model entries.'
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {PROGRAM_VERSION}'
    )
    _add_refresh_args(parser)
    sub = parser.add_subparsers(dest='command')

    s = sub.add_parser(
        'search', help='Search the Hugging Face Hub for candidate models.'
    )
    s.add_argument('--search', default=None, help='Free-text search string.')
    s.add_argument('--author', default=None, help='Optional author/org filter.')
    s.add_argument('--limit', type=int, default=20)
    s.add_argument(
        '--token', default=None, help='HF token; defaults to HF_TOKEN env var.'
    )
    s.set_defaults(func=cmd_search)

    s = sub.add_parser(
        'add',
        help='Add one or more explicit Hugging Face repo IDs into the manifest.',
    )
    s.add_argument(
        'repo_ids',
        nargs='+',
        help='One or more Hugging Face repo IDs like Qwen/Qwen3.6-35B-A3B.',
    )
    s.add_argument(
        '--model-key',
        default=None,
        help='Override the generated model key for a single repo.',
    )
    s.add_argument(
        '--manifest',
        default='models.yaml',
        help='Repo manifest to append to. Default: ./models.yaml',
    )
    s.add_argument(
        '--include-memory-hints',
        action='store_true',
        help='Emit rough VRAM-related hints from safetensors metadata.',
    )
    s.add_argument(
        '--yes',
        action='store_true',
        help='Apply the diff without interactive confirmation.',
    )
    s.add_argument(
        '--token', default=None, help='HF token; defaults to HF_TOKEN env var.'
    )
    s.set_defaults(func=cmd_add)

    s = sub.add_parser(
        'refresh',
        help='Refresh the manifest by discovering new high-signal models.',
    )
    _add_refresh_args(s)
    s.set_defaults(func=cmd_refresh)

    parser.set_defaults(func=cmd_refresh)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
