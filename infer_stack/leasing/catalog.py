"""The serving catalog: the declarative input to the leasing controller.

This is the new, version-controlled, lifecycle-free description of *what can be
served* (the redesign demotes the old "profiles" singleton; see
``dev/infer-stack-redesign-critique.md`` §9). It parses a ``catalog.yaml`` into
typed specs and resolves an endpoint or bundle name into the
:class:`~infer_stack.leasing.models.EndpointRequest` objects the ledger already
consumes — so the catalog never has to know about leases, demand, or backends.

Schema (all sections optional except as referenced)::

    models:                 # vLLM models (what exists)
      qwen-coder-32b:
        source: hf://Qwen/Qwen2.5-Coder-32B-Instruct
        revision: main
        quantization: null
        dtype: null

    endpoints:              # served API names (what users ask for)
      qwen-coder:
        model: qwen-coder-32b
        engine: vllm
        runtime: {tensor_parallel_size: 1, max_model_len: 32768}
        sharing: {mode: shared-compatible}
        reclaim: {policy: keep-warm}
        protocol: chat        # 'chat' (default) or 'completions' — which OpenAI
                              # surface the readiness probe (and clients) use; a
                              # completions-only model needs protocol: completions
      qwen-small:
        engine: ollama
        host: local-ollama
        model: qwen3.5:4b

    runtime_hosts:          # Ollama daemons (one daemon, many tags)
      local-ollama:
        engine: ollama
        placement: {gpu_indices: [1]}
        settings: {keep_alive: 2m, num_parallel: 1, max_loaded_models: 2}
        storage: {model_store: shared-ollama-store}

    bundles:                # convenience: a named list of endpoints
      draft-and-verify: [draft-model, verifier-model]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import (
    EndpointRequest,
    Sharing,
    ollama_structural,
    vllm_structural,
)

VLLM = 'vllm'
OLLAMA = 'ollama'

DEFAULT_RECLAIM = 'keep-warm'


class CatalogError(ValueError):
    """Raised when a catalog is structurally invalid or has dangling refs."""


@dataclass
class ModelSpec:
    """A vLLM model: what exists, independent of how it is run."""

    name: str
    source: str
    revision: str | None = None
    quantization: str | None = None
    dtype: str | None = None

    @property
    def hf_model_id(self) -> str:
        """The Hugging Face id (``source`` minus any ``hf://`` scheme)."""
        if '://' in self.source:
            return self.source.split('://', 1)[1]
        return self.source


@dataclass
class RuntimeHostSpec:
    """A long-running engine instance. Today only Ollama needs an explicit one
    (one daemon serves many tags); vLLM hosts are synthesized per endpoint."""

    name: str
    engine: str
    gpu_indices: list[int] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    model_store: str | None = None
    image: str | None = None


@dataclass
class EndpointSpec:
    """A served API name → (model/tag, engine, runtime, sharing policy)."""

    name: str
    engine: str
    model: str
    host: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    sharing: str = Sharing.SHARED
    reclaim: str = DEFAULT_RECLAIM
    served_name: str | None = None
    # OpenAI surface the readiness probe (and clients) should use: 'chat' hits
    # /chat/completions, 'completions' hits /completions. A completions-only
    # model never answers a chat probe, so this must match how it is served.
    protocol: str = 'chat'


def _parse_sharing(value: Any) -> str:
    if value is None:
        return Sharing.SHARED
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('mode', Sharing.SHARED)
    raise CatalogError(f'invalid sharing spec: {value!r}')


def _parse_reclaim(value: Any) -> str:
    if value is None:
        return DEFAULT_RECLAIM
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('policy', DEFAULT_RECLAIM)
    raise CatalogError(f'invalid reclaim spec: {value!r}')


def _parse_protocol(value: Any) -> str:
    if value is None:
        return 'chat'
    if value in ('chat', 'completions'):
        return value
    raise CatalogError(
        f"invalid protocol {value!r} (use 'chat' or 'completions')"
    )


@dataclass
class Catalog:
    """A parsed, validated serving catalog."""

    models: dict[str, ModelSpec] = field(default_factory=dict)
    endpoints: dict[str, EndpointSpec] = field(default_factory=dict)
    hosts: dict[str, RuntimeHostSpec] = field(default_factory=dict)
    bundles: dict[str, list[str]] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Catalog:
        """Parse and validate a catalog mapping.

        Example:
            >>> cat = Catalog.from_dict({
            ...     'models': {'m': {'source': 'hf://org/m'}},
            ...     'endpoints': {
            ...         'e': {'engine': 'vllm', 'model': 'm',
            ...               'runtime': {'max_model_len': 8192}}},
            ... })
            >>> req = cat.resolve_endpoint('e')
            >>> req.engine, req.capacity
            ('vllm', {'max_model_len': 8192})
        """
        data = data or {}
        models = {
            name: ModelSpec(
                name=name,
                source=spec['source'],
                revision=spec.get('revision'),
                quantization=spec.get('quantization'),
                dtype=spec.get('dtype'),
            )
            for name, spec in (data.get('models') or {}).items()
        }
        hosts = {}
        for name, spec in (data.get('runtime_hosts') or {}).items():
            spec = spec or {}
            hosts[name] = RuntimeHostSpec(
                name=name,
                engine=spec.get('engine', OLLAMA),
                gpu_indices=list((spec.get('placement') or {}).get('gpu_indices', [])),
                settings=dict(spec.get('settings') or {}),
                model_store=(spec.get('storage') or {}).get('model_store'),
                image=spec.get('image'),
            )
        endpoints = {}
        for name, spec in (data.get('endpoints') or {}).items():
            spec = spec or {}
            endpoints[name] = EndpointSpec(
                name=name,
                engine=spec.get('engine', VLLM),
                model=spec.get('model'),
                host=spec.get('host'),
                runtime=dict(spec.get('runtime') or {}),
                sharing=_parse_sharing(spec.get('sharing')),
                reclaim=_parse_reclaim(spec.get('reclaim')),
                served_name=spec.get('public_name') or spec.get('served_name'),
                protocol=_parse_protocol(spec.get('protocol')),
            )
        bundles = {
            name: list(members or [])
            for name, members in (data.get('bundles') or {}).items()
        }
        catalog = cls(
            models=models, endpoints=endpoints, hosts=hosts, bundles=bundles
        )
        catalog.validate()
        return catalog

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        text = Path(path).expanduser().read_text()
        return cls.from_dict(yaml.safe_load(text))

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Check engine values and cross-references; raise if any are wrong."""
        errors = self.errors()
        if errors:
            raise CatalogError('; '.join(errors))

    def errors(self) -> list[str]:
        errors: list[str] = []
        for ep in self.endpoints.values():
            if ep.engine == VLLM:
                if not ep.model:
                    errors.append(
                        f"endpoint '{ep.name}' (vllm) needs a 'model'"
                    )
                elif ep.model not in self.models:
                    errors.append(
                        f"endpoint '{ep.name}' (vllm) references unknown "
                        f"model '{ep.model}'"
                    )
            elif ep.engine == OLLAMA:
                if not ep.host:
                    errors.append(
                        f"endpoint '{ep.name}' (ollama) needs a 'host'"
                    )
                elif ep.host not in self.hosts:
                    errors.append(
                        f"endpoint '{ep.name}' (ollama) references unknown "
                        f"host '{ep.host}'"
                    )
                if not ep.model:
                    errors.append(
                        f"endpoint '{ep.name}' (ollama) needs a 'model' tag"
                    )
            else:
                errors.append(
                    f"endpoint '{ep.name}' has unknown engine '{ep.engine}'"
                )
        for bundle, members in self.bundles.items():
            for member in members:
                if member not in self.endpoints:
                    errors.append(
                        f"bundle '{bundle}' references unknown endpoint "
                        f"'{member}'"
                    )
        return errors

    # -- resolution --------------------------------------------------------

    def _unknown_endpoint_error(self, name: str) -> CatalogError:
        """A helpful error for a name that isn't an endpoint.

        You acquire *endpoints*, not models — a common slip is to pass a
        model name (``qwen05``) instead of one of its endpoints (``qwen05-1``).
        Recognize that case and point at the endpoints that run the model;
        otherwise fall back to a did-you-mean over the known endpoints/bundles.
        """
        if name in self.models:
            serving = sorted(
                n for n, ep in self.endpoints.items() if ep.model == name
            )
            if serving:
                return CatalogError(
                    f"'{name}' is a model, not an endpoint — you bring up an "
                    f"endpoint that runs it. Endpoints for '{name}': "
                    f"{', '.join(serving)}  (e.g. `infer-stack acquire "
                    f"{serving[0]}`)."
                )
            return CatalogError(
                f"'{name}' is a model with no endpoints yet — add one with "
                f"`infer-stack catalog endpoint add --model {name}` "
                f"(it defaults to the endpoint name '{name}-1')."
            )

        import difflib

        pool = list(self.endpoints) + list(self.bundles)
        close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
        if close:
            return CatalogError(
                f"unknown endpoint '{name}' — did you mean: "
                f"{', '.join(close)}?"
            )
        if self.endpoints:
            return CatalogError(
                f"unknown endpoint '{name}'. Available endpoints: "
                f"{', '.join(sorted(self.endpoints))}  "
                f"(`infer-stack catalog show` for details)."
            )
        return CatalogError(
            f"unknown endpoint '{name}' — the catalog has no endpoints yet "
            f"(add one with `infer-stack catalog endpoint add …`)."
        )

    def resolve_endpoint(
        self, name: str, *, sharing: str | None = None
    ) -> EndpointRequest:
        """Resolve one endpoint name into a ledger :class:`EndpointRequest`.

        ``sharing`` overrides the catalog's declared policy (e.g. the CLI
        ``--dedicated`` flag) when given.
        """
        if name not in self.endpoints:
            raise self._unknown_endpoint_error(name)
        ep = self.endpoints[name]
        share = sharing or ep.sharing
        if ep.engine == VLLM:
            return self._resolve_vllm(ep, share)
        if ep.engine == OLLAMA:
            return self._resolve_ollama(ep, share)
        raise CatalogError(
            f"endpoint '{name}' has unknown engine '{ep.engine}'"
        )

    def resolve_names(
        self, names: list[str], *, sharing: str | None = None
    ) -> list[EndpointRequest]:
        """Expand a mix of endpoint and bundle names into requests.

        Bundles expand to their member endpoints; duplicates (e.g. an endpoint
        named directly and also via a bundle) are de-duplicated, preserving
        order.
        """
        ordered: list[str] = []
        for name in names:
            members = self.bundles.get(name, [name])
            for member in members:
                if member not in ordered:
                    ordered.append(member)
        return [self.resolve_endpoint(n, sharing=sharing) for n in ordered]

    def _resolve_vllm(
        self, ep: EndpointSpec, sharing: str
    ) -> EndpointRequest:
        model = self.models[ep.model]
        rt = ep.runtime
        served_name = ep.served_name or ep.name
        structural = vllm_structural(
            model_ref=model.source,
            revision=model.revision,
            quantization=model.quantization,
            dtype=model.dtype,
            tensor_parallel_size=rt.get('tensor_parallel_size', 1),
            pipeline_parallel_size=rt.get('pipeline_parallel_size', 1),
            image=rt.get('image'),
            chat_template=rt.get('chat_template'),
            trust_remote_code=rt.get('trust_remote_code', False),
            lora_adapters=rt.get('lora_adapters'),
            served_name=served_name,
        )
        capacity: dict[str, Any] = {}
        if rt.get('max_model_len') is not None:
            capacity['max_model_len'] = rt['max_model_len']
        spec = {
            'engine': VLLM,
            'hf_model_id': model.hf_model_id,
            'served_model_name': served_name,
            'runtime': dict(rt),
            'reclaim': ep.reclaim,
        }
        served = {
            'served_model_name': served_name,
            'hf_model_id': model.hf_model_id,
            'protocol': ep.protocol,
        }
        return EndpointRequest(
            endpoint=ep.name,
            engine=VLLM,
            structural=structural,
            capacity=capacity,
            sharing=sharing,
            spec=spec,
            served=served,
        )

    def _resolve_ollama(
        self, ep: EndpointSpec, sharing: str
    ) -> EndpointRequest:
        host = self.hosts[ep.host]
        settings = host.settings
        structural = ollama_structural(
            host=host.name,
            gpu_indices=host.gpu_indices,
            keep_alive=settings.get('keep_alive'),
            num_parallel=settings.get('num_parallel'),
            max_loaded_models=settings.get('max_loaded_models'),
            model_store=host.model_store,
        )
        spec = {
            'engine': OLLAMA,
            'host': host.name,
            'image': host.image,
            'gpu_indices': host.gpu_indices,
            'settings': dict(settings),
            'model_store': host.model_store,
            'reclaim': ep.reclaim,
        }
        served = {'model': ep.model}
        return EndpointRequest(
            endpoint=ep.name,
            engine=OLLAMA,
            structural=structural,
            capacity={},
            sharing=sharing,
            spec=spec,
            served=served,
            host=host.name,
        )
