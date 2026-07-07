"""Data model for the leasing ledger.

This is the *controller* half of infer-stack (see the redesign doc
``dev/infer-stack-redesign-critique.md`` in the aiq-eval-runner repo). The
compiler half (``resolver``/``validator``/``renderer``) turns a desired
deployment into backend artifacts; the ledger tracks *who wants what now* so
that multiple users / pipeline nodes can ``acquire`` and ``release`` models
without clobbering each other.

Three record kinds, in three layers:

* :class:`EndpointRequest` — catalog-resolved input to the ledger. The
  catalog/resolver layer builds these; the ledger consumes them and never needs
  to know engine-specific details beyond what is packed in here.
* :class:`Lease` — who wants a set of endpoints, with a soft TTL.
* :class:`Deployment` — compatible requests coalesced into one realizable
  backend deployment. ``demand`` is the number of *protecting* leases pointing
  at it.

The unit that gets reference-counted is **the thing that gets a process**:
one model for vLLM (one process per model), one daemon for Ollama (one daemon
serves many tags). Both are expressed uniformly through the
:func:`compatibility_key`: for vLLM the structural fields describe the model +
runtime; for Ollama they describe the *daemon* config, so many tag endpoints
coalesce onto one deployment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class LeaseState:
    """Lifecycle states for a :class:`Lease`."""

    ACTIVE = 'active'
    EXPIRED = 'expired'
    RELEASED = 'released'


class DeploymentState:
    """Lifecycle states for a :class:`Deployment`.

    ``LIVE``/``IDLE`` are driven by the ledger from demand. ``RECLAIMING``/
    ``STOPPED`` are owned by the (future) reconciler/backend and are reserved
    here for forward compatibility.
    """

    LIVE = 'live'
    IDLE = 'idle'
    RECLAIMING = 'reclaiming'
    STOPPED = 'stopped'


class Sharing:
    """Sharing policy a requester can ask for."""

    SHARED = 'shared-compatible'
    DEDICATED = 'dedicated'


# Fields that must match *exactly* for two requests to share one deployment.
# Capacity fields (e.g. ``max_model_len``) are deliberately NOT here: they are
# handled by subsumption (existing >= requested) in
# :func:`capacity_satisfies`, so a 32k deployment can serve an 8k request.
VLLM_STRUCTURAL_FIELDS = (
    'engine',
    'model_ref',
    'revision',
    'quantization',
    'dtype',
    'tensor_parallel_size',
    'pipeline_parallel_size',
    'data_parallel_size',
    'image',
    'chat_template',
    'trust_remote_code',
    'lora_adapters',
    'served_name',
)

# For Ollama the coalescing unit is the *daemon*, so the structural identity is
# the host config, not the model tag (tags load/unload inside the daemon).
OLLAMA_STRUCTURAL_FIELDS = (
    'engine',
    'host',
    'gpu_indices',
    'keep_alive',
    'num_parallel',
    'max_loaded_models',
    'model_store',
)


def _canonical(value: Any) -> Any:
    """Normalize a value so equal-meaning configs hash identically."""
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def compatibility_key(engine: str, structural: dict[str, Any]) -> str:
    """Stable hash identifying deployments that may be shared.

    Two :class:`EndpointRequest` s with the same ``compatibility_key`` describe
    the *same realizable deployment* and are eligible to coalesce (subject to
    capacity subsumption and sharing policy, which are handled separately so the
    key stays a pure structural identity).

    Args:
        engine: the serving engine (``'vllm'`` / ``'ollama'``).
        structural: the engine-appropriate structural fields (see
            :func:`vllm_structural` / :func:`ollama_structural`).

    Returns:
        A 16-hex-char digest.

    Example:
        >>> a = vllm_structural(model_ref='qwen', tensor_parallel_size=1)
        >>> b = vllm_structural(model_ref='qwen', tensor_parallel_size=1)
        >>> c = vllm_structural(model_ref='qwen', tensor_parallel_size=2)
        >>> compatibility_key('vllm', a) == compatibility_key('vllm', b)
        True
        >>> compatibility_key('vllm', a) == compatibility_key('vllm', c)
        False
    """
    payload = {'engine': engine, 'structural': _canonical(structural)}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def vllm_structural(
    *,
    model_ref: str,
    revision: str | None = None,
    quantization: str | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    image: str | None = None,
    chat_template: str | None = None,
    trust_remote_code: bool = False,
    lora_adapters: list[str] | None = None,
    attention_backend: str | None = None,
    served_name: str | None = None,
) -> dict[str, Any]:
    """Build the structural dict for a vLLM endpoint (one process per model)."""
    return {
        'engine': 'vllm',
        'model_ref': model_ref,
        'revision': revision,
        'quantization': quantization,
        'dtype': dtype,
        'tensor_parallel_size': tensor_parallel_size,
        'pipeline_parallel_size': pipeline_parallel_size,
        'data_parallel_size': data_parallel_size,
        'image': image,
        'chat_template': chat_template,
        'trust_remote_code': trust_remote_code,
        'lora_adapters': sorted(lora_adapters or []),
        # The attention backend (VLLM_ATTENTION_BACKEND) changes the engine's
        # numerics, so two endpoints differing only in it must be distinct
        # deployments (not coalesced onto one process).
        'attention_backend': attention_backend,
        'served_name': served_name or model_ref,
    }


def ollama_structural(
    *,
    host: str,
    gpu_indices: list[int] | None = None,
    keep_alive: str | None = None,
    num_parallel: int | None = None,
    max_loaded_models: int | None = None,
    model_store: str | None = None,
) -> dict[str, Any]:
    """Build the structural dict for an Ollama daemon (one daemon, many tags).

    The model *tag* is intentionally absent — it is per-endpoint detail, not
    part of the daemon's identity — so several tag endpoints sharing a host
    config coalesce onto a single daemon deployment.
    """
    return {
        'engine': 'ollama',
        'host': host,
        'gpu_indices': list(gpu_indices or []),
        'keep_alive': keep_alive,
        'num_parallel': num_parallel,
        'max_loaded_models': max_loaded_models,
        'model_store': model_store,
    }


def capacity_satisfies(
    existing: dict[str, Any], requested: dict[str, Any]
) -> bool:
    """Whether an existing deployment's capacity covers a new request.

    Coalescing is *subsumption*, not equality: every capacity field the request
    needs must be present on the existing deployment and at least as large
    (e.g. a 32k ``max_model_len`` deployment serves an 8k request, but not the
    reverse). An empty request (e.g. Ollama, which has no capacity fields) is
    trivially satisfied.

    Example:
        >>> capacity_satisfies({'max_model_len': 32768}, {'max_model_len': 8192})
        True
        >>> capacity_satisfies({'max_model_len': 8192}, {'max_model_len': 32768})
        False
        >>> capacity_satisfies({}, {'max_model_len': 8192})
        False
        >>> capacity_satisfies({'max_model_len': 8192}, {})
        True
    """
    for key, need in requested.items():
        have = existing.get(key)
        if have is None or have < need:
            return False
    return True


@dataclass(frozen=True)
class EndpointRequest:
    """A catalog-resolved request for one served endpoint.

    Produced by the catalog/resolver layer, consumed by the ledger. Carries
    everything the ledger needs to coalesce without knowing engine specifics.

    Attributes:
        endpoint: public served name the user asked for.
        engine: ``'vllm'`` / ``'ollama'``.
        structural: fields that must match exactly to coalesce (see
            :func:`vllm_structural` / :func:`ollama_structural`).
        capacity: subsumption fields where existing must be ``>=`` requested.
        sharing: :class:`Sharing` policy.
        spec: opaque host-realization payload the backend uses to start the
            process/daemon (shared across coalesced requests).
        served: opaque per-endpoint payload (e.g. the Ollama tag) merged into
            the deployment's served map under ``endpoint``.
        host: Ollama daemon name; ``None`` for auto-synthesized vLLM hosts.
    """

    endpoint: str
    engine: str
    structural: dict[str, Any]
    capacity: dict[str, Any] = field(default_factory=dict)
    sharing: str = Sharing.SHARED
    spec: dict[str, Any] = field(default_factory=dict)
    served: dict[str, Any] = field(default_factory=dict)
    host: str | None = None

    @property
    def compat_key(self) -> str:
        return compatibility_key(self.engine, self.structural)


@dataclass
class Lease:
    """Who wants a set of endpoints, with a soft TTL.

    A lease *protects* its deployment deployments while it is ``ACTIVE`` and not past
    its TTL. Protection is what ``demand`` counts.
    """

    id: str
    owner: str
    state: str
    created_at: float
    ttl_seconds: float | None
    expires_at: float | None
    heartbeat_at: float
    endpoints: list[str] = field(default_factory=list)
    deployment_ids: list[str] = field(default_factory=list)

    def is_protecting(self, now: float) -> bool:
        """True if this lease still protects its allocations at ``now``."""
        if self.state != LeaseState.ACTIVE:
            return False
        return self.expires_at is None or self.expires_at > now


@dataclass
class Deployment:
    """Compatible requests coalesced into one realizable deployment.

    ``served`` maps endpoint name -> per-endpoint payload; for vLLM it has one
    entry, for Ollama one per tag served by the daemon. ``demand`` is the count
    of protecting leases and is computed on read, not stored.
    """

    id: str
    compat_key: str
    engine: str
    sharing: str
    capacity: dict[str, Any]
    spec: dict[str, Any]
    served: dict[str, Any]
    state: str
    created_at: float
    updated_at: float
    demand: int = 0
