"""What a catalog endpoint means: an alias, a protocol, and a target.

An endpoint is the public name a client asks for. Its *target* says who
fulfils it:

* :class:`ManagedTarget` -- a runtime infer-stack realizes (vLLM, Ollama),
  protected by a lease; it derives the ledger's :class:`EndpointRequest`;
* :class:`ExternalTarget` -- an OpenAI-compatible server that already runs
  elsewhere; nothing in the ledger.

:class:`ResolvedEndpoint` is the one normalized meaning of an endpoint.
Catalog-union conflicts, profile drift and "did this endpoint change" all
compare its :meth:`~ResolvedEndpoint.semantic_key`, which is built from the
resolved request (model source, revision, runtime knobs...), never from the
catalog's own model keys, so two catalogs that name one model differently
still agree. It knows nothing of leases, routes or backends.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .models import EndpointRequest


@dataclass(frozen=True)
class ManagedTarget:
    """A runtime infer-stack realizes. ``request`` is the catalog's own
    request (its declared sharing); :meth:`ResolvedEndpoint.to_request` applies
    a request-time override."""

    request: EndpointRequest

    kind = 'managed'


@dataclass(frozen=True)
class ExternalTarget:
    """An OpenAI-compatible server infer-stack does not run.

    ``model`` is what that server expects as ``model``; ``api_key_env`` names
    the variable holding its key (never the key).
    """

    api_base: str
    model: str
    api_key_env: str | None = None

    kind = 'external'


@dataclass(frozen=True)
class ResolvedEndpoint:
    """One endpoint's meaning: ``alias`` (what clients request), ``protocol``
    (``chat`` / ``completions``) and ``target``.

    Callers from before this type read ``EndpointRequest`` fields off
    ``Catalog.resolve_endpoint(name)`` (``engine``, ``served``, ``spec``,
    ``compat_key``...); for a managed target those still work, delegated to
    its request. New code uses :meth:`to_request`.

    Example:
        >>> from infer_stack.leasing.models import vllm_structural
        >>> req = EndpointRequest('qwen', 'vllm', vllm_structural(model_ref='hf://org/m'),
        ...                       served={'served_model_name': 'qwen', 'protocol': 'chat'})
        >>> local = ResolvedEndpoint('qwen', 'chat', ManagedTarget(req))
        >>> remote = ResolvedEndpoint('qwen', 'chat', ExternalTarget('http://box/v1', 'Qwen/Q'))
        >>> local.managed, remote.managed, local.engine
        (True, False, 'vllm')
        >>> local.semantic_key() == remote.semantic_key()
        False
        >>> remote.to_request()
        Traceback (most recent call last):
        ...
        ValueError: 'qwen' is externally provided; it has no lease request
    """

    alias: str
    protocol: str
    target: ManagedTarget | ExternalTarget

    @property
    def managed(self) -> bool:
        return isinstance(self.target, ManagedTarget)

    def to_request(self, *, sharing_override: str | None = None) -> EndpointRequest:
        """The ledger request for a managed target (``--dedicated`` overrides
        the catalog's sharing); an external target has none."""
        if not isinstance(self.target, ManagedTarget):
            raise ValueError(f'{self.alias!r} is externally provided; it has no lease request')
        request = self.target.request
        if sharing_override and sharing_override != request.sharing:
            request = dataclasses.replace(request, sharing=sharing_override)
        return request

    def semantic(self) -> dict[str, Any]:
        """The endpoint's meaning as plain data (what :meth:`semantic_key` hashes)."""
        if isinstance(self.target, ManagedTarget):
            target: dict[str, Any] = {'kind': 'managed',
                                      'request': dataclasses.asdict(self.target.request)}
        else:
            target = {'kind': 'external', **dataclasses.asdict(self.target)}
        return {'alias': self.alias, 'protocol': self.protocol, 'target': target}

    def semantic_key(self) -> str:
        """A stable digest of :meth:`semantic`: equal iff the endpoint means the same."""
        text = json.dumps(self.semantic(), sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    # -- EndpointRequest fields, for callers from before this type -------------

    def _request(self) -> EndpointRequest:
        return self.to_request()

    @property
    def endpoint(self) -> str:
        return self.alias

    @property
    def engine(self) -> str:
        return self._request().engine

    @property
    def structural(self) -> dict[str, Any]:
        return self._request().structural

    @property
    def capacity(self) -> dict[str, Any]:
        return self._request().capacity

    @property
    def sharing(self) -> str:
        return self._request().sharing

    @property
    def spec(self) -> dict[str, Any]:
        return self._request().spec

    @property
    def served(self) -> dict[str, Any]:
        return self._request().served

    @property
    def host(self) -> str | None:
        return self._request().host

    @property
    def compat_key(self) -> str:
        return self._request().compat_key
