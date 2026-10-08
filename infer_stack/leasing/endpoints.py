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

Three names, and only these (new code uses them; older spellings stay
readable):

* **endpoint alias** -- what a client requests through the front door, the
  catalog's endpoint name (``ResolvedEndpoint.alias``). It is the public name.
* **upstream model** -- what the serving process expects as ``model``: a
  managed vLLM's ``--served-model-name`` (catalog ``served_name``, also spelled
  ``public_name``; defaults to the alias), an Ollama tag, or
  ``ExternalTarget.model``. Not public: clients never send it.
* **deployment id** -- one managed realization of an endpoint, when there is
  one (external targets have none).
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


#: Variables infer-stack itself keeps in the gateway's .env. An external
#: target may not name one as its key: the gateway would send infer-stack's
#: own credential to a server the catalog names.
RESERVED_KEY_ENVS = frozenset({
    'LITELLM_MASTER_KEY', 'LITELLM_SALT_KEY', 'LITELLM_DB_PASSWORD',
    'WEBUI_SECRET_KEY', 'HF_TOKEN',
})

#: Endpoint keys that describe a runtime infer-stack realizes, so they mean
#: nothing beside ``external:``.
MANAGED_ONLY_KEYS = ('engine', 'model', 'host', 'runtime', 'placement', 'sharing',
                     'reclaim', 'served_name', 'public_name')


def external_errors(name: str, raw: dict[str, Any], external: Any) -> list[str]:
    """What is wrong with endpoint ``name`` whose ``external:`` is ``external``
    (``raw`` is the endpoint's whole mapping).

    >>> external_errors('q', {'external': {}, 'runtime': {}}, {'api_base': 'box', 'model': ''})
    ["endpoint 'q' is external: 'runtime' describes a runtime infer-stack runs and does not apply", "endpoint 'q': external.api_base must be an http(s) URL, not 'box'", "endpoint 'q': external.model is required (the name the upstream server expects)"]
    >>> external_errors('q', {}, {'api_base': 'http://b/v1', 'model': 'm', 'api_key_env': 'HF_TOKEN'})
    ["endpoint 'q': external.api_key_env may not be HF_TOKEN, one of infer-stack's own secrets"]
    """
    import re

    errors = [f"endpoint {name!r} is external: {key!r} describes a runtime "
              'infer-stack runs and does not apply'
              for key in MANAGED_ONLY_KEYS if key in raw]
    if not isinstance(external, dict):
        return errors + [f"endpoint {name!r}: 'external' must be a mapping "
                         '(api_base, model, optional api_key_env)']
    unknown = sorted(set(external) - {'api_base', 'model', 'api_key_env'})
    if unknown:
        errors.append(f"endpoint {name!r}: unknown external key(s) {unknown} "
                      '(api_base, model, api_key_env)')
    base = external.get('api_base')
    if not _is_http_url(base):
        errors.append(f"endpoint {name!r}: external.api_base must be an http(s) URL, "
                      f'not {base!r}')
    if not external.get('model') or not isinstance(external.get('model'), str):
        errors.append(f"endpoint {name!r}: external.model is required (the name the "
                      'upstream server expects)')
    key_env = external.get('api_key_env')
    if key_env is not None:
        if not isinstance(key_env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env):
            errors.append(f"endpoint {name!r}: external.api_key_env must be an "
                          f'environment variable name, not {key_env!r}')
        elif key_env in RESERVED_KEY_ENVS:
            errors.append(f"endpoint {name!r}: external.api_key_env may not be "
                          f"{key_env}, one of infer-stack's own secrets")
    return errors


def _is_http_url(value: Any) -> bool:
    """An absolute http(s) URL with a host, no whitespace, a valid port.

    >>> [_is_http_url(v) for v in ('http://box:8000/v1', 'https://h/v1', 'box:8000',
    ...  'http:///v1', 'http://a b/v1', 'ftp://h/', 'http://h:99999/', None)]
    [True, True, False, False, False, False, False, False]
    """
    from urllib.parse import urlsplit

    if not isinstance(value, str) or any(c.isspace() for c in value):
        return False
    try:
        parts = urlsplit(value)
        parts.port                      # raises on a malformed port
    except ValueError:
        return False
    return parts.scheme in ('http', 'https') and bool(parts.hostname)


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


def external_endpoints(catalog: Any) -> list[tuple[str, ExternalTarget]]:
    """``[(alias, target), ...]`` for every external endpoint of ``catalog``."""
    out = []
    for alias in sorted(getattr(catalog, 'endpoints', None) or {}):
        try:
            target = catalog.resolve_endpoint(alias).target
        except Exception:  # noqa: BLE001 - an invalid endpoint is not listed
            continue
        if isinstance(target, ExternalTarget):
            out.append((alias, target))
    return out


def key_references(catalog: Any) -> dict[str, list[str]]:
    """``{variable: [endpoint alias, ...]}`` for every external endpoint of
    ``catalog`` that names a key.

    >>> from infer_stack.leasing.catalog import Catalog
    >>> key_references(Catalog.from_dict({'endpoints': {'r': {'external': {
    ...     'api_base': 'http://b/v1', 'model': 'm', 'api_key_env': 'R_KEY'}}}}))
    {'R_KEY': ['r']}
    """
    refs: dict[str, list[str]] = {}
    for alias in sorted(getattr(catalog, 'endpoints', None) or {}):
        try:
            target = catalog.resolve_endpoint(alias).target
        except Exception:  # noqa: BLE001 - an invalid endpoint names no key
            continue
        if isinstance(target, ExternalTarget) and target.api_key_env:
            refs.setdefault(target.api_key_env, []).append(alias)
    return refs


@dataclass(frozen=True)
class ResolvedEndpoint:
    """One endpoint's meaning: ``alias`` (what clients request), ``protocol``
    (``chat`` / ``completions``) and ``target``.

    New code should use :meth:`to_request` when it needs the managed ledger
    shape.  The read-only properties below are compatibility views for callers
    that predate this type; they are derived from the managed target and are not
    a second source of deployment identity or capacity.

    Example:
        >>> from infer_stack.leasing.models import vllm_structural
        >>> req = EndpointRequest('qwen', 'vllm', vllm_structural(model_ref='hf://org/m'),
        ...                       served={'served_model_name': 'qwen', 'protocol': 'chat'})
        >>> local = ResolvedEndpoint('qwen', 'chat', ManagedTarget(req))
        >>> remote = ResolvedEndpoint('qwen', 'chat', ExternalTarget('http://box/v1', 'Qwen/Q'))
        >>> local.managed, remote.managed, local.to_request().engine
        (True, False, 'vllm')
        >>> local.semantic_key() == remote.semantic_key()
        False
        >>> remote.to_request()
        Traceback (most recent call last):
        ...
        ValueError: 'qwen' is externally provided and does not require a lease; use `infer-stack access qwen` or `infer-stack run ...`
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
            raise ValueError(
                f'{self.alias!r} is externally provided and does not require a lease; '
                f'use `infer-stack access {self.alias}` or `infer-stack run ...`')
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

    # -- Compatibility-only EndpointRequest views ---------------------------
    #
    # These preserve the pre-ResolvedEndpoint Python API without giving this
    # object a second copy of managed deployment state.  Repository code uses
    # ``to_request()`` explicitly; external callers may keep reading these
    # fields while migrating.

    def _compat_request(self) -> EndpointRequest:
        return self.to_request()

    @property
    def endpoint(self) -> str:
        return self.alias

    @property
    def engine(self) -> str:
        return self._compat_request().engine

    @property
    def structural(self) -> dict[str, Any]:
        return self._compat_request().structural

    @property
    def capacity(self) -> dict[str, Any]:
        return self._compat_request().capacity

    @property
    def sharing(self) -> str:
        return self._compat_request().sharing

    @property
    def spec(self) -> dict[str, Any]:
        return self._compat_request().spec

    @property
    def served(self) -> dict[str, Any]:
        return self._compat_request().served

    @property
    def host(self) -> str | None:
        return self._compat_request().host

    @property
    def compat_key(self) -> str:
        return self._compat_request().compat_key
