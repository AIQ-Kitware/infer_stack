"""The published profile: every non-ledger render input, frozen in the ledger.

Without it, each command rebuilt the backend from its own flags, settings, the
installed package's image pins and whatever ``--catalog`` it was given. Two
processes, or one process before and after an upgrade, could render different
projects from the same ledger. With serialised publication, a recovery
re-renders, and could then recreate unrelated services.

The first controller mutation against a ledger with no profile freezes that
invocation's resolved settings. Later operations render from the stored copy
and warn when their own settings differ. ``infer-stack config publish`` replaces
it while the stack is quiescent.

Not in the profile:

* secrets, which live in the managed ``.env``;
* ``allowed_gpus``, a per-caller admission scope (see
  :meth:`Controller._mark_pending`'s placement context).

Catalogs are a **published union**. Endpoints, bundles and route rows from
several catalogs are merged on what they resolve to. Identical definitions are
deduplicated; the same name defined differently raises :class:`CatalogConflict`.

Example:
    >>> a = {'models': {'m': {'source': 'hf://org/m'}},
    ...      'endpoints': {'e': {'engine': 'vllm', 'model': 'm'}}}
    >>> b = {'models': {'other': {'source': 'hf://org/m'}},
    ...      'endpoints': {'e': {'engine': 'vllm', 'model': 'other'},
    ...                    'f': {'engine': 'vllm', 'model': 'other'}}}
    >>> union = CatalogUnion.from_sources([a, b])
    >>> sorted(union.endpoints)
    ['e', 'f']
    >>> c = {'models': {'m': {'source': 'hf://org/different'}},
    ...      'endpoints': {'e': {'engine': 'vllm', 'model': 'm'}}}
    >>> CatalogUnion.from_sources([a, c])
    Traceback (most recent call last):
    ...
    infer_stack.leasing.profile.CatalogConflict: endpoint 'e' is defined differently in two published catalogs
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .catalog import Catalog, CatalogError

PROFILE_VERSION = 1


class CatalogConflict(CatalogError):
    """Two catalogs in one published union define the same name differently."""


class ProfileMismatch(RuntimeError):
    """A request or backend does not match the published profile."""


def canonical_digest(data: Any) -> str:
    """Stable digest of JSON-compatible data (key order does not matter)."""
    text = json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _request_key(catalog: Catalog, name: str, sharing: str | None = None) -> Any:
    try:
        return dataclasses.asdict(catalog.resolve_endpoint(name, sharing=sharing))
    except CatalogError as ex:
        return {'unresolvable': str(ex)}


class CatalogUnion:
    """Several catalogs presented as one, for resolution and route rendering.

    Implements the part of :class:`Catalog` the leasing paths use:
    ``endpoints``, ``bundles``, ``resolve_endpoint`` and ``resolve_names``.
    """

    def __init__(self, sources: list[dict[str, Any]], catalogs: list[Catalog]):
        self.sources = sources
        self.catalogs = catalogs
        self._owner: dict[str, Catalog] = {}
        self.endpoints: dict[str, Any] = {}
        self.bundles: dict[str, list[str]] = {}
        from .compose import _registry_incoming_from_catalog

        routes: dict[str, Any] = {}
        for cat in catalogs:
            for name, spec in cat.endpoints.items():
                if name in self._owner:
                    if _request_key(self._owner[name], name) != _request_key(cat, name):
                        raise CatalogConflict(
                            f'endpoint {name!r} is defined differently in two '
                            'published catalogs'
                        )
                    continue
                self._owner[name] = cat
                self.endpoints[name] = spec
            for name, members in cat.bundles.items():
                if name in self.bundles and self.bundles[name] != list(members):
                    raise CatalogConflict(
                        f'bundle {name!r} is defined differently in two published catalogs'
                    )
                self.bundles[name] = list(members)
            for alias, row in _registry_incoming_from_catalog(cat).items():
                if alias in routes and routes[alias] != row:
                    raise CatalogConflict(
                        f'route {alias!r} is defined differently in two published catalogs'
                    )
                routes[alias] = row
        clash = sorted(set(self.bundles) & set(self.endpoints))
        if clash:
            raise CatalogConflict(
                f'{clash[0]!r} is an endpoint in one published catalog and a bundle in another'
            )

    @classmethod
    def from_sources(cls, sources: list[dict[str, Any]]) -> CatalogUnion:
        catalogs = [Catalog.from_dict(src) for src in sources]
        return cls([dict(src or {}) for src in sources], catalogs)

    @property
    def digests(self) -> list[str]:
        return [canonical_digest(src) for src in self.sources]

    def resolve_endpoint(self, name: str, *, sharing: str | None = None):
        if name not in self._owner:
            if self.catalogs:
                # Reuse the catalog's helpful unknown-name message.
                raise self._merged_view()._unknown_endpoint_error(name)
            raise CatalogError(f"unknown endpoint '{name}'")
        return self._owner[name].resolve_endpoint(name, sharing=sharing)

    def resolve_names(self, names: list[str], *, sharing: str | None = None):
        ordered: list[str] = []
        for name in names:
            for member in self.bundles.get(name, [name]):
                if member not in ordered:
                    ordered.append(member)
        return [self.resolve_endpoint(n, sharing=sharing) for n in ordered]

    def request_matches(self, request) -> bool:
        """Whether ``request`` is exactly what this union resolves its name to."""
        cat = self._owner.get(request.endpoint)
        if cat is None:
            return False
        return _request_key(cat, request.endpoint, request.sharing) == dataclasses.asdict(request)

    def _merged_view(self) -> Catalog:
        view = Catalog()
        for cat in self.catalogs:
            view.models.update(cat.models)
        view.endpoints = dict(self.endpoints)
        view.bundles = dict(self.bundles)
        return view


def catalog_sources(catalog: Any) -> list[dict[str, Any]]:
    """The raw catalog mappings behind ``catalog`` (a Catalog, a union, or None)."""
    if catalog is None:
        return []
    if isinstance(catalog, CatalogUnion):
        return list(catalog.sources)
    source = getattr(catalog, 'source', None)
    if source is None:
        raise ProfileMismatch(
            'this catalog has no source mapping to publish (build it with '
            'Catalog.from_dict or Catalog.load)'
        )
    return [source]


def profile_drift(published: dict[str, Any], invocation: dict[str, Any]) -> list[str]:
    """Keys where this invocation's settings differ from the published profile.

    Catalogs drift only when the invocation names one that is not part of the
    published union; naming a subset is the normal multi-runbook case.
    """
    drift = []
    for key in sorted(set(published) | set(invocation)):
        if key in {'version'}:
            continue
        if key == 'catalogs':
            have = {canonical_digest(s) for s in published.get(key) or []}
            want = {canonical_digest(s) for s in invocation.get(key) or []}
            if not want <= have:
                drift.append(key)
            continue
        if published.get(key) != invocation.get(key):
            drift.append(key)
    return drift
