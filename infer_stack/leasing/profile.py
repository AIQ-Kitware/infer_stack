"""The recovery snapshot: every non-ledger render input frozen in the ledger.

Without it, each command rebuilt the backend from its own flags, settings, the
installed package's image pins and whatever ``--catalog`` it was given. Two
processes, or one process before and after an upgrade, could render different
projects from the same ledger. With serialised publication, a recovery
re-renders, and could then recreate unrelated services.

The first controller mutation against a ledger with no profile freezes that
invocation's resolved settings. Later operations render from the stored copy.
For the ordinary ``config init -> catalog suggest --apply -> acquire`` workflow,
acquire advances this snapshot automatically: compatible catalog additions are
merged into a live epoch, and a quiescent stack adopts the current invocation
profile wholesale. ``infer-stack config publish`` remains an advanced explicit
pre-seed/preview operation; it is not a required third configuration step.

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
    """Two catalogs in one published union define the same name differently.

    ``names`` carries the colliding names, so a caller can ask the question
    that decides whether the collision matters: is any of them pinned by a
    workload that is actually resident?
    """

    def __init__(self, message: str, names=()):
        super().__init__(message)
        self.names = list(names)


class ProfileMismatch(RuntimeError):
    """A request or backend does not match the active recovery snapshot."""


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
                            'published catalogs', [name]
                        )
                    continue
                self._owner[name] = cat
                self.endpoints[name] = spec
            for name, members in cat.bundles.items():
                if name in self.bundles and self.bundles[name] != list(members):
                    raise CatalogConflict(
                        f'bundle {name!r} is defined differently in two published catalogs',
                        [name]
                    )
                self.bundles[name] = list(members)
            for alias, row in _registry_incoming_from_catalog(cat).items():
                if alias in routes and routes[alias] != row:
                    raise CatalogConflict(
                        f'route {alias!r} is defined differently in two published catalogs',
                        [alias]
                    )
                routes[alias] = row
        clash = sorted(set(self.bundles) & set(self.endpoints))
        if clash:
            raise CatalogConflict(
                f'{clash[0]!r} is an endpoint in one published catalog and a bundle in another',
                clash
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
    """Keys where this invocation's settings differ from the recovery snapshot.

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


def merge_catalog_sources(
    published: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append compatible catalog snapshots, deduplicated by content.

    This is the live-epoch catalog rule: an acquire may add definitions without
    mutating any definition the recovery snapshot already knows.  Building the
    union is the semantic conflict check -- if an incoming endpoint/bundle/route
    resolves differently, :class:`CatalogConflict` is raised before anything is
    written.

    The snapshots are intentionally retained rather than replacing the old
    source while workloads are resident.  Once the stack is quiescent the next
    acquire replaces the profile wholesale with the current user configuration,
    compacting this history back to the authoritative source(s).
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in [*(published or []), *(incoming or [])]:
        digest = canonical_digest(source)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(dict(source or {}))
    # Validate the semantic union now, before a controller persists it.
    CatalogUnion.from_sources(out)
    return out


def prune_catalog_sources(
    sources: list[dict[str, Any]], keep: set[str]
) -> list[dict[str, Any]]:
    """Snapshot sources reduced to the definitions ``keep`` still pins.

    A frozen snapshot only has to protect what resident workloads are running.
    Editing an endpoint nothing is serving -- the normal case while iterating
    with ``catalog endpoint add --force`` -- should not be refused merely
    because an older snapshot also defined it, so the stale definitions of
    everything else are dropped before the union is rebuilt.
    """
    pruned: list[dict[str, Any]] = []
    for source in sources or []:
        endpoints = {
            name: spec for name, spec in (source.get('endpoints') or {}).items()
            if name in keep
        }
        bundles = {
            name: members for name, members in (source.get('bundles') or {}).items()
            if name in keep
        }
        if not endpoints and not bundles:
            continue
        kept = dict(source)
        kept['endpoints'] = endpoints
        kept['bundles'] = bundles
        pruned.append(kept)
    return pruned


def validate_requests_against(union: Any, requests) -> None:
    """Refuse requests a published catalog union does not define identically.

    ``union`` of ``None`` (nothing published) accepts everything. Reservations
    are not catalog endpoints and always pass.
    """
    from .models import RESERVED_ENGINE

    if not isinstance(union, CatalogUnion):
        return
    for req in requests:
        if req.engine == RESERVED_ENGINE:
            continue
        if req.endpoint not in union.endpoints:
            raise ProfileMismatch(
                f'endpoint {req.endpoint!r} is not in the active recovery snapshot. '
                'Acquire normally imports compatible additions from the current '
                'catalog automatically; if this persists, the catalog conflicts '
                'with definitions already frozen for resident workloads'
            )
        if not union.request_matches(req):
            raise ProfileMismatch(
                f'endpoint {req.endpoint!r} differs from the definition frozen for '
                'the active leasing epoch. Quiesce the managed stack (release/evict '
                'resident deployments), then retry; the next acquire adopts the '
                'current user catalog automatically'
            )


def check_invocation_catalog(union: Any, catalog: Any) -> None:
    """Compatibility helper for callers that explicitly require a seeded union.

    The ordinary acquire path no longer uses this gate: it resolves from the
    current user catalog and advances the recovery snapshot automatically.
    Advanced multi-runbook code may still use this helper when it specifically
    wants to require membership in a pre-seeded union.
    """
    if not isinstance(union, CatalogUnion) or catalog is None:
        return
    source = getattr(catalog, 'source', None)
    if source is None or canonical_digest(source) in set(union.digests):
        return
    raise ProfileMismatch(
        'this catalog differs from the active recovery snapshot. Ordinary acquire '
        'adopts compatible catalog additions automatically; use explicit profile '
        'publication only for advanced multi-catalog pre-seeding'
    )
