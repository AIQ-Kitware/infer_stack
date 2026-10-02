"""The recovery snapshot: every non-ledger render input frozen in the ledger.

Without it, each command rebuilt the backend from its own flags, settings, the
installed package's image pins and whatever ``--catalog`` it was given. Two
processes, or one process before and after an upgrade, could render different
projects from the same ledger. With serialised publication, a recovery
re-renders, and could then recreate unrelated services.

The first controller mutation against a ledger with no profile freezes that
invocation's resolved settings. Later operations render from the stored copy.
For the ordinary ``config init -> catalog suggest --apply -> acquire`` workflow,
acquire advances this snapshot automatically: the invocation's catalogs are
merged into the published endpoint definitions (:func:`adopt_catalog_sources`),
and a quiescent stack adopts the invocation's settings. ``infer-stack config publish`` remains an advanced explicit
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


def _endpoint_key(catalog: Catalog, name: str) -> str:
    """What ``name`` means in ``catalog`` (its semantic key), or why it cannot."""
    try:
        return catalog.resolve_endpoint(name).semantic_key()
    except CatalogError as ex:
        return f'unresolvable: {ex}'


class CatalogUnion:
    """Several catalogs presented as one, for resolution and route rendering.

    Implements the part of :class:`Catalog` the leasing paths use:
    ``endpoints``, ``bundles``, ``resolve_endpoint``, ``resolve`` and
    ``resolve_requests``.
    """

    def __init__(self, sources: list[dict[str, Any]], catalogs: list[Catalog]):
        self.sources = sources
        self.catalogs = catalogs
        self._owner: dict[str, Catalog] = {}
        self.endpoints: dict[str, Any] = {}
        self.bundles: dict[str, list[str]] = {}
        # Conflicts are endpoint meaning only: routes are derived from it.
        for cat in catalogs:
            for name, spec in cat.endpoints.items():
                if name in self._owner:
                    if _endpoint_key(self._owner[name], name) != _endpoint_key(cat, name):
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

    def expand(self, names: list[str]) -> list[str]:
        ordered: list[str] = []
        for name in names:
            for member in self.bundles.get(name, [name]):
                if member not in ordered:
                    ordered.append(member)
        return ordered

    def resolve(self, names: list[str]):
        return [self.resolve_endpoint(n) for n in self.expand(names)]

    def resolve_requests(self, names: list[str], *, sharing: str | None = None):
        try:
            return [self.resolve_endpoint(n).to_request(sharing_override=sharing)
                    for n in self.expand(names)]
        except ValueError as ex:        # an external member: no lease request
            raise CatalogError(str(ex)) from ex

    resolve_names = resolve_requests

    def request_matches(self, request) -> bool:
        """Whether ``request`` is exactly what this union resolves its name to."""
        cat = self._owner.get(request.endpoint)
        if cat is None:
            return False
        try:
            mine = cat.resolve_endpoint(request.endpoint).to_request(
                sharing_override=request.sharing)
        except (CatalogError, ValueError):
            return False
        return dataclasses.asdict(mine) == dataclasses.asdict(request)

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

    Catalogs drift only when the invocation defines an endpoint the published
    union does not define identically (by meaning, not by source bytes: the
    union's sources are rewritten as catalogs merge); defining a subset is the
    normal multi-runbook case.

    Example:
        >>> ep = {'engine': 'vllm', 'model': 'm'}
        >>> src = lambda **eps: {'models': {'m': {'source': 'hf://o/m'}}, 'endpoints': eps}
        >>> published = {'catalogs': [src(a=ep, b=ep)]}
        >>> profile_drift(published, {'catalogs': [src(a=ep)]})
        []
        >>> profile_drift(published, {'catalogs': [{'endpoints': {}}]})
        []
        >>> profile_drift(published, {'catalogs': [src(c=ep)]})
        ['catalogs']
    """
    drift = []
    for key in sorted(set(published) | set(invocation)):
        if key in {'version'}:
            continue
        if key == 'catalogs':
            if not _catalogs_within(invocation.get(key) or [], published.get(key) or []):
                drift.append(key)
            continue
        if published.get(key) != invocation.get(key):
            drift.append(key)
    return drift


def _catalogs_within(want: list[dict[str, Any]], have: list[dict[str, Any]]) -> bool:
    """Whether every endpoint ``want`` defines, ``have`` defines identically."""
    have_digests = {canonical_digest(s) for s in have}
    if all(canonical_digest(s) in have_digests for s in want):
        return True
    try:
        mine = CatalogUnion.from_sources(want) if want else None
        theirs = CatalogUnion.from_sources(have) if have else None
    except CatalogError:
        return False
    if mine is None:
        return True
    for name in mine.endpoints:
        if theirs is None or name not in theirs.endpoints:
            return False
        try:
            if (mine.resolve_endpoint(name).semantic_key()
                    != theirs.resolve_endpoint(name).semantic_key()):
                return False
        except CatalogError:
            return False
    return True


def merge_catalog_sources(
    published: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append compatible catalog snapshots, deduplicated by content.

    This is the live-epoch catalog rule: an acquire may add definitions without
    mutating any definition the recovery snapshot already knows.  Building the
    union is the semantic conflict check -- if an incoming endpoint/bundle/route
    resolves differently, :class:`CatalogConflict` is raised before anything is
    written.

    The snapshots are the published endpoint definitions (see
    :func:`adopt_catalog_sources` for how an acquire updates them); an
    external endpoint has no lease to pin it, so they are never dropped merely
    because the stack is quiescent.
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


def drop_catalog_names(
    sources: list[dict[str, Any]], names: set[str]
) -> list[dict[str, Any]]:
    """``sources`` without the endpoints and bundles named ``names``; a source
    left with neither goes.

    A bundle goes too when a member it names is dropped: a catalog's bundles
    name its own endpoints, and a source must stay a valid catalog. (The
    catalog that redefined the member publishes its own bundles.)

    Example:
        >>> src = {'endpoints': {'a': {}, 'b': {}}, 'bundles': {'ab': ['a', 'b']}}
        >>> drop_catalog_names([src], {'a'})
        [{'endpoints': {'b': {}}, 'bundles': {}}]
    """
    out: list[dict[str, Any]] = []
    for source in sources or []:
        endpoints = {n: s for n, s in (source.get('endpoints') or {}).items()
                     if n not in names}
        dropped = set(source.get('endpoints') or {}) - set(endpoints)
        bundles = {n: m for n, m in (source.get('bundles') or {}).items()
                   if n not in names and not set(m or []) & dropped}
        if not endpoints and not bundles:
            continue
        kept = dict(source)
        kept['endpoints'] = endpoints
        kept['bundles'] = bundles
        out.append(kept)
    return out


def adopt_catalog_sources(
    published: list[dict[str, Any]], incoming: list[dict[str, Any]],
    pinned: set[str],
) -> list[dict[str, Any]]:
    """The published endpoint definitions after an acquire or access.

    The published union is where endpoint definitions live between runs
    (docs/planning/external-endpoints.md, decision 1). ``incoming`` (the
    invocation's catalogs) is merged in: a definition it redefines replaces the
    published one, unless a resident workload runs it (``pinned``: then
    :class:`CatalogConflict`); every other published definition stays, however
    unrelated. A published snapshot also sheds definitions ``incoming`` repeats,
    so editing a catalog over time does not pile up copies.

    Example:
        >>> ep = lambda src: {'engine': 'vllm', 'model': 'm'}
        >>> cat = lambda **eps: {'models': {'m': {'source': 'hf://o/m'}}, 'endpoints': eps}
        >>> old = cat(a={'engine': 'vllm', 'model': 'm'},
        ...           x={'external': {'api_base': 'http://h/v1', 'model': 'q'}})
        >>> new = cat(a={'engine': 'vllm', 'model': 'm', 'runtime': {'max_model_len': 8}})
        >>> out = adopt_catalog_sources([old], [new], pinned=set())
        >>> sorted(CatalogUnion.from_sources(out).endpoints)      # x survives
        ['a', 'x']
        >>> adopt_catalog_sources([old], [new], pinned={'a'})
        Traceback (most recent call last):
        ...
        infer_stack.leasing.profile.CatalogConflict: endpoint 'a' is defined differently in two published catalogs
    """
    base = list(published or [])
    want = [dict(s or {}) for s in incoming or []]
    mine = CatalogUnion.from_sources(want) if want else None
    if mine is not None:
        # Definitions the invocation repeats identically live in its snapshot.
        same = set()
        for source in base:
            for name in (source.get('endpoints') or {}):
                if name in mine.endpoints:
                    try:
                        other = Catalog.from_dict(source).resolve_endpoint(name)
                        if other.semantic_key() == mine.resolve_endpoint(name).semantic_key():
                            same.add(name)
                    except CatalogError:
                        pass
        base = drop_catalog_names(base, same)
    for _ in range(64):             # each round drops at least one name
        try:
            return merge_catalog_sources(base, want)
        except CatalogConflict as ex:
            names: set[str] = {str(n) for n in ex.names}
            if not names or names & pinned:
                raise
            base = drop_catalog_names(base, names)
    raise CatalogConflict('catalog union did not settle', [])


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
