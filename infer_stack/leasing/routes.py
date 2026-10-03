"""Gateway routes, and planning ``routes seed`` / ``prune``.

A :class:`GatewayRoute` is one alias the gateway serves and where it sends
it; :meth:`GatewayRoute.entry` is the one rendering of it as a LiteLLM
``model_list`` entry or admin-API route. Routes are derived at render
(docs/planning/external-endpoints.md, decision 2), one layer per owner, in
increasing precedence:

* ``registry`` -- rows ``litellm_registry.json`` keeps: routes of
  deployments no published catalog defines, remembered past their release
  so the gateway is not recreated for one, and rows from before routes were
  derived; ``routes prune`` drops those nothing serves;
* ``catalog`` / ``external`` -- every endpoint of the published catalog union;
* ``deployment`` -- every placed deployment;
* ``upstream`` -- routes another backend supplies (KubeAI's Models).

``routes seed`` merges catalogs into the published union and ``routes
prune`` unpublishes. Adding an alias is additive; changing what an existing
one means redirects every client that uses it, so a seed that would redefine
one refuses by default and replaces it only when asked. A plan says which is
which before anything is written; :class:`~infer_stack.leasing.controller.
Controller` commits it through the normal publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Where a route came from, lowest precedence first (see the module doc).
ORIGINS = ('registry', 'catalog', 'external', 'deployment', 'upstream')


@dataclass(frozen=True)
class GatewayRoute:
    """One alias the gateway serves, and the upstream it sends it to.

    ``kind`` is ``openai`` (any OpenAI-compatible server: vLLM, a KubeAI
    Model, an external target) or ``ollama``; ``model`` is the upstream
    model, what that server expects as ``model``. ``key_env`` names the
    variable holding the upstream's key (never the key); without one an
    ``openai`` route sends ``EMPTY``. ``route_id`` is a dynamic route's
    managed id. ``origin`` says which owner produced it and is not part of
    what it means. ``max_input_tokens`` is the public endpoint's context
    contract: normally its effective vLLM ``max_model_len``. A
    shared-compatible endpoint may be backed by a larger deployment, but its
    advertised contract stays the endpoint's own configured window. It is
    emitted in ``model_info`` so clients read the managed window from
    ``/model/info`` instead of guessing (absent when unknown, e.g. for
    servers infer-stack does not run).

    Example:
        >>> GatewayRoute('qwen', 'openai', 'Qwen/Q', 'http://box/v1', key_env='BOX_KEY').entry()
        {'model_name': 'qwen', 'litellm_params': {'model': 'openai/Qwen/Q', 'api_base': 'http://box/v1', 'api_key': 'os.environ/BOX_KEY'}}
        >>> GatewayRoute('tiny', 'ollama', 'tinyllama', 'http://ollama:11434', route_id='isr-1').entry()
        {'model_name': 'tiny', 'litellm_params': {'model': 'ollama/tinyllama', 'api_base': 'http://ollama:11434'}, 'model_info': {'id': 'isr-1'}}
        >>> GatewayRoute('big', 'openai', 'Big/B', 'http://b/v1', route_id='isr-9',
        ...              max_input_tokens=262144).entry()
        {'model_name': 'big', 'litellm_params': {'model': 'openai/Big/B', 'api_base': 'http://b/v1', 'api_key': 'EMPTY'}, 'model_info': {'id': 'isr-9', 'max_input_tokens': 262144}}
    """

    alias: str
    kind: str
    model: str
    api_base: str
    key_env: str | None = None
    route_id: str | None = None
    origin: str = field(default='catalog', compare=False)
    max_input_tokens: int | None = None

    def entry(self) -> dict[str, Any]:
        """The LiteLLM entry: a static ``model_list`` item, or with a
        ``route_id`` the body of an admin-API ``/model/new``.

        ``model_info`` carries the managed id and the advertised context
        window. The window is ``max_input_tokens`` (the total sequence
        budget the upstream was started with), never a ``litellm_params``
        ``max_tokens`` (that would change request defaults) and never
        ``max_output_tokens`` (a completion cap is prompt-dependent).
        """
        params: dict[str, Any] = {'model': f'{self.kind}/{self.model}',
                                  'api_base': self.api_base}
        if self.kind == 'openai':
            params['api_key'] = f'os.environ/{self.key_env}' if self.key_env else 'EMPTY'
        entry: dict[str, Any] = {'model_name': self.alias, 'litellm_params': params}
        model_info: dict[str, Any] = {}
        if self.route_id:
            model_info['id'] = self.route_id
            if self.key_env:
                # The key's name is part of what the route means, and LiteLLM
                # redacts the key itself, so reconcile compares the name here.
                model_info['infer_stack_key_env'] = self.key_env
        if self.max_input_tokens is not None:
            model_info['max_input_tokens'] = self.max_input_tokens
        if model_info:
            entry['model_info'] = model_info
        return entry

    def describe(self) -> str:
        """``kind/model -> api_base`` (and the key's name), for people."""
        key = f' (key ${self.key_env})' if self.key_env else ''
        return f'{self.kind}/{self.model} -> {self.api_base}{key}'


def route_table(*layers: Iterable[GatewayRoute]) -> list[GatewayRoute]:
    """One route per alias, a later layer winning, sorted by alias.

    >>> a = GatewayRoute('a', 'openai', 'm', 'http://old/v1', origin='registry')
    >>> b = GatewayRoute('a', 'openai', 'm', 'http://new/v1')
    >>> [r.api_base for r in route_table([a], [b])]
    ['http://new/v1']
    """
    table: dict[str, GatewayRoute] = {}
    for layer in layers:
        for route in layer:
            table[route.alias] = route
    return [table[alias] for alias in sorted(table)]


class RouteConflict(ValueError):
    """A seed would redefine existing aliases; ``names`` are the conflicts.

    ``changed``: a ``--replace`` whose confirmed redefinitions no longer hold,
    because another process changed those aliases after the plan was shown.
    """

    def __init__(self, names: list[str], *, changed: bool = False):
        self.names = list(names)
        self.changed = changed
        if changed:
            message = (f'{", ".join(self.names)} changed since the redefinition was '
                       'shown; nothing was written, run `routes seed` again to see '
                       'what is there now')
        else:
            message = (f'routes seed would redefine {len(self.names)} existing '
                       f'route(s): {", ".join(self.names)}; pass --replace to '
                       'redefine them')
        super().__init__(message)


@dataclass(frozen=True, eq=False)
class Meaning:
    """What an alias means, for a seed: the endpoint definition's semantic
    key (``None`` for a registry row, which has only a route) and the route it
    renders (``None`` where this gateway cannot route it).

    Two meanings are the same when both keys match or, for a registry row,
    when the routes do.
    """

    route: GatewayRoute | None
    key: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Meaning):
            return NotImplemented
        if self.key is not None and other.key is not None:
            return self.key == other.key
        return self.route == other.route

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        text = self.route.describe() if self.route is not None else '(not routed here)'
        return text if self.key is not None else f'{text} [registry]'


@dataclass
class RoutePlan:
    """What a seed or prune would do.

    ``added``: aliases the seed publishes; ``unchanged``: aliases already
    published with the same meaning; ``conflicted``: aliases whose incoming
    meaning differs (``{name: (current, incoming)}``); ``dropped``: aliases a
    prune unpublishes. ``incoming`` keeps a seed's meanings and ``sources``
    its catalogs, so the commit can recheck them under the lock.
    """

    added: dict[str, Any] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    conflicted: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    incoming: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)


def plan_seed(current: dict[str, Any], incoming: dict[str, Any]) -> RoutePlan:
    """Classify ``incoming`` meanings against ``current`` ones.

    >>> plan = plan_seed({'a': {'up': 1}, 'b': {'up': 2}},
    ...                  {'a': {'up': 1}, 'b': {'up': 9}, 'c': {'up': 3}})
    >>> sorted(plan.added), plan.unchanged, sorted(plan.conflicted)
    (['c'], ['a'], ['b'])
    """
    plan = RoutePlan(incoming=dict(incoming))
    for name in sorted(incoming):
        row = incoming[name]
        if name not in current:
            plan.added[name] = row
        elif current[name] == row:
            plan.unchanged.append(name)
        else:
            plan.conflicted[name] = (current[name], row)
    return plan


def plan_prune(current: Iterable[str], keep: Iterable[str]) -> RoutePlan:
    """Which of the ``current`` aliases a prune drops: all not in ``keep``.

    >>> plan_prune({'a': {}, 'b': {}}, {'a': {}}).dropped
    ['b']
    """
    return RoutePlan(dropped=sorted(set(current) - set(keep)))
