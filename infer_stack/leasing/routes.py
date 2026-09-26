"""Planning changes to the gateway's route registry (``routes seed`` / ``prune``).

The registry maps a public alias to the upstream it routes to. Adding an
alias is additive; changing where an existing alias points redirects every
client that uses it, so a seed that would redefine one refuses by default
and replaces it only when asked. A plan says which is which before anything
is written; :class:`~infer_stack.leasing.controller.Controller` commits it
through the normal publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RouteConflict(ValueError):
    """A seed would redefine existing aliases; ``names`` are the conflicts."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        super().__init__(
            f'routes seed would redefine {len(self.names)} existing route(s): '
            f'{", ".join(self.names)}; pass --replace to redefine them')


@dataclass
class RoutePlan:
    """What a seed or prune would do to the registry's entries.

    ``added``: new aliases; ``unchanged``: identical rows already there;
    ``conflicted``: aliases whose incoming row differs (``{name: (current,
    incoming)}``); ``dropped``: aliases a prune removes. ``incoming`` keeps a
    seed's rows so the commit can recheck them under the lock.
    """

    added: dict[str, dict[str, Any]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    conflicted: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    incoming: dict[str, dict[str, Any]] = field(default_factory=dict)


def plan_seed(current: dict[str, dict[str, Any]],
              incoming: dict[str, dict[str, Any]]) -> RoutePlan:
    """Classify ``incoming`` rows against the registry's ``current`` entries.

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


def plan_prune(current: dict[str, dict[str, Any]],
               keep: dict[str, dict[str, Any]]) -> RoutePlan:
    """Which of ``current`` a prune drops: everything not in ``keep``.

    >>> plan_prune({'a': {}, 'b': {}}, {'a': {}}).dropped
    ['b']
    """
    return RoutePlan(dropped=sorted(set(current) - set(keep)))
