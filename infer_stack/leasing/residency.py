"""Strict physical residency: which deployment containers exist, on which GPUs.

The lenient :meth:`ComposeBackend.observe` answers "roughly what is running"
and deliberately returns an empty set when Docker cannot be read, because
acquire must not brick on a stale compose file. That contract is right for
reporting and wrong for any decision that stops, removes, or hands over a GPU:
there, "nothing is running" and "could not look" must never be confused.

This module is the strict counterpart. A :class:`Residency` snapshot is built
from Docker's own view of the containers, keyed by the ``infer-stack.deployment``
label and the Compose project label, with GPUs taken from each container's
actual device reservation. Nothing here consults the rendered compose file or
its sidecar, which describe what was *rendered*, not what exists.

Failure modes are explicit:

* Docker cannot be read, or returns something unparseable → :class:`ResidencyUnknown`
  is raised. A snapshot is never silently empty.
* One deployment has more than one container → the deployment is *ambiguous*.
  All containers are kept (never collapsed to one), :meth:`Residency.resident`
  returns ``None`` for it, and callers must fail closed.
* A container's GPUs cannot be mapped to physical indices (a count-based
  reservation, or device UUIDs rather than indices) → it is treated as occupying
  *every* GPU, so no GPU is ever handed over on a guess.

Example:
    >>> import json
    >>> raw = json.dumps([
    ...     {'Id': 'c1', 'State': {'Status': 'running'},
    ...      'Config': {'Labels': {'com.docker.compose.project': 'infer-stack',
    ...                            'infer-stack.deployment': 'grp-a'}},
    ...      'HostConfig': {'DeviceRequests': [{'Count': 0, 'DeviceIDs': ['1', '2']}]}},
    ... ])
    >>> res = residency_from_inspect(raw, project='infer-stack')
    >>> res.resident('grp-a').gpus
    (1, 2)
    >>> [c.container_id for c in res.occupants(2)]
    ['c1']
    >>> res.resident('grp-missing') is None
    True
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Label every infer-stack model service carries (rendered in ``compose.py``).
DEPLOYMENT_LABEL = 'infer-stack.deployment'
#: Labels every rendered service carries: its service name, and a behavioural
#: fingerprint that changes only when the service's behaviour does.
SERVICE_LABEL = 'infer-stack.service'
FINGERPRINT_LABEL = 'infer-stack.fingerprint'
#: Labels Docker Compose puts on every container of a project.
COMPOSE_PROJECT_LABEL = 'com.docker.compose.project'
COMPOSE_SERVICE_LABEL = 'com.docker.compose.service'

#: Container states that hold, or will reclaim on their own, a warm model: the
#: process is up, is being restarted by Docker's restart policy, or is paused
#: (a paused process keeps its GPU memory). ``created``, ``exited``,
#: ``removing`` and ``dead`` are not warm residency; such containers are still
#: reported (see :meth:`Residency.occupants`) so nothing starts on top of them.
WARM_STATES = frozenset({'running', 'restarting', 'paused'})


class ResidencyUnknown(RuntimeError):
    """Docker's view of the deployment containers could not be established.

    Callers must treat this as "unknown", never as "nothing is resident".
    """


@dataclass(frozen=True)
class Container:
    """One deployment container as Docker reports it."""

    container_id: str
    #: Empty for infrastructure (gateway, database, UI, proxy).
    deployment_id: str
    state: str
    #: Physical GPU indices from the container's device reservation. Empty for a
    #: container with no GPU reservation.
    gpus: tuple[int, ...] = ()
    #: True when the reservation could not be mapped to indices; the container
    #: is then conservatively treated as occupying every GPU.
    all_gpus: bool = False
    #: Compose service name (``infer-stack.service``, else Compose's own label).
    service: str = ''
    #: ``infer-stack.fingerprint``; empty on a container rendered before labels.
    fingerprint: str = ''
    #: Carries both infer-stack ownership labels (service and fingerprint).
    labelled: bool = False
    #: IPv4 addresses on the container's networks.
    ips: tuple[str, ...] = ()
    #: Docker healthcheck status (``healthy``, ``starting``, ``unhealthy``), or
    #: empty when the container defines no healthcheck.
    health: str = ''
    #: How many times Docker's restart policy has restarted this container, and
    #: the exit code of its last run. A container that keeps exiting non-zero is
    #: crash-looping, which readiness must not mistake for a slow model load.
    restart_count: int = 0
    exit_code: int | None = None
    #: Docker's restart policy for this container, and its retry cap
    #: (``on-failure`` only). Together with the state these say whether anything
    #: will try to start the container again.
    restart_policy: str = ''
    restart_max: int = 0

    @property
    def warm(self) -> bool:
        return self.state in WARM_STATES

    def occupies(self, gpu: int) -> bool:
        return self.all_gpus or gpu in self.gpus

    @property
    def will_be_restarted(self) -> bool:
        """Whether Docker will start this container again on its own.

        ``always``/``unless-stopped`` always will; ``on-failure`` until its cap;
        ``no`` (or no policy) never will. A container nothing will retry is as
        good as dead, whatever its log says about the cause.
        """
        if self.state in WARM_STATES:
            return True
        if self.restart_policy in {'always', 'unless-stopped'}:
            return True
        if self.restart_policy == 'on-failure':
            return self.restart_max == 0 or self.restart_count < self.restart_max
        return False


@dataclass(frozen=True)
class Residency:
    """A strict snapshot of deployment containers, keyed by deployment id.

    Every container matching a deployment is kept. Nothing is collapsed, so a
    duplicate is visible rather than silently lost.
    """

    by_deployment: dict[str, tuple[Container, ...]] = field(default_factory=dict)
    #: Project containers that belong to no deployment (infrastructure, or
    #: containers without infer-stack labels at all).
    others: tuple[Container, ...] = ()

    def containers(self, deployment_id: str) -> tuple[Container, ...]:
        """Every container carrying this deployment's label, in any state."""
        return self.by_deployment.get(deployment_id, ())

    def ambiguous(self, deployment_id: str) -> bool:
        """More than one container claims this deployment; fail closed."""
        return len(self.containers(deployment_id)) > 1

    def resident(self, deployment_id: str) -> Container | None:
        """The deployment's single warm container, or ``None``.

        ``None`` covers "no container", "one container that is not warm", and
        "ambiguous". Use :meth:`ambiguous` to tell the last apart.
        """
        found = self.containers(deployment_id)
        if len(found) != 1 or not found[0].warm:
            return None
        return found[0]

    def occupants(self, gpu: int) -> tuple[Container, ...]:
        """Every container, in any state, whose reservation includes ``gpu``."""
        return tuple(c for c in self.all_containers() if c.occupies(gpu))

    def all_containers(self) -> tuple[Container, ...]:
        return (
            *(c for group in self.by_deployment.values() for c in group),
            *self.others,
        )


def _gpus_from_device_requests(requests: Any) -> tuple[tuple[int, ...], bool]:
    """Map Docker ``HostConfig.DeviceRequests`` to ``(indices, all_gpus)``.

    Compose's ``deploy.resources.reservations.devices[].device_ids`` surfaces as
    ``DeviceRequests[].DeviceIDs`` (index strings, ``Count`` 0). Anything that
    cannot be mapped to indices is reported as ``all_gpus`` rather than guessed.
    """
    if not requests:
        return (), False
    if not isinstance(requests, list):
        return (), True
    indices: set[int] = set()
    for req in requests:
        if not isinstance(req, dict):
            return (), True
        ids = req.get('DeviceIDs') or []
        count = req.get('Count') or 0
        if not ids:
            if count:
                # e.g. `--gpus all` (Count -1) or `--gpus 2`: no fixed indices.
                return (), True
            continue
        for raw in ids:
            try:
                indices.add(int(str(raw)))
            except ValueError:
                # A GPU UUID or MIG id: not a physical index we can compare.
                return (), True
    return tuple(sorted(indices)), False


def residency_from_inspect(raw: str, *, project: str) -> Residency:
    """Build a :class:`Residency` from ``docker inspect`` JSON output.

    Containers outside ``project`` are ignored, even if a caller's listing let
    them through. Containers of the project without a deployment label are kept
    in :attr:`Residency.others`. Raises :class:`ResidencyUnknown` on output
    that is not a JSON array of container objects.
    """
    try:
        data = json.loads(raw or '[]')
    except json.JSONDecodeError as ex:
        raise ResidencyUnknown(f'docker inspect output is not JSON: {ex}') from ex
    if not isinstance(data, list):
        raise ResidencyUnknown('docker inspect output is not a JSON array')
    grouped: dict[str, list[Container]] = {}
    others: list[Container] = []
    for item in data:
        if not isinstance(item, dict):
            raise ResidencyUnknown('docker inspect returned a non-object entry')
        labels = ((item.get('Config') or {}).get('Labels')) or {}
        if labels.get(COMPOSE_PROJECT_LABEL) != project:
            continue
        deployment_id = labels.get(DEPLOYMENT_LABEL) or ''
        container_id = item.get('Id')
        state = (item.get('State') or {}).get('Status')
        if not container_id or not state:
            raise ResidencyUnknown(
                f'docker inspect entry for {deployment_id or "a project container"!r} '
                'lacks an Id or State.Status'
            )
        gpus, all_gpus = _gpus_from_device_requests(
            (item.get('HostConfig') or {}).get('DeviceRequests')
        )
        container = Container(
            container_id=str(container_id),
            deployment_id=str(deployment_id),
            state=str(state).lower(),
            gpus=gpus,
            all_gpus=all_gpus,
            service=str(labels.get(SERVICE_LABEL) or labels.get(COMPOSE_SERVICE_LABEL) or ''),
            fingerprint=str(labels.get(FINGERPRINT_LABEL) or ''),
            labelled=bool(labels.get(SERVICE_LABEL) and labels.get(FINGERPRINT_LABEL)),
            health=str(((item.get('State') or {}).get('Health') or {}).get('Status') or ''),
            restart_count=int(item.get('RestartCount') or 0),
            restart_policy=str((((item.get('HostConfig') or {}).get('RestartPolicy')
                                 or {}).get('Name')) or ''),
            restart_max=int((((item.get('HostConfig') or {}).get('RestartPolicy')
                              or {}).get('MaximumRetryCount')) or 0),
            exit_code=(None if (item.get('State') or {}).get('ExitCode') is None
                       else int((item.get('State') or {})['ExitCode'])),
            ips=tuple(sorted(
                str(n.get('IPAddress')) for n in
                (((item.get('NetworkSettings') or {}).get('Networks')) or {}).values()
                if isinstance(n, dict) and n.get('IPAddress')
            )),
        )
        if deployment_id:
            grouped.setdefault(deployment_id, []).append(container)
        else:
            others.append(container)
    return Residency(
        {
            gid: tuple(sorted(found, key=lambda c: c.container_id))
            for gid, found in grouped.items()
        },
        tuple(sorted(others, key=lambda c: c.container_id)),
    )
