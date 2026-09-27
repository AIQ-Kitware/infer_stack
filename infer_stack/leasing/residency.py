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
* One deployment has more than one container on Docker → the deployment is
  *conflicted*: two authorities claim it. All containers are kept (never
  collapsed to one), it is not resident, and callers must fail closed. On
  Kubernetes several pods for one deployment are its replicas (or a rollout
  in progress), so the same shape is ordinary residency there: the snapshot
  says which rule applies (:attr:`Residency.replicated`).
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
    >>> res.is_resident('grp-a'), res.unique_unit('grp-a').gpus
    (True, (1, 2))
    >>> [c.container_id for c in res.occupants(2)]
    ['c1']
    >>> res.is_resident('grp-missing'), res.unique_unit('grp-missing')
    (False, None)
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
#: Which engine a service runs (``vllm``, ``ollama``, ``litellm``...).
ENGINE_LABEL = 'infer-stack.engine'
FINGERPRINT_LABEL = 'infer-stack.fingerprint'
#: Labels Docker Compose puts on every container of a project.
COMPOSE_PROJECT_LABEL = 'com.docker.compose.project'
COMPOSE_SERVICE_LABEL = 'com.docker.compose.service'

def _published_ports(ports) -> str:
    """``14042->4000/tcp`` for each published port of a ``docker inspect``.

    >>> _published_ports({'4000/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '14042'},
    ...                                {'HostIp': '::', 'HostPort': '14042'}],
    ...                   '8000/tcp': None})
    '14042->4000/tcp'
    """
    out = []
    for inner, bindings in sorted((ports or {}).items()):
        for host in sorted({b.get('HostPort') for b in bindings or [] if b.get('HostPort')}):
            out.append(f'{host}->{inner}')
    return ', '.join(out)


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
    #: Why the instance is not running, when its runtime says:
    #: ``CrashLoopBackOff``, ``ImagePullBackOff``, ``OOMKilled`` (Kubernetes).
    #: Empty when there is nothing to say, or the runtime does not say.
    reason: str = ''
    #: The runtime's own words for ``reason``, when it gives any: a
    #: scheduler's ``0/2 nodes are available: 2 Insufficient nvidia.com/gpu``.
    message: str = ''
    #: When the current run started (the runtime's timestamp), for display.
    started: str = ''
    #: Published ports, ``host->container/proto`` joined by ``, ``; display only.
    ports: str = ''

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
    """A strict snapshot of deployment units, keyed by deployment id.

    A unit is a Docker container or a Kubernetes pod. Every unit matching a
    deployment is kept; nothing is collapsed, so a duplicate is visible
    rather than silently lost.

    The questions are asked of a deployment, not of a unit. Whether several
    units for one deployment are normal depends on the runtime, and the
    snapshot carries that rule (:attr:`replicated`), so no caller branches on
    the backend:

    >>> pod = lambda name, state='running': Container(name, 'grp-a', state)
    >>> pods = Residency({'grp-a': (pod('p1'), pod('p2'))}, replicated=True)
    >>> pods.is_resident('grp-a'), pods.is_conflicted('grp-a'), pods.unique_unit('grp-a')
    (True, False, None)
    >>> docker = Residency({'grp-a': (pod('c1'), pod('c2'))})
    >>> docker.is_resident('grp-a'), docker.is_conflicted('grp-a')
    (False, True)
    >>> rollout = Residency({'grp-a': (pod('old', 'removing'), pod('new'))}, replicated=True)
    >>> rollout.is_resident('grp-a'), [c.container_id for c in rollout.warm_units('grp-a')]
    (True, ['new'])
    """

    by_deployment: dict[str, tuple[Container, ...]] = field(default_factory=dict)
    #: Project units that belong to no deployment (infrastructure, or
    #: containers without infer-stack labels at all).
    others: tuple[Container, ...] = ()
    #: Whether one deployment may run as several units. True on Kubernetes
    #: (replicas; old and new pods during a rollout); False on Docker, where a
    #: second container for one deployment is a conflicting authority.
    replicated: bool = False

    def units(self, deployment_id: str) -> tuple[Container, ...]:
        """Every unit carrying this deployment's label, in any state."""
        return self.by_deployment.get(deployment_id, ())

    def warm_units(self, deployment_id: str) -> tuple[Container, ...]:
        """The units that hold, or will reclaim on their own, a warm model."""
        return tuple(c for c in self.units(deployment_id) if c.warm)

    def is_conflicted(self, deployment_id: str) -> bool:
        """Several units claim a deployment that may have only one; fail closed."""
        return not self.replicated and len(self.units(deployment_id)) > 1

    def is_resident(self, deployment_id: str) -> bool:
        """The deployment is running: at least one warm unit, and no conflict."""
        return not self.is_conflicted(deployment_id) and bool(self.warm_units(deployment_id))

    def unique_unit(self, deployment_id: str) -> Container | None:
        """The deployment's one warm unit, or ``None``.

        For decisions that recover one physical fact from one unit: which
        GPUs a deployment's reservation holds. ``None`` for no unit, a unit
        that is not warm, a conflict, and several replicas alike: none of
        those names one reservation, so GPU adoption fails closed.
        """
        found = self.units(deployment_id)
        if len(found) != 1 or not found[0].warm:
            return None
        return found[0]

    def resident_gpus(self, deployment_id: str) -> list[int] | None:
        """The physical GPUs a resident deployment holds, or ``None`` if unknown.

        ``[]`` for a resident deployment holding none (a CPU engine, or a pod:
        the cluster owns placement). ``None`` when it is not resident, or a
        unit's reservation could not be mapped to indices.
        """
        if not self.is_resident(deployment_id):
            return None
        warm = self.warm_units(deployment_id)
        if any(c.all_gpus for c in warm):
            return None
        return sorted({g for c in warm for g in c.gpus})

    def occupants(self, gpu: int) -> tuple[Container, ...]:
        """Every container, in any state, whose reservation includes ``gpu``."""
        return tuple(c for c in self.all_containers() if c.occupies(gpu))

    def all_containers(self) -> tuple[Container, ...]:
        return (
            *(c for group in self.by_deployment.values() for c in group),
            *self.others,
        )


def deployment_health(residency: Residency, deployment_id: str) -> str | None:
    """Whether a deployment serves, as opposed to whether it is resident.

    Residency is physical: a crash-looping unit is warm (it holds, and will
    reclaim, its resources). Serving is not: a deployment is ``up`` only when
    some unit runs and has not failed or not yet passed its health check. A
    Docker container with no healthcheck reports an empty health, which is
    up when running; a Kubernetes pod reports ``healthy`` when Ready.

    * ``up``: some unit running, not ``starting`` or ``unhealthy``;
    * ``starting``: none up, some running but not ready yet, or created;
    * ``restarting``: every warm unit crash-looping (restarting);
    * ``conflicted``: units that may not coexist (see :meth:`Residency.is_conflicted`);
    * otherwise the units' states (``exited``, ``removing`` ...), or ``None``
      when the deployment has no unit.

    >>> def unit(name, state='running', health=''):
    ...     return Container(name, 'grp', state, health=health)
    >>> def health(*units):
    ...     return deployment_health(Residency({'grp': units}, replicated=True), 'grp')
    >>> health(unit('a', 'restarting'), unit('b', 'restarting'))
    'restarting'
    >>> health(unit('a', health='healthy'), unit('b', 'restarting'))
    'up'
    >>> health(unit('a', health='starting'), unit('b', 'restarting'))
    'starting'
    >>> health(unit('c'))                    # Docker, no healthcheck, running
    'up'
    """
    units = residency.units(deployment_id)
    if not units:
        return None
    if residency.is_conflicted(deployment_id):
        return 'conflicted'
    if any(u.state == 'running' and u.health not in ('starting', 'unhealthy')
           for u in units):
        return 'up'
    if any(u.state == 'created' or (u.state == 'running' and u.health == 'starting')
           for u in units):
        return 'starting'
    warm = residency.warm_units(deployment_id)
    if warm and all(u.state == 'restarting' for u in warm):
        return 'restarting'
    return '/'.join(sorted({u.state for u in units}))


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
            started=str((item.get('State') or {}).get('StartedAt') or ''),
            ports=_published_ports((item.get('NetworkSettings') or {}).get('Ports')),
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


#: Label a managed Kubernetes pod carries (copied by KubeAI from its Model).
POD_DEPLOYMENT_LABEL = 'infer-stack/deployment'
POD_MANAGED_LABEL = 'infer-stack/managed'


def residency_from_pods(raw: str) -> Residency:
    """Build a :class:`Residency` from ``kubectl get pods -o json`` output.

    One pod is one instance (a :class:`Container`), keyed by the
    ``infer-stack/deployment`` label KubeAI copies from the Model. The state
    is mapped to the Docker vocabulary the rest of infer-stack reads:

    * running container -> ``running``
    * ``CrashLoopBackOff`` -> ``restarting`` (still warm: the kubelet retries)
    * any other waiting reason, or a pod not started yet -> ``created``
    * terminated -> ``exited``; a pod being deleted -> ``removing``

    The waiting or last-termination reason is kept in :attr:`Container.reason`.
    GPUs are left empty: the cluster, not this host, owns placement. Raises
    :class:`ResidencyUnknown` on output that is not a pod list.

    Example:
        >>> raw = json.dumps({'items': [{
        ...     'metadata': {'name': 'model-q-1', 'labels': {
        ...         'infer-stack/deployment': 'grp-a', 'infer-stack/managed': 'true',
        ...         'model': 'q'}},
        ...     'status': {'phase': 'Running', 'containerStatuses': [{
        ...         'state': {'waiting': {'reason': 'CrashLoopBackOff'}},
        ...         'lastState': {'terminated': {'exitCode': 1, 'reason': 'Error'}},
        ...         'restartCount': 3}]}}]})
        >>> c = residency_from_pods(raw).units('grp-a')[0]
        >>> (c.state, c.reason, c.restart_count, c.exit_code, c.warm)
        ('restarting', 'CrashLoopBackOff', 3, 1, True)
    """
    try:
        data = json.loads(raw or '{}')
    except json.JSONDecodeError as ex:
        raise ResidencyUnknown(f'kubectl get pods output is not JSON: {ex}') from ex
    items = data.get('items') if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ResidencyUnknown('kubectl get pods output has no items list')
    grouped: dict[str, list[Container]] = {}
    others: list[Container] = []
    for pod in items:
        meta = pod.get('metadata') or {}
        labels = meta.get('labels') or {}
        status = pod.get('status') or {}
        statuses = status.get('containerStatuses') or []
        first = statuses[0] if statuses else {}
        current = first.get('state') or {}
        last = (first.get('lastState') or {}).get('terminated') or {}
        waiting = (current.get('waiting') or {}).get('reason') or ''
        if meta.get('deletionTimestamp'):
            state = 'removing'
        elif 'running' in current:
            state = 'running'
        elif waiting == 'CrashLoopBackOff':
            state = 'restarting'
        elif 'terminated' in current:
            state = 'exited'
        else:
            state = 'created'
        ended = current.get('terminated') or last
        message = ''
        if not waiting and not statuses:
            # Not started at all: the pod's own condition says why, e.g. the
            # scheduler found no node with the resources ("Unschedulable"),
            # and its message says which constraint each node failed.
            failed = next((c for c in status.get('conditions') or []
                           if c.get('status') == 'False' and c.get('reason')), {})
            waiting = str(failed.get('reason') or '')
            message = str(failed.get('message') or '')
        ready = any(c.get('type') == 'Ready' and c.get('status') == 'True'
                    for c in status.get('conditions') or [])
        container = Container(
            container_id=str(meta.get('name') or ''),
            deployment_id=str(labels.get(POD_DEPLOYMENT_LABEL) or ''),
            state=state,
            service=str(labels.get('model') or ''),
            labelled=labels.get(POD_MANAGED_LABEL) == 'true',
            health='healthy' if ready else ('starting' if state == 'running' else ''),
            restart_count=int(first.get('restartCount') or 0),
            exit_code=(None if ended.get('exitCode') is None else int(ended['exitCode'])),
            # A Deployment's pods are always restarted by the kubelet.
            restart_policy='always',
            reason=waiting or str(ended.get('reason') or ''),
            message=message,
            started=str((current.get('running') or {}).get('startedAt')
                        or status.get('startTime') or ''),
        )
        if container.deployment_id:
            grouped.setdefault(container.deployment_id, []).append(container)
        else:
            others.append(container)
    return Residency(
        by_deployment={gid: tuple(found) for gid, found in grouped.items()},
        others=tuple(others),
        replicated=True,
    )
