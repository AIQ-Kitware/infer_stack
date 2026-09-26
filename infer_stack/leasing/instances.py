"""What a backend is running, one row per unit, and how to read its log.

An :class:`Instance` is a container (Compose, or the KubeAI gateway) or a pod
(KubeAI). Backends build them from their strict residency
(:meth:`instances`), so ``ps``, ``logs``, ``status`` and the TUI read the
same view the controller decides with. The runtime an instance lives in is
recorded on it, so reading its log needs no knowledge of which backend
produced it: :func:`history_argv` and :func:`follow_argv` are the only place
that knows ``docker logs`` from ``kubectl logs``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping

#: Runtimes an instance can live in.
DOCKER = 'docker'
KUBERNETES = 'kubernetes'
#: In-process backends (dry-run, tests): nothing to read a log from.
MEMORY = 'memory'


@dataclass(frozen=True)
class Instance:
    """One running unit: an engine container or pod, or a front-door service."""

    #: What a user types: the Compose service, or the pod name.
    name: str
    #: The runtime's own id: container id, or the pod name.
    id: str
    #: The deployment it serves; empty for the gateway, UI, database, proxy.
    deployment_id: str
    state: str
    restarts: int = 0
    #: Why it is not running, when the runtime says (``CrashLoopBackOff``...).
    reason: str = ''
    #: ``healthy`` / ``starting`` / ``unhealthy``, when the runtime has a check.
    health: str = ''
    gpus: tuple[int, ...] = ()
    started: str = ''
    ports: str = ''
    runtime: str = DOCKER
    #: Kubernetes only: where the pod lives and which container is the engine.
    namespace: str = ''
    container: str = ''

    @property
    def is_engine(self) -> bool:
        return bool(self.deployment_id)

    @property
    def status(self) -> str:
        """``running``, ``restarting (CrashLoopBackOff, 3 restarts)``, ..."""
        bits = [b for b in (self.reason, self.health if self.health != 'healthy' else '')
                if b]
        if self.restarts:
            bits.append(f'{self.restarts} restart{"s" if self.restarts != 1 else ""}')
        return f'{self.state} ({", ".join(bits)})' if bits else self.state


def from_residency(residency, *, runtime: str = DOCKER, namespace: str = '',
                   container: str = '') -> list[Instance]:
    """Every instance in a :class:`~infer_stack.leasing.residency.Residency`."""
    out = []
    for c in residency.all_containers():
        name = c.service if runtime == DOCKER and c.service else c.container_id
        out.append(Instance(
            name=name, id=c.container_id, deployment_id=c.deployment_id,
            state=c.state, restarts=c.restart_count, reason=c.reason,
            health=c.health, gpus=tuple(c.gpus), started=c.started, ports=c.ports,
            runtime=runtime, namespace=namespace, container=container,
        ))
    return sorted(out, key=lambda i: (not i.is_engine, i.name, i.id))


def history_argv(instance: Instance, *, tail: str | int | None = None,
                 timestamps: bool = False) -> list[str] | None:
    """The command that prints ``instance``'s log so far, or ``None``."""
    flags = [] if tail is None else ['--tail', str(tail)]
    if timestamps:
        flags.append('--timestamps')
    if instance.runtime == DOCKER:
        return ['docker', 'logs', *flags, instance.id]
    if instance.runtime == KUBERNETES:
        return ['kubectl', '-n', instance.namespace, 'logs', *flags, instance.id,
                *(['-c', instance.container] if instance.container else [])]
    return None


def follow_argv(instance: Instance, *, timestamps: bool = False) -> list[str] | None:
    r"""The command that streams ``instance``'s output from now on, or ``None``.

    Docker: ``docker attach`` with stdin closed and signals not forwarded, not
    ``docker logs -f``: the log driver holds a partial line until its newline,
    so a progress bar redrawn with ``\r`` shows nothing for minutes (measured).
    Ending the attach never touches the container. Kubernetes has no such
    driver in the way: ``kubectl logs -f --tail 0``.
    """
    if instance.runtime == DOCKER:
        return ['docker', 'attach', '--no-stdin', '--sig-proxy=false', instance.id]
    if instance.runtime == KUBERNETES:
        argv = history_argv(instance, tail=0, timestamps=timestamps)
        assert argv is not None
        return [*argv, '--follow']
    return None


def runtime_env(instance: Instance) -> dict[str, str] | None:
    """The environment to run the instance's log commands in (None: inherit)."""
    if instance.runtime == DOCKER:
        from .compose import docker_environment

        return docker_environment()
    return None


class UnknownTarget(ValueError):
    """A name that matches no instance; the message lists what exists."""


def resolve(instances: list[Instance], names: Iterable[str],
            served: Mapping[str, Iterable[str]] | None = None) -> list[Instance]:
    """The instances ``names`` refer to, in order, without repeats.

    A name matches an instance's name, its id (a prefix of at least four
    characters, as Docker accepts), its deployment id, or an endpoint alias
    the deployment serves (``served``: deployment id -> aliases). One alias
    or deployment can match several instances (a pod and its restart).

    >>> a = Instance('vllm-qwen', 'c0ffee12', 'grp-1', 'running')
    >>> b = Instance('litellm', 'deadbeef', '', 'running')
    >>> [i.name for i in resolve([a, b], ['qwen', 'litellm'], {'grp-1': ['qwen']})]
    ['vllm-qwen', 'litellm']
    >>> [i.name for i in resolve([a, b], ['c0ff'])]
    ['vllm-qwen']
    >>> resolve([a, b], ['nope'])
    Traceback (most recent call last):
    ...
    infer_stack.leasing.instances.UnknownTarget: no instance matches 'nope'; running: litellm, vllm-qwen
    """
    served = {gid: set(aliases) for gid, aliases in (served or {}).items()}
    picked: list[Instance] = []
    for name in names:
        found = [
            i for i in instances
            if name in (i.name, i.id, i.deployment_id)
            or (len(name) >= 4 and i.id.startswith(name))
            or (i.deployment_id and name in served.get(i.deployment_id, ()))
        ]
        if not found:
            known = ', '.join(sorted({i.name for i in instances}))
            raise UnknownTarget(f'no instance matches {name!r}; '
                                + (f'running: {known}' if known else 'nothing is running'))
        picked.extend(i for i in found if i not in picked)
    return picked


class LogFollower:
    r"""Follow instances' logs: recent history, then live output.

    ``list_instances`` is called every ``poll`` seconds, so an instance that
    appears later (a model starting, a recreate) is followed from its first
    line, and one that restarted is followed again. ``stdout`` yields
    ``name  | line``; with ``prefix='auto'`` the name is left off while only
    one instance has been followed (it says nothing then, and costs a narrow
    pane most of its width). Output written between the history read and the
    live stream (a few milliseconds) can be missed.
    """

    poll = 3.0

    def __init__(self, list_instances: Callable[[], list[Instance]], *,
                 history: str | int = 200, timestamps: bool = False,
                 prefix: str = 'always'):
        import queue
        import threading

        self._list = list_instances
        self._history = history
        self._timestamps = timestamps
        self._prefix = prefix
        self._names: set[str] = set()
        self._lines: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._seen: set[str] = set()
        self._live: dict[str, subprocess.Popen] = {}
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self) -> None:
        import threading

        first = True
        while not self._stop.is_set():
            try:
                found = self._list()
            except Exception:  # noqa: BLE001 - runtime unreachable: try again
                found = []
            for inst in found:
                running = inst.state == 'running'
                live = self._live.get(inst.id)
                attached = live is not None and live.poll() is None
                if inst.id not in self._seen:
                    # Existing instances: the recent tail. A new one: all of it.
                    history: str | int | None = self._history if first else 'all'
                elif running and not attached:
                    history = None          # restarted: follow again, no repeat
                else:
                    continue
                self._seen.add(inst.id)
                self._names.add(inst.name)
                threading.Thread(target=self._follow, args=(inst, history, running),
                                 daemon=True).start()
            first = False
            self._stop.wait(self.poll)

    def _follow(self, inst: Instance, history, running: bool) -> None:
        from ..log_filter import LogLineSplitter

        prefix = inst.name
        env = runtime_env(inst)
        if history is not None:
            argv = history_argv(inst, tail=history, timestamps=self._timestamps)
            old = ''
            if argv is not None:
                try:
                    old = subprocess.run(
                        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=60, env=env,
                    ).stdout.decode('utf-8', 'replace')
                except Exception:  # noqa: BLE001
                    old = ''
            split = LogLineSplitter(every=0.0)
            for line in split.feed(old) + split.flush():
                self._lines.put((prefix, line))
        argv = follow_argv(inst, timestamps=self._timestamps)
        if not running or argv is None or self._stop.is_set():
            return
        # Engines log to stderr, which docker attach passes out on its own stderr.
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env)
        self._live[inst.id] = proc
        self._pump(proc, prefix)

    def _pump(self, proc: subprocess.Popen, prefix: str) -> None:
        import codecs
        import os

        from ..log_filter import LogLineSplitter

        decode = codecs.getincrementaldecoder('utf-8')('replace').decode
        split = LogLineSplitter()
        assert proc.stdout is not None           # Popen(stdout=PIPE)
        fd = proc.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                for line in split.feed(decode(chunk)):
                    self._lines.put((prefix, line))
        finally:
            proc.stdout.close()
        for line in split.flush():
            self._lines.put((prefix, line))

    @property
    def stdout(self) -> Iterator[str]:
        import queue

        while not self._stop.is_set():
            try:
                name, line = self._lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._prefix == 'auto' and len(self._names) <= 1:
                yield line + '\n'
            else:
                yield f'{name}  | {line}\n'

    def terminate(self) -> None:
        self._stop.set()
        for proc in list(self._live.values()):
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                proc.kill()
