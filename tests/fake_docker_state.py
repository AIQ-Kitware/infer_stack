"""A container-level fake of the Docker CLI for compose-backed leasing tests.

``docker compose up`` creates a container per service (all services, or the
ones named after the flags) from the current compose file, recreating one whose
labels changed; ``rm -f``/``stop``/``unpause`` act on containers; ``ps -a`` and
``inspect`` answer strict residency; ``compose ps --format json`` answers
``observe()``. Containers carry the labels and GPU reservations of the stanza
they were created from, so ownership and fingerprints behave as on a daemon.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from infer_stack.leasing.residency import COMPOSE_PROJECT_LABEL, COMPOSE_SERVICE_LABEL

_ids = itertools.count(1)


class ComposeFake:
    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.calls: list[list[str]] = []
        self.compose_file = None
        self.project = 'infer-stack'
        self.started: list[list[str]] = []     # service batches per `up`

    # Existing tests read `running` as the list of running service names.
    @property
    def running(self) -> list[str]:
        return sorted(c['service'] for c in self.containers.values()
                      if c['state'] == 'running')

    @running.setter
    def running(self, services):
        keep = set(services)
        self.containers = {i: c for i, c in self.containers.items() if c['service'] in keep}

    def add_container(self, service, *, labels=None, state='running', device_ids=(),
                      project=None, cid=None):
        cid = cid or f'{service}-{next(_ids)}'
        labels = dict(labels or {})
        labels.setdefault(COMPOSE_PROJECT_LABEL, project or self.project)
        labels.setdefault(COMPOSE_SERVICE_LABEL, service)
        self.containers[cid] = {'service': service, 'labels': labels, 'state': state,
                                'device_ids': [str(d) for d in device_ids]}
        return cid

    def __call__(self, args, **_):
        self.calls.append(list(args))
        if args[:2] == ['docker', 'ps']:
            fmt = args[args.index('--format') + 1] if '--format' in args else '{{.ID}}'
            rows = [f'{i} {c["state"]}' if '.State' in fmt else i
                    for i, c in sorted(self.containers.items())
                    if c['labels'].get(COMPOSE_PROJECT_LABEL) == self.project]
            return '\n'.join(rows) + ('\n' if rows else '')
        if args[:2] == ['docker', 'inspect']:
            out = []
            for cid in args[2:]:
                c = self.containers[cid]
                requests = ([{'Driver': 'nvidia', 'Count': 0, 'DeviceIDs': c['device_ids']}]
                            if c['device_ids'] else None)
                out.append({'Id': cid, 'State': {'Status': c['state']},
                            'Config': {'Labels': c['labels']},
                            'HostConfig': {'DeviceRequests': requests}})
            return json.dumps(out)
        if args[:3] == ['docker', 'rm', '-f']:
            for cid in args[3:]:
                self.containers.pop(cid, None)
            return ''
        if args[:2] == ['docker', 'stop']:
            for cid in args[2:]:
                if cid in self.containers:
                    self.containers[cid]['state'] = 'exited'
            return ''
        if args[:2] == ['docker', 'unpause']:
            for cid in args[2:]:
                self.containers[cid]['state'] = 'running'
            return ''
        if args[:2] != ['docker', 'compose']:
            return ''
        if '-f' in args:
            self.compose_file = args[args.index('-f') + 1]
        if '-p' in args:
            self.project = args[args.index('-p') + 1]
        if 'up' in args:
            services = self._services()
            named = [a for a in args[args.index('up') + 1:] if not a.startswith('-')]
            targets = named or list(services)
            self.started.append(list(targets))
            if '--remove-orphans' in args and not named:
                self.containers = {i: c for i, c in self.containers.items()
                                   if c['service'] in services}
            for name in targets:
                self._up_service(name, services[name])
            return ''
        if 'down' in args:
            self.containers = {}
            return ''
        if 'ps' in args:
            return json.dumps([{'Service': c['service'], 'State': c['state']}
                               for c in self.containers.values()])
        return ''

    def _services(self):
        if not self.compose_file or not Path(self.compose_file).exists():
            return {}
        return (yaml.safe_load(Path(self.compose_file).read_text()) or {}).get('services') or {}

    def _up_service(self, name, svc):
        labels = dict(svc.get('labels') or {})
        devices = (((svc.get('deploy') or {}).get('resources') or {})
                   .get('reservations') or {}).get('devices') or []
        device_ids = [d for dev in devices for d in (dev.get('device_ids') or [])]
        existing = [i for i, c in self.containers.items() if c['service'] == name]
        for cid in existing:
            c = self.containers[cid]
            wanted = {**labels, COMPOSE_PROJECT_LABEL: self.project, COMPOSE_SERVICE_LABEL: name}
            if c['labels'] == wanted and c['device_ids'] == [str(d) for d in device_ids]:
                c['state'] = 'running'
                return
            del self.containers[cid]                      # compose recreates on change
        self.add_container(name, labels=labels, device_ids=device_ids)


def answer_residency(args, *, compose_file, running, project):  # pragma: no cover - legacy
    raise NotImplementedError('use ComposeFake')
