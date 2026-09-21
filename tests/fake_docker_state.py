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
        self.initial_health = 'healthy'        # health a new healthchecked container reports
        self.networks: dict[str, str] = {}     # name -> subnet (persist like the daemon's)

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
                      project=None, cid=None, ips=()):
        cid = cid or f'{service}-{next(_ids)}'
        labels = dict(labels or {})
        labels.setdefault(COMPOSE_PROJECT_LABEL, project or self.project)
        labels.setdefault(COMPOSE_SERVICE_LABEL, service)
        self.containers[cid] = {'service': service, 'labels': labels, 'state': state,
                                'device_ids': [str(d) for d in device_ids],
                                'ips': list(ips), 'restart_count': 0, 'exit_code': 0,
                                'restart_policy': '', 'restart_max': 0}
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
                state = {'Status': c['state'], 'ExitCode': c.get('exit_code', 0)}
                if c.get('health'):
                    state['Health'] = {'Status': c['health']}
                out.append({'Id': cid, 'State': state,
                            'RestartCount': c.get('restart_count', 0),
                            'Config': {'Labels': c['labels']},
                            'HostConfig': {'DeviceRequests': requests,
                                           'RestartPolicy': {
                                               'Name': c.get('restart_policy', ''),
                                               'MaximumRetryCount': c.get('restart_max', 0)}},
                            'NetworkSettings': {'Networks': {
                                'n': {'IPAddress': ip} for ip in c.get('ips', [])}}})
            return json.dumps(out)
        if args[:3] == ['docker', 'network', 'ls']:
            name = args[args.index('--filter') + 1].split('=^', 1)[1].rstrip('$') \
                if '--filter' in args else None
            return '\n'.join(n for n in self.networks if name in (None, n))
        if args[:3] == ['docker', 'network', 'inspect']:
            out = []
            for name in args[3:]:
                attached = {cid: {} for cid, c in self.containers.items()
                            if name in c.get('networks', ())}
                out.append({'Name': name, 'IPAM': {'Config': [{'Subnet': self.networks[name]}]},
                            'Containers': attached})
            return json.dumps(out)
        if args[:3] == ['docker', 'network', 'rm']:
            for name in args[3:]:
                if any(name in c.get('networks', ()) for c in self.containers.values()):
                    raise RuntimeError(f'network {name} has active endpoints')
                self.networks.pop(name, None)
            return ''
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
            for name, spec in (self._doc().get('networks') or {}).items():
                subnet = ((spec.get('ipam') or {}).get('config') or [{}])[0].get('subnet')
                if name in self.networks and self.networks[name] != subnet:
                    # The daemon does: compose will not change an existing network.
                    raise RuntimeError(f'network {name} exists with a different subnet')
                self.networks[name] = subnet
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

    def _doc(self):
        if not self.compose_file or not Path(self.compose_file).exists():
            return {}
        return yaml.safe_load(Path(self.compose_file).read_text()) or {}

    def _services(self):
        return self._doc().get('services') or {}

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
        cid = self.add_container(name, labels=labels, device_ids=device_ids)
        self.containers[cid]['networks'] = list((svc.get('networks') or {}).keys())
        self.containers[cid]['restart_policy'] = str(svc.get('restart') or '')
        if svc.get('healthcheck'):
            self.containers[cid]['health'] = self.initial_health


def answer_residency(args, *, compose_file, running, project):  # pragma: no cover - legacy
    raise NotImplementedError('use ComposeFake')
