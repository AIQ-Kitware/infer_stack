"""Answer ``docker ps -a`` / ``docker inspect`` for compose fakes.

The compose test fakes model ``docker compose up`` as "every service in the
file is running". Admission-mode leasing also reads strict residency, which is
plain ``docker ps -a`` plus ``docker inspect``; this answers those from the same
running set, with labels and GPU reservations taken from the compose file.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from infer_stack.leasing.residency import COMPOSE_PROJECT_LABEL


def answer_residency(args, *, compose_file, running, project):
    """Return the stdout for a residency query, or ``None`` if ``args`` is not one."""
    if args[:2] == ['docker', 'ps']:
        fmt = args[args.index('--format') + 1] if '--format' in args else '{{.ID}}'
        want_deployments = any(a == 'label=infer-stack.deployment' for a in args)
        services = _services(compose_file)
        rows = []
        for name in running:
            labels = (services.get(name) or {}).get('labels') or {}
            if want_deployments and 'infer-stack.deployment' not in labels:
                continue
            rows.append(f'{name} running' if '.State' in fmt else name)
        return '\n'.join(rows) + ('\n' if rows else '')
    if args[:2] == ['docker', 'inspect']:
        services = _services(compose_file)
        out = []
        for name in args[2:]:
            svc = services.get(name) or {}
            labels = dict(svc.get('labels') or {})
            labels[COMPOSE_PROJECT_LABEL] = project
            devices = (((svc.get('deploy') or {}).get('resources') or {})
                       .get('reservations') or {}).get('devices') or []
            requests = [{'Driver': d.get('driver'), 'Count': 0,
                         'DeviceIDs': list(d.get('device_ids') or [])} for d in devices] or None
            out.append({'Id': name, 'State': {'Status': 'running'},
                        'Config': {'Labels': labels},
                        'HostConfig': {'DeviceRequests': requests}})
        return json.dumps(out)
    return None


def _services(compose_file):
    if not compose_file or not Path(compose_file).exists():
        return {}
    return (yaml.safe_load(Path(compose_file).read_text()) or {}).get('services') or {}
