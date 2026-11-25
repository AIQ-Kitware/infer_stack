"""Host-local hardware facts, including GPU identity and the PCIe path."""
from __future__ import annotations

import csv
import hashlib
import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _query(fields):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return [dict(zip(fields, (s.strip() for s in row)))
            for row in csv.reader(result.stdout.splitlines()) if len(row) == len(fields)]


def _bdf(bus):
    # nvidia-smi uses eight hex digits for the domain; Linux sysfs uses four.
    if not bus or ':' not in bus:
        return None
    domain, rest = bus.split(':', 1)
    return domain[-4:].lower().zfill(4) + ':' + rest.lower()


def pci_path(bus, root=Path('/sys/bus/pci/devices')):
    """Return GPU and ancestor bridges, not just the GPU's maximum link."""
    bdf = _bdf(bus)
    if not bdf:
        return []
    try:
        path = (root / bdf).resolve(strict=True)
    except OSError:
        return []
    attrs = ('vendor', 'device', 'subsystem_vendor', 'subsystem_device',
             'class', 'numa_node', 'local_cpulist', 'max_link_speed',
             'max_link_width', 'current_link_speed', 'current_link_width')
    return [{'bdf': p.name, **{k: _read(p / k) for k in attrs}}
            for p in [*reversed(path.parents), path]
            if re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', p.name)]


def host_facts():
    machine = _read('/etc/machine-id')
    cpuinfo = _read('/proc/cpuinfo') or ''
    cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                if line.startswith('model name') and ':' in line), None)
    meminfo = _read('/proc/meminfo') or ''
    ram = next((line.split(':', 1)[1].strip() for line in meminfo.splitlines()
                if line.startswith('MemTotal:')), None)
    return {'hostname': platform.node(), 'machine_id_hash': hashlib.sha256(machine.encode()).hexdigest() if machine else None,
            'system': platform.system(), 'kernel': platform.release(), 'architecture': platform.machine(),
            'cpu': cpu, 'logical_cpus': os.cpu_count(), 'ram': ram,
            'board_vendor': _read('/sys/class/dmi/id/board_vendor'),
            'board_name': _read('/sys/class/dmi/id/board_name')}


def gpu_facts(indices, inventory=None):
    # The placement inventory is the source of the index -> UUID mapping. Never
    # silently assign a different UUID if the driver's enumeration has changed.
    inventory = {int(g['index']): g for g in (inventory or {}).get('gpus', [])}
    base = _query(['index', 'uuid', 'name', 'pci.bus_id', 'memory.total', 'driver_version'])
    extras = _query(['uuid', 'vbios_version', 'pci.device_id', 'pci.sub_device_id',
                     'pcie.link.gen.max', 'pcie.link.width.max', 'power.limit'])
    by_uuid = {g['uuid']: g for g in base}
    extra_by_uuid = {g['uuid']: g for g in extras}
    rows = []
    for idx in indices:
        known = inventory.get(idx, {})
        found = by_uuid.get(known.get('uuid')) if known.get('uuid') else next((g for g in base if g['index'] == str(idx)), None)
        uuid = known.get('uuid') or (found or {}).get('uuid')
        row = {**known, **(found or {}), **extra_by_uuid.get(uuid, {}),
               'index': idx, 'uuid': uuid}
        row['pci_path'] = pci_path(row.get('pci.bus_id'))
        rows.append(row)
    return rows


class Sampler:
    """Sample negotiated link and resource use while requests are in flight."""
    def __init__(self, gpus, interval=1.0):
        self.gpus = {g['uuid']: g for g in gpus if g.get('uuid')}
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.gpus:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join()

    def _loop(self):
        fields = ['uuid', 'memory.used', 'utilization.gpu', 'power.draw',
                  'temperature.gpu', 'clocks.sm', 'clocks.mem',
                  'pcie.link.gen.current', 'pcie.link.width.current']
        start = time.monotonic()
        while not self.stop.is_set():
            rows = _query(fields)
            for row in rows:
                if row['uuid'] not in self.gpus:
                    continue
                path = pci_path(self.gpus[row['uuid']].get('pci.bus_id'))
                self.samples.append({'seconds': round(time.monotonic() - start, 3), **row,
                                     'pci_links': [{k: p[k] for k in ('bdf', 'current_link_speed', 'current_link_width')} for p in path]})
            self.stop.wait(self.interval)


def image_facts(deployment_id):
    """Inspect only the image ID, never container env or mounted credentials."""
    try:
        ids = subprocess.check_output(['docker', 'ps', '-q', '--filter',
                                       'label=infer-stack.deployment=' + deployment_id],
                                      text=True, stderr=subprocess.DEVNULL, timeout=5).split()
        if len(ids) == 1:
            image = subprocess.check_output(['docker', 'inspect', '--format', '{{.Image}}', ids[0]],
                                            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            return {'container_id': ids[0], 'image_id': image}
    except (OSError, subprocess.SubprocessError):
        pass
    return {}
