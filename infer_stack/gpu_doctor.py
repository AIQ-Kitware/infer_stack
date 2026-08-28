"""
Why a GPU looks busy when nothing is on it, and who is actually holding it.

Twice now a card has read 100% utilization with ~0 MiB allocated and an empty
process table, and twice the diagnosis took hours because the obvious tools lie
in a specific way: **unprivileged ``lsof``/``fuser`` only see your own
processes**, so they report "nothing holds it" while ``nvidia-smi -r`` answers
``In use by another client``. A holder check without root is not a clean bill
of health, it is *no information*, and these checks say so rather than
reporting a false all-clear.

What that costs when you get it wrong: an operator resets a card another job is
using. So nothing here acts. It reports, and names the next thing to look at.

The order matters, and is the order these run in:

1. *Sample* utilization. It is a windowed average, so one high reading straight
   after a process exits means nothing; sustained load while sibling cards idle
   is real.
2. Ask the driver for compute apps, which is a different question from the
   process table.
3. Find who holds the device nodes, **as root**, and map each pid to its
   cgroup -- the cgroup names the container or Kubernetes pod, and that
   mapping is the part that is otherwise hard to find.

Known-normal holders are treated as such. ``nvidia-persistenced`` holds every
device by design; flagging it would cry wolf on every healthy machine. It is
still worth naming, because it is why ``nvidia-smi -pm 0`` does not let a reset
through: that turns the *mode* off and leaves the *daemon* holding its handles.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

__all__ = ['GpuSample', 'Holder', 'gpu_checks']

#: A card this full is doing real work; below it, "busy" is noise.
_BUSY_UTIL = 50
#: More than this allocated means somebody's memory is on the card, so high
#: utilization is explained and there is nothing to report.
_IDLE_MEM_MIB = 512
#: Holders that are expected on a healthy machine, matched against argv[0].
_EXPECTED_HOLDERS = ('nvidia-persistenced',)


@dataclass
class GpuSample:
    index: int
    util: int
    mem_used_mib: int
    pstate: str = ''

    @property
    def phantom_busy(self) -> bool:
        """Utilization without memory to explain it."""
        return self.util >= _BUSY_UTIL and self.mem_used_mib <= _IDLE_MEM_MIB


@dataclass
class Holder:
    pid: int
    device: str
    cmdline: str = ''
    cgroup: str = ''

    @property
    def expected(self) -> bool:
        return any(name in self.cmdline for name in _EXPECTED_HOLDERS)

    @property
    def where(self) -> str:
        """The container or pod, read off the cgroup, or '' for a host process."""
        cg = self.cgroup
        pod = re.search(r'kubepods[^/]*/[^/]*pod([0-9a-f_-]{8})', cg)
        if pod:
            return f'kubernetes pod {pod.group(1)}…'
        docker = re.search(r'docker[-/]([0-9a-f]{12})', cg)
        if docker:
            return f'docker container {docker.group(1)}'
        scope = re.search(r'cri-containerd-([0-9a-f]{12})', cg)
        if scope:
            return f'containerd {scope.group(1)}'
        return ''


def _run(cmd, timeout=15):
    """Text stdout, or None. Never raises, never blocks past ``timeout``."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def sample_gpus(samples=3, interval=1.0, _sleep=None):
    """Utilization over time, so a single stale reading cannot fool us."""
    import time
    sleep = _sleep or time.sleep
    seen: dict[int, list[GpuSample]] = {}
    for i in range(samples):
        text = _run(['nvidia-smi',
                     '--query-gpu=index,utilization.gpu,memory.used,pstate',
                     '--format=csv,noheader,nounits'])
        if text is None:
            return []
        for line in text.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                s = GpuSample(int(parts[0]), int(parts[1]), int(parts[2]),
                              parts[3] if len(parts) > 3 else '')
            except ValueError:
                continue
            seen.setdefault(s.index, []).append(s)
        if i + 1 < samples:
            sleep(interval)
    # The minimum utilization across samples: one spike is not sustained load,
    # and this is the number that should drive a "busy" claim.
    out = []
    for idx, series in sorted(seen.items()):
        out.append(GpuSample(idx,
                             min(s.util for s in series),
                             max(s.mem_used_mib for s in series),
                             series[-1].pstate))
    return out


def compute_apps():
    """PIDs the driver attributes to a GPU. Distinct from the process table."""
    text = _run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'])
    if not text:
        return []
    pids = []
    for line in text.strip().splitlines():
        head = line.split(',')[0].strip()
        if head.isdigit():
            pids.append(int(head))
    return pids


def device_holders(use_sudo: bool):
    """Processes holding /dev/nvidia*, or None when we are not allowed to look.

    ``None`` is not "nobody". Unprivileged, this can only see processes the
    caller owns, and every containerd shim and Kubernetes pod on the box is
    owned by root -- which is exactly how a false all-clear happens.

    ``find -lname`` rather than a ``/proc/[0-9]*/fd/*`` glob: the glob expands
    every fd of every process before iteration starts and stalls on a busy box.
    """
    if not use_sudo:
        return None
    text = _run(['sudo', '-n', 'find', '/proc', '-maxdepth', '3',
                 '-path', '*/fd/*', '-lname', '/dev/nvidia*',
                 '-printf', '%h %l\n'], timeout=30)
    if text is None:
        return None
    holders: dict[int, Holder] = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        m = re.match(r'/proc/(\d+)/fd', parts[0])
        if not m:
            continue
        pid = int(m.group(1))
        if pid in holders:
            continue
        holders[pid] = Holder(pid, parts[1],
                              _proc_text(pid, 'cmdline').replace('\0', ' ').strip(),
                              _proc_text(pid, 'cgroup').strip())
    return sorted(holders.values(), key=lambda h: h.pid)


def _proc_text(pid: int, name: str) -> str:
    out = _run(['sudo', '-n', 'cat', f'/proc/{pid}/{name}'], timeout=5)
    return out or ''


def gpu_checks(use_sudo: bool = False, *, _sample=None, _apps=None,
               _holders=None):
    """Yield ``(check, ok, detail)``, matching the backend doctor's shape."""
    gpus = _sample() if _sample else sample_gpus()
    if not gpus:
        yield ('GPUs visible', False, 'nvidia-smi returned nothing')
        return

    apps = _apps() if _apps else compute_apps()
    phantom = [g for g in gpus if g.phantom_busy]
    yield ('GPUs visible', True,
           f'{len(gpus)} card(s), {len(apps)} compute app(s)')

    if not phantom:
        yield ('utilization explained', True,
               'no card is busy without memory allocated')
    else:
        names = ', '.join(f'GPU{g.index} ({g.util}% / {g.mem_used_mib} MiB)'
                          for g in phantom)
        yield ('utilization explained', False,
               f'{names} — busy with nothing allocated. Sustained across '
               f'samples, so not a stale reading. Usually a context left by a '
               f'SIGKILLed container (docker `Exited (137)`). Check whether the '
               f'card still computes before trying to fix it; if it does, this '
               f'is cosmetic and placement is unaffected.')

    holders = _holders() if _holders else device_holders(use_sudo)
    if holders is None:
        yield ('device holders', True,
               'not checked — needs root, and unprivileged this can only see '
               'your own processes, which would read as a false all-clear. '
               'Re-run with --sudo.')
        return

    unexpected = [h for h in holders if not h.expected]
    expected = [h for h in holders if h.expected]
    if not unexpected:
        detail = 'only expected holders'
        if expected:
            detail += f" ({', '.join(sorted({h.cmdline.split()[0].split('/')[-1] for h in expected}))})"
        yield ('device holders', True, detail)
    else:
        lines = []
        for h in unexpected:
            where = f' in {h.where}' if h.where else ''
            lines.append(f'pid {h.pid} {h.cmdline.split()[0][:40]}{where}')
        yield ('device holders', False,
               '; '.join(lines) + ' — these hold a device, so `nvidia-smi -r` '
               'will refuse. Stop the owning container or pod rather than '
               'resetting.')

    if expected:
        yield ('reset would be blocked', True,
               'nvidia-persistenced holds every device by design. `nvidia-smi '
               '-pm 0` turns the mode off but leaves the daemon holding its '
               'handles — stop the service if a reset is genuinely needed.')
