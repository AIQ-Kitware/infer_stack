"""Stable per-service addresses (plan step P7).

The problem: when a container is recreated and another container receives its
old IP, the LiteLLM gateway can keep a pooled connection to that IP and send one
model's traffic to another indefinitely, while Docker DNS is correct (reproduced
on the host). The fix is that an address, once given to a service name, is never
given to any other service.

* **Explicit network.** Once ``infer-stack network migrate`` has run, the
  rendered project declares one named network with a fixed IPAM subnet, and
  every service gets a static ``ipv4_address`` on it.
* **Append-only address table.** ``service_addresses(service, ipv4)`` in the
  ledger. Allocation is sequential and skips the network and broadcast
  addresses and the gateway (``.1``). A service name keeps its address across
  recreation; no other name ever receives it.
* **Preflight.** Before the first allocation, a subnet overlapping an existing
  Docker network or a host route is rejected.

Example:
    >>> next_free_address('172.30.0.0/29', {'a': '172.30.0.2'})
    '172.30.0.3'
    >>> next_free_address('172.30.0.0/30', {'a': '172.30.0.2'}) is None
    True
"""

from __future__ import annotations

import ipaddress
import json
import subprocess

NETWORK_NAME = 'infer-stack-net'


def next_free_address(subnet: str, taken: dict[str, str]) -> str | None:
    """The lowest usable host address not in ``taken``, or ``None`` if exhausted.

    ``.1`` (the Docker gateway) is never allocated.
    """
    net = ipaddress.ip_network(subnet)
    used = {ipaddress.ip_address(ip) for ip in taken.values()}
    gateway = net.network_address + 1
    for host in net.hosts():
        if host == gateway or host in used:
            continue
        return str(host)
    return None


def allocate(subnet: str, existing: dict[str, str], services) -> dict[str, str]:
    """Addresses for ``services``: existing ones kept, new ones appended in order."""
    table = dict(existing)
    for name in sorted(services):
        if name in table:
            continue
        ip = next_free_address(subnet, table)
        if ip is None:
            raise RuntimeError(f'subnet {subnet} has no free address for service {name!r}')
        table[name] = ip
    return table


def stamp_network(compose: dict, subnet: str, addresses: dict[str, str]) -> None:
    """Put every service on the fixed network at its address (in place)."""
    compose['networks'] = {
        NETWORK_NAME: {
            'name': NETWORK_NAME,
            'ipam': {'config': [{'subnet': subnet}]},
        }
    }
    for name, svc in (compose.get('services') or {}).items():
        svc['networks'] = {NETWORK_NAME: {'ipv4_address': addresses[name]}}


def overlapping_subnets(subnet: str, run) -> list[str]:
    """Docker networks and host routes that overlap ``subnet`` (preflight)."""
    net = ipaddress.ip_network(subnet)
    found = []
    ids = [i for i in (run(['docker', 'network', 'ls', '-q']) or '').split() if i]
    if ids:
        for item in json.loads(run(['docker', 'network', 'inspect', *ids]) or '[]'):
            if item.get('Name') == NETWORK_NAME:
                continue
            for cfg in ((item.get('IPAM') or {}).get('Config') or []):
                other = cfg.get('Subnet')
                if other and ':' not in other and net.overlaps(ipaddress.ip_network(other, strict=False)):
                    found.append(f'docker network {item.get("Name")} ({other})')
    try:
        routes = subprocess.run(['ip', '-4', 'route'], capture_output=True, text=True,
                                timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        routes = ''
    for line in routes.splitlines():
        first = line.split()[0] if line.split() else ''
        if first in ('default', '') or '/' not in first:
            continue
        try:
            if net.overlaps(ipaddress.ip_network(first, strict=False)):
                found.append(f'host route {first}')
        except ValueError:
            continue
    return found


UPSTREAM_CHECK_SCRIPT = (
    'import json,sys,urllib.request\n'
    'try:\n'
    '    r = urllib.request.urlopen(sys.argv[1], timeout=5)\n'
    '    print(json.dumps([m.get("id") for m in json.load(r).get("data", [])]))\n'
    'except Exception as ex:\n'
    '    print(json.dumps({"error": str(ex)}))\n'
)


def classify_upstream(expected: str, answer) -> str:
    """``healthy`` | ``not-ready`` | ``routing-fault`` for one upstream probe.

    ``answer`` is what the upstream reported when reached by name from inside
    the gateway's network: a list of model ids, or ``{'error': ...}``.

    Example:
        >>> classify_upstream('qwen', ['qwen'])
        'healthy'
        >>> classify_upstream('qwen', ['llama'])
        'routing-fault'
        >>> classify_upstream('qwen', {'error': 'connection refused'})
        'not-ready'
    """
    if isinstance(answer, dict) or answer is None:
        return 'not-ready'
    return 'healthy' if expected in answer else 'routing-fault'
