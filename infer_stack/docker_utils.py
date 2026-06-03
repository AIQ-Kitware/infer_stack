from __future__ import annotations

import errno
import json
import socket
import subprocess
from pathlib import Path


class DockerCommandError(RuntimeError):
    pass


class PortInUseError(RuntimeError):
    """One or more host ports the stack needs are already bound."""

    def __init__(self, busy: list[tuple[str, int, str]]):
        self.busy = busy
        lines = ['Cannot start stack: required host ports are already bound.']
        for service, port, host in busy:
            lines.append('')
            lines.append(
                f'  {service}: {host}:{port} is already in use. Find the owner with:'
            )
            lines.append(f"    ss -tlnp 'sport = :{port}'")
            lines.append(f'    sudo lsof -nP -iTCP:{port} -sTCP:LISTEN')
            lines.append(f'    docker ps --filter publish={port}')
        lines.append('')
        lines.append(
            'If the conflict is a leftover container from this stack, run '
            '`infer-stack down` (or `docker stop <name> && docker rm <name>`).'
        )
        lines.append(
            'If a non-stack process owns the port, either stop that process or '
            'pick different ports: `infer-stack setup --litellm-port N '
            '--open-webui-port M`, then `infer-stack render --yes`.'
        )
        super().__init__('\n'.join(lines))


class DockerComposeVersionError(RuntimeError):
    """Docker Compose version is too old and unsupported."""

    def __init__(self, found_version: str):
        msg = (
            f'Docker Compose version {found_version} is too old. '
            'This tool requires Docker Compose v2.0 or later (the newer Go-based implementation).\n\n'
            'Upgrade with one of:\n'
            '  - macOS (Docker Desktop): Update Docker Desktop to 4.4.0+\n'
            '  - Linux (docker-compose standalone): Install docker-compose-plugin\n'
            '    pip install --upgrade docker-compose-plugin\n'
            '    or: curl https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose\n'
            '  - Docker Desktop (Windows/Linux): Update to version 4.4.0+\n\n'
            'After upgrade, verify with: docker compose --version'
        )
        super().__init__(msg)


def check_docker_compose_version(compose_cmd: str) -> None:
    """Verify docker compose is v2.0 or later.

    Raises DockerComposeVersionError if the version is too old.
    """
    try:
        proc = subprocess.run(
            compose_cmd.split() + ['--version'],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return
    if proc.returncode != 0:
        return
    output = proc.stdout.strip() + proc.stderr.strip()
    version_str = _extract_docker_compose_version(output)
    if version_str and _is_old_docker_compose(version_str):
        raise DockerComposeVersionError(version_str)


def _extract_docker_compose_version(output: str) -> str | None:
    """Extract version number from docker-compose/docker compose output.

    Examples of outputs:
      - "docker-compose version 1.29.2, build unknown"
      - "Docker Compose version v2.0.0 (deadbeef)"
      - "Docker Compose version 2.6.1"
    """
    import re

    match = re.search(r'version[:\s]+v?(\d+\.\d+)', output, re.IGNORECASE)
    return match.group(1) if match else None


def _is_old_docker_compose(version: str) -> bool:
    """Check if version is less than 2.0."""
    try:
        major, minor = version.split('.', 1)
        major_int = int(major)
        return major_int < 2
    except (ValueError, AttributeError):
        return False


def our_published_ports(
    compose_cmd: str, compose_file: Path, env_file: Path
) -> set[int]:
    """Return host ports currently published by our own compose project.

    Used by the pre-flight check to skip ports that are "in use" only because
    one of our containers is already publishing them. Requires
    ``docker compose ps --format json`` (compose v2.6+); on older versions or
    when the command otherwise fails this returns an empty set, falling back
    to the strict check.
    """
    if not compose_file.exists():
        return set()
    cmd = compose_cmd.split() + [
        '-f',
        str(compose_file),
        '--env-file',
        str(env_file),
        'ps',
        '--format',
        'json',
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    if proc.returncode != 0 or not proc.stdout.strip():
        return set()
    ports: set[int] = set()
    # `docker compose ps --format json` emits either one JSON object per line
    # (newer versions) or a single JSON array (older versions). Accept both.
    text = proc.stdout.strip()
    parsed_any = False
    if text.startswith('['):
        try:
            for entry in json.loads(text):
                _collect_published_ports(entry, ports)
                parsed_any = True
        except json.JSONDecodeError:
            return set()
    if not parsed_any:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            _collect_published_ports(entry, ports)
    return ports


def _collect_published_ports(entry: dict, out: set[int]) -> None:
    for pub in entry.get('Publishers') or []:
        port = pub.get('PublishedPort')
        if isinstance(port, int) and port > 0:
            out.add(port)


def check_ports_available(ports: list[tuple[str, int, str]]) -> None:
    """Pre-flight check: try to bind each (service, port, host) tuple.

    ``host`` is the interface the rendered compose file binds the publication
    on (e.g. ``"0.0.0.0"`` for litellm / open-webui, ``"127.0.0.1"`` for vllm
    services). Binding on ``0.0.0.0`` collides with any listener on any
    interface for that port (including IPv6 ``[::]`` due to
    IPv4-mapped-IPv6), which is the most common cause of the
    ``failed to bind port`` errors users hit at ``compose up`` time.

    Raises ``PortInUseError`` listing every conflicting service so the user
    sees the full picture in one shot, not one failure at a time.
    """
    busy: list[tuple[str, int, str]] = []
    for service, port, host in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as ex:
                if ex.errno in (errno.EADDRINUSE, errno.EACCES):
                    busy.append((service, port, host))
                else:
                    raise
        finally:
            sock.close()
    if busy:
        raise PortInUseError(busy)


def _cmd(
    compose_cmd: str, compose_file: Path, env_file: Path, *args: str
) -> list[str]:
    return compose_cmd.split() + [
        '-f',
        str(compose_file),
        '--env-file',
        str(env_file),
        *args,
    ]


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise DockerCommandError(
            f'Command failed with exit code {proc.returncode}: {" ".join(cmd)}'
        )


def compose_up(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    *,
    detach: bool = False,
    remove_orphans: bool = True,
    force_recreate: bool = False,
    services: list[str] | None = None,
) -> None:
    args = ['up']
    if detach:
        args.append('-d')
    if remove_orphans:
        args.append('--remove-orphans')
    if force_recreate:
        args.append('--force-recreate')
    if services:
        args.extend(services)
    run(_cmd(compose_cmd, compose_file, env_file, *args))


def compose_down(compose_cmd: str, compose_file: Path, env_file: Path) -> None:
    """Stop and remove services. Never removes named volumes."""
    run(_cmd(compose_cmd, compose_file, env_file, 'down', '--remove-orphans'))


def docker_rm_dirs(dirs: list[Path], docker_cmd: str = 'docker') -> None:
    """Delete host directories that may be root-owned (written by Docker containers).

    Groups paths by parent directory and removes them from inside a temporary
    Alpine container so that permission errors from user-space rm are avoided.
    """
    from collections import defaultdict

    by_parent: dict[Path, list[str]] = defaultdict(list)
    for d in dirs:
        if d.exists():
            by_parent[d.parent].append(d.name)

    for parent, names in by_parent.items():
        targets = ' '.join(f'/mnt/{name}' for name in names)
        cmd = [
            docker_cmd,
            'run',
            '--rm',
            '-v',
            f'{parent}:/mnt',
            'alpine',
            'sh',
            '-c',
            f'rm -rf {targets}',
        ]
        run(cmd)


def compose_recreate_router(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    *,
    detach: bool = True,
) -> None:
    """Recreate the LiteLLM router container in place.

    Forces LiteLLM to reload its config from the rendered YAML. Open WebUI is
    deliberately *not* recreated: it re-fetches ``/v1/models`` from LiteLLM
    on user actions, so a brief stale-cache window is fine, while
    force-recreating would log every user out of the chat UI.
    """
    args = ['up']
    if detach:
        args.append('-d')
    args.extend(
        ['--remove-orphans', '--force-recreate', '--no-deps', 'litellm']
    )
    run(_cmd(compose_cmd, compose_file, env_file, *args))
