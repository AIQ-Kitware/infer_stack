"""K3s-specific cluster provisioning convenience.

This is a leaf integration beneath the distribution-neutral ``infer_stack.kube``
capability layer.  Generic setup/backends must not depend on this module.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Callable

import yaml

from .manage import default_run

RunFunc = Callable[..., str]
K3S_INSTALL_URL = 'https://get.k3s.io'
K3S_INSTALL_TIMEOUT_SECONDS = 600
HELM_INSTALL_URL = (
    'https://raw.githubusercontent.com/helm/helm/'
    '83a46119086589a593a62ca544982977a60318ca/scripts/get-helm-4'
)


def _require_local_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f'{name} is required')


def _fetch_installer(run: RunFunc) -> str:
    _require_local_tool('curl')
    return run(['curl', '-sfL', K3S_INSTALL_URL])


def _active(run: RunFunc, service: str) -> bool:
    try:
        run(['systemctl', 'is-active', '--quiet', service])
    except Exception:
        return False
    return True


K3S_KUBECONFIG = Path('/etc/rancher/k3s/k3s.yaml')
K3S_NODE_TOKEN = Path('/var/lib/rancher/k3s/server/node-token')


def _ensure_default_kubeconfig_link() -> bool:
    """Point a clean user's default kubeconfig at K3s without copying it.

    Existing kubeconfig state is operator authority and is never replaced.
    """
    if os.environ.get('KUBECONFIG'):
        return os.environ['KUBECONFIG'] == str(user_kubeconfig())
    target = Path.home() / '.kube' / 'config'
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            try:
                return target.resolve() == user_kubeconfig()
            except OSError:
                return False
        return False
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.symlink_to(user_kubeconfig())
    return True


def user_kubeconfig() -> Path:
    return Path.home() / '.kube' / 'infer-stack-k3s.yaml'


def _provision_user_kubeconfig(run: RunFunc) -> None:
    """Refresh an explicitly owned 0600 admin copy; never expose root credentials.

    K3s rotates its admin certificates. Rerunning bootstrap refreshes this copy.
    An unrelated default or KUBECONFIG remains untouched.
    """
    target = user_kubeconfig()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = run(['sudo', '-n', 'cat', str(K3S_KUBECONFIG)])
    if not text.strip():
        raise RuntimeError('Local K3s kubeconfig is empty; cannot provision user access')
    fd, source = tempfile.mkstemp(prefix='.infer-stack-k3s-', dir=target.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(source, target)
    finally:
        Path(source).unlink(missing_ok=True)


def _ensure_helm(run: RunFunc) -> None:
    if shutil.which('helm') is not None:
        return
    installer = run(['curl', '-fsSL', HELM_INSTALL_URL])
    run(['sudo', '-n', 'bash', '-s', '--'], input_text=installer)


def bootstrap(*, version: str | None = None, run: RunFunc | None = None) -> bool:
    """Install/start a local K3s server and provision restricted user access.

    NVIDIA drivers/container runtime are intentionally not installed here.
    K3s detects an already-installed NVIDIA container runtime when it starts;
    ``infer-stack kube doctor`` reports the resulting RuntimeClass/capabilities.
    """
    run = run or default_run
    _require_local_tool('sudo')
    if _active(run, 'k3s-agent'):
        raise RuntimeError('This host is a K3s agent; refusing to replace it with a server')
    already_active = _active(run, 'k3s')
    if already_active and version:
        installed = run(['k3s', '--version']).splitlines()[0]
        if installed.split()[2] != version:
            raise RuntimeError(
                f'K3s is already active as {installed!r}; requested {version!r}. '
                'Refusing an implicit cluster upgrade; upgrade K3s explicitly.'
            )

    # The root admin authority stays 0600. Only this invoking user receives
    # a private copy; bootstrap refreshes it when credentials rotate.
    with tempfile.NamedTemporaryFile('w', prefix='infer-stack-k3s-', delete=False) as file:
        file.write('write-kubeconfig-mode: "0600"\n')
        source = file.name
    try:
        run(['sudo', '-n', 'mkdir', '-p', '/etc/rancher/k3s/config.yaml.d'])
        run([
            'sudo', '-n', 'install', '-m', '0644', source,
            '/etc/rancher/k3s/config.yaml.d/zz-infer-stack-kubeconfig-mode.yaml',
        ])
    finally:
        Path(source).unlink(missing_ok=True)

    run(['sudo', '-n', 'rm', '-f', '/etc/rancher/k3s/config.yaml.d/10-infer-stack-kubeconfig-mode.yaml'])
    if already_active:
        # Updating the mode fragment does not require restarting a working
        # control plane. Repair current access directly; next start uses it too.
        run(['sudo', '-n', 'chmod', '0600', str(K3S_KUBECONFIG)])
    elif shutil.which('k3s') is not None:
        run(['sudo', '-n', 'systemctl', 'start', 'k3s'])
    else:
        installer = _fetch_installer(run)
        env = os.environ.copy()
        preserve = []
        if version:
            env['INSTALL_K3S_VERSION'] = version
            preserve.append('INSTALL_K3S_VERSION')
        cmd = ['sudo', '-n']
        if preserve:
            cmd.append('--preserve-env=' + ','.join(preserve))
        cmd.extend(['sh', '-'])
        run(
            cmd,
            input_text=installer,
            env=env,
            timeout=K3S_INSTALL_TIMEOUT_SECONDS,
        )

    run(['sudo', '-n', 'chmod', '0600', str(K3S_KUBECONFIG)])
    _provision_user_kubeconfig(run)
    run([
        'kubectl', '--kubeconfig', str(user_kubeconfig()),
        'wait', '--for=condition=Ready', 'node', '--all', '--timeout=180s',
    ])
    default_kubeconfig_ready = _ensure_default_kubeconfig_link()
    _ensure_helm(run)
    return default_kubeconfig_ready


def join(
    *,
    server: str,
    token_file: str | Path,
    node_name: str | None = None,
    version: str | None = None,
    run: RunFunc | None = None,
) -> None:
    """Join this host to an existing K3s server without putting its token on argv."""
    run = run or default_run
    _require_local_tool('sudo')
    if _active(run, 'k3s'):
        raise RuntimeError('This host runs a K3s server; refusing to replace it with an agent')
    if _active(run, 'k3s-agent'):
        membership = local_status(run=run)
        if membership['errors']:
            raise RuntimeError('Cannot verify active K3s membership: ' + '; '.join(membership['errors']))
        if membership['server'].rstrip('/') != server.rstrip('/'):
            raise RuntimeError(f"Active K3s agent server {membership['server']!r} differs from requested {server!r}; stop/reconfigure membership explicitly")
        if node_name and membership['node_name'] != node_name:
            raise RuntimeError(f"Active K3s agent node name {membership['node_name']!r} differs from requested {node_name!r}")
        if version:
            installed = run(['k3s', '--version']).splitlines()[0]
            if installed.split()[2] != version:
                raise RuntimeError(
                    f'K3s agent is already active as {installed!r}; requested '
                    f'{version!r}. Refusing an implicit agent upgrade.'
                )
        return

    token_path = Path(token_file).expanduser()
    if not token_path.is_file():
        raise RuntimeError(f'K3s token file does not exist: {token_path}')
    token = token_path.read_text(encoding='utf-8').strip()
    if not token:
        raise RuntimeError(f'K3s token file is empty: {token_path}')

    installer = _fetch_installer(run)
    env = os.environ.copy()
    env['K3S_URL'] = server
    env['K3S_TOKEN'] = token
    preserve = ['K3S_URL', 'K3S_TOKEN']
    if node_name:
        env['K3S_NODE_NAME'] = node_name
        preserve.append('K3S_NODE_NAME')
    if version:
        env['INSTALL_K3S_VERSION'] = version
        preserve.append('INSTALL_K3S_VERSION')
    run(
        ['sudo', '-n', '--preserve-env=' + ','.join(preserve), 'sh', '-'],
        input_text=installer,
        env=env,
        timeout=K3S_INSTALL_TIMEOUT_SECONDS,
    )
    run(['systemctl', 'is-active', '--quiet', 'k3s-agent'])


def local_status(*, run: RunFunc | None = None) -> dict:
    """Local membership independent of selected admin kubeconfig, without secrets.

    Inspect the running agent's actual argv/environment, then K3s config with
    its drop-ins. Unknown/unreadable configuration is reported and cannot make
    a requested join look idempotent.
    """
    run = run or default_run
    report = {'agent_active': _active(run, 'k3s-agent'),
              'server_active': _active(run, 'k3s'), 'server': '',
              'node_name': '', 'version': '', 'errors': []}
    if not report['agent_active'] and not report['server_active']:
        return report
    try:
        report['version'] = run(['k3s', '--version']).splitlines()[0]
    except Exception:
        report['errors'].append('Unable to read local K3s version')
    if not report['agent_active']:
        return report
    try:
        pid = run(['systemctl', 'show', 'k3s-agent', '--property=MainPID', '--value']).strip()
        if not pid.isdigit() or int(pid) == 0:
            raise ValueError('agent PID unavailable')
        argv = run(['sudo', '-n', 'cat', f'/proc/{pid}/cmdline']).split('\0')
        environ = run(['sudo', '-n', 'cat', f'/proc/{pid}/environ']).split('\0')
        env = dict(v.split('=', 1) for v in environ if '=' in v)
        flags = {}
        for i, arg in enumerate(argv):
            if arg.startswith('--'):
                key, sep, value = arg[2:].partition('=')
                flags[key] = value if sep else (argv[i + 1] if i + 1 < len(argv) else '')
        config_path = flags.get('config') or env.get('K3S_CONFIG_FILE') or '/etc/rancher/k3s/config.yaml'
        # Read only through the privilege seam. The command emits YAML only
        # into memory; secrets never enter the status report or diagnostics.
        files = json.loads(run(['sudo', '-n', 'python3', '-c',
            'import glob,json,pathlib,sys; p=sys.argv[1]; '
            'print(json.dumps([pathlib.Path(f).read_text() for f in ([p] if pathlib.Path(p).exists() else []) + sorted(glob.glob(p+".d/*.yaml"))]))',
            config_path]))
        config = {}
        for text in files:
            values = yaml.safe_load(text) or {}
            config.update(values)
        report['server'] = str(flags.get('server') or env.get('K3S_URL') or config.get('server') or '')
        report['node_name'] = str(flags.get('node-name') or env.get('K3S_NODE_NAME') or config.get('node-name') or socket.gethostname().lower())
        if not report['server']:
            report['errors'].append('Active agent server is unknown')
    except Exception:
        report['errors'].append('Cannot inspect active agent configuration; authenticate with sudo -v then retry')
    return report


def export_join_info(*, server: str, directory: str | Path, run: RunFunc | None = None) -> dict:
    """Export a private join bundle; refresh only a bundle owned by this cluster.

    Token and rewritten admin kubeconfig never appear on stdout. The original
    root/default configs are unchanged. The manifest is written first so an
    interrupted export can safely complete on retry.
    """
    import hashlib

    run = run or default_run
    if not server.startswith('https://'):
        raise RuntimeError('Join server must use https:// with a reachable LAN/VPN address')
    if not _active(run, 'k3s'):
        raise RuntimeError('Export must run on the local K3s server; an agent/selected remote context is not sufficient')
    config = yaml.safe_load(run(['sudo', '-n', 'cat', str(K3S_KUBECONFIG)]))
    if not isinstance(config, dict):
        raise RuntimeError('Local K3s kubeconfig must contain a mapping')
    token = run(['sudo', '-n', 'cat', str(K3S_NODE_TOKEN)]).strip()
    if not token:
        raise RuntimeError('Local K3s node token is empty')
    contexts = {c['name']: c['context'] for c in config.get('contexts') or []}
    selected = contexts.get(config.get('current-context'), {}).get('cluster')
    clusters = [c for c in config.get('clusters') or [] if c.get('name') == selected]
    if len(clusters) != 1 or not clusters[0].get('cluster', {}).get('certificate-authority-data'):
        raise RuntimeError('Local K3s kubeconfig has no unique selected cluster with embedded CA data')
    ca_data = clusters[0]['cluster']['certificate-authority-data']
    if not isinstance(ca_data, str):
        raise RuntimeError('Local K3s CA data must be a string')
    fingerprint = hashlib.sha256(ca_data.encode()).hexdigest()
    clusters[0]['cluster']['server'] = server.rstrip('/')
    target = Path(directory).expanduser()
    marker = target / 'manifest.json'
    metadata = {'kind': 'infer-stack-k3s-join', 'cluster_ca_sha256': fingerprint, 'server': server.rstrip('/')}
    if target.exists():
        if target.is_symlink() or not target.is_dir() or target.stat().st_uid != os.getuid():
            raise RuntimeError('Join export directory must be a directory owned by this user, not a symlink')
        if target.stat().st_mode & 0o077:
            raise RuntimeError(f'Join export directory must be private (chmod 700): {target}')
        if any(target.iterdir()):
            if not marker.is_file():
                raise RuntimeError('Refusing to overwrite a nonempty directory without an infer-stack join manifest')
            previous = json.loads(marker.read_text())
            if previous.get('kind') != metadata['kind'] or previous.get('cluster_ca_sha256') != fingerprint:
                raise RuntimeError('Join export belongs to a different cluster; choose a new directory')
    else:
        target.mkdir(parents=True, mode=0o700)
    documents = {'manifest.json': json.dumps(metadata, indent=2) + '\n',
                 'kubeconfig.yaml': yaml.safe_dump(config), 'token': token + '\n'}
    for name, content in documents.items():
        # NamedTemporaryFile creates mode 0600 before any secret is written.
        with tempfile.NamedTemporaryFile('w', dir=target, prefix='.join-', delete=False) as temporary:
            pending = Path(temporary.name)
            try:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(pending, target / name)
            finally:
                pending.unlink(missing_ok=True)
    return {'server': metadata['server'], 'directory': str(target),
            'kubeconfig': str(target / 'kubeconfig.yaml'), 'token_file': str(target / 'token')}
