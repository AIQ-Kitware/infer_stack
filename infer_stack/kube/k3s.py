"""K3s-specific cluster provisioning convenience.

This is a leaf integration beneath the distribution-neutral ``infer_stack.kube``
capability layer.  Generic setup/backends must not depend on this module.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .manage import default_run

RunFunc = Callable[..., str]
K3S_INSTALL_URL = 'https://get.k3s.io'
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
        return os.environ['KUBECONFIG'] == str(K3S_KUBECONFIG)
    target = Path.home() / '.kube' / 'config'
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            try:
                return target.resolve() == K3S_KUBECONFIG
            except OSError:
                return False
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(K3S_KUBECONFIG)
    return True


def _ensure_helm(run: RunFunc) -> None:
    if shutil.which('helm') is not None:
        return
    installer = run(['curl', '-fsSL', HELM_INSTALL_URL])
    run(['sudo', '-n', 'bash', '-s', '--'], input_text=installer)


def bootstrap(*, version: str | None = None, run: RunFunc | None = None) -> bool:
    """Install/start a local K3s server and make its kubeconfig readable.

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
        if version not in installed:
            raise RuntimeError(
                f'K3s is already active as {installed!r}; requested {version!r}. '
                'Refusing an implicit cluster upgrade; upgrade K3s explicitly.'
            )

    # Keep this as persistent K3s configuration rather than copying the
    # root-owned kubeconfig into ~/.kube (which creates a second authority that
    # can drift when K3s rotates credentials).
    with tempfile.NamedTemporaryFile('w', prefix='infer-stack-k3s-', delete=False) as file:
        file.write('write-kubeconfig-mode: "0644"\n')
        source = file.name
    try:
        run(['sudo', '-n', 'mkdir', '-p', '/etc/rancher/k3s/config.yaml.d'])
        run([
            'sudo', '-n', 'install', '-m', '0644', source,
            '/etc/rancher/k3s/config.yaml.d/10-infer-stack-kubeconfig-mode.yaml',
        ])
    finally:
        Path(source).unlink(missing_ok=True)

    if already_active:
        # Updating the mode fragment does not require restarting a working
        # control plane. Repair current access directly; next start uses it too.
        run(['sudo', '-n', 'chmod', '0644', str(K3S_KUBECONFIG)])
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
        run(cmd, input_text=installer, env=env)

    run([
        'kubectl', '--kubeconfig', str(K3S_KUBECONFIG),
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
    if _active(run, 'k3s-agent'):
        if version:
            installed = run(['k3s', '--version']).splitlines()[0]
            if version not in installed:
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
        ['sudo', '--preserve-env=' + ','.join(preserve), 'sh', '-'],
        input_text=installer,
        env=env,
    )
    run(['systemctl', 'is-active', '--quiet', 'k3s-agent'])
