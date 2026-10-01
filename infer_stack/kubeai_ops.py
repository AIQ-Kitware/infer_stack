from __future__ import annotations

import copy
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from .config import KUBEAI_GENERATED_SUBDIR, normalized_output


class CommandError(RuntimeError):
    pass


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise CommandError(
            f'Command failed with exit code {proc.returncode}: {" ".join(cmd)}'
        )


def deploy_rendered_artifacts(deployment: dict) -> None:
    cluster = deployment.get('cluster', {})
    namespace = cluster.get('namespace', 'kubeai')
    release_name = cluster.get('kubeai_release_name', 'kubeai')
    chart = cluster.get('kubeai_chart', 'kubeai/kubeai')
    output_root = Path(
        normalized_output(deployment.get('output'))['generated_dir']
    )
    generated = output_root / KUBEAI_GENERATED_SUBDIR
    values_file = generated / 'kubeai-values.yaml'
    namespace_file = generated / 'namespace.yaml'
    models_file = generated / 'models.yaml'
    ingress_file = generated / 'ingress.yaml'

    run(['kubectl', 'apply', '-f', str(namespace_file)])
    install_chart(values=yaml.safe_load(values_file.read_text()) or {},
                  values_path=values_file, namespace=namespace,
                  release=release_name, chart=chart, run=run)

    run(['kubectl', 'apply', '-f', str(models_file)])
    if ingress_file.exists():
        run(['kubectl', 'apply', '-f', str(ingress_file)])


def print_status(namespace: str) -> None:
    run(['kubectl', '-n', namespace, 'get', 'pods'])
    run(['kubectl', '-n', namespace, 'get', 'svc'])
    run(['kubectl', '-n', namespace, 'get', 'ingress'])
    run(['kubectl', '-n', namespace, 'get', 'models'])


def install_chart(*, values: dict, values_path: Path, namespace: str,
                  release: str, chart: str = 'kubeai/kubeai',
                  version: str | None = None, run=run) -> None:
    """Shared Helm upgrade authority; secrets live only in a temporary 0600 file."""
    public = copy.deepcopy(values)
    secrets = public.pop('secrets', {}) or {}
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        secrets.setdefault('huggingface', {})['token'] = token
    values_path.parent.mkdir(parents=True, exist_ok=True)
    values_path.write_text(yaml.safe_dump(public, sort_keys=False), encoding='utf-8')
    run(['helm', 'repo', 'add', 'kubeai', 'https://www.kubeai.org', '--force-update'])
    run(['helm', 'repo', 'update'])
    cmd = ['helm', 'upgrade', '--install', release, chart, '-n', namespace,
           '--create-namespace', '-f', str(values_path), '--wait', '--timeout=300s']
    if version:
        cmd.extend(['--version', version])
    secret_path = None
    try:
        if secrets:
            with tempfile.NamedTemporaryFile('w', prefix='infer-stack-kubeai-secret-',
                                             suffix='.yaml', delete=False) as file:
                os.chmod(file.name, 0o600)
                yaml.safe_dump({'secrets': secrets}, file)
                secret_path = file.name
            cmd.extend(['-f', secret_path])
        run(cmd)
    finally:
        if secret_path:
            Path(secret_path).unlink(missing_ok=True)
