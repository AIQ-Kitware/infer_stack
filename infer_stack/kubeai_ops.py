from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    run(
        [
            'helm',
            'repo',
            'add',
            'kubeai',
            'https://www.kubeai.org',
            '--force-update',
        ]
    )
    run(['helm', 'repo', 'update'])

    helm_cmd = [
        'helm',
        'upgrade',
        '--install',
        release_name,
        chart,
        '-n',
        namespace,
        '--create-namespace',
        '-f',
        str(values_file),
        '--wait',
    ]
    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if hf_token:
        helm_cmd.extend(['--set', f'secrets.huggingface.token={hf_token}'])
    run(helm_cmd)

    run(['kubectl', 'apply', '-f', str(models_file)])
    if ingress_file.exists():
        run(['kubectl', 'apply', '-f', str(ingress_file)])


def print_status(namespace: str) -> None:
    run(['kubectl', '-n', namespace, 'get', 'pods'])
    run(['kubectl', '-n', namespace, 'get', 'svc'])
    run(['kubectl', '-n', namespace, 'get', 'ingress'])
    run(['kubectl', '-n', namespace, 'get', 'models'])
