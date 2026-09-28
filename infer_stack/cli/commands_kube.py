"""Kubernetes integration commands.

``kube`` intentionally exposes only infer-stack's integration surface.  Use
``kubectl``/``helm`` for arbitrary cluster administration.
"""

from __future__ import annotations

import json
from pathlib import Path

import kwconf as kw

from ..kube import KubeManager
from ..kube.manage import NVIDIA_DEVICE_PLUGIN_VERSION
from .context import _apply_path_overrides
from .options import _PathOverridesMixin


def _namespace(config) -> str:
    from ..paths import get_setting

    return str(config.namespace or get_setting('kubeai_namespace') or 'kubeai')


def _print_plan(plan) -> None:
    if plan.context:
        print(f'Kubernetes context: {plan.context}')
    for check in plan.checks:
        if check.ok:
            mark = 'ok  '
        elif check.level == 'action':
            mark = 'NEED'
        elif check.level == 'advisory':
            mark = 'WARN'
        else:
            mark = 'FAIL'
        line = f'[{mark}] {check.name}'
        if check.detail:
            line += f' — {check.detail}'
        print(line)
    if plan.resource_profiles:
        print('\nresource profiles discovered:')
        for name, spec in plan.resource_profiles.items():
            product = (spec.get('nodeSelector') or {}).get('nvidia.com/gpu.product', '?')
            print(f'  {name}: {product}')
    if plan.actions:
        print('\nplan:')
        for action in plan.actions:
            print(f'  [plan] {action}')
    else:
        print('\nplan: no managed changes needed')


class KubeNodesCLI(_PathOverridesMixin):
    """Show the cluster nodes and the GPU facts infer-stack uses."""

    __command__ = 'nodes'
    json = kw.Value(False, isflag=True, help='Emit machine-readable JSON.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        try:
            rows = manager.node_rows(manager.nodes())
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        if config.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
            return 0
        if not rows:
            print('(cluster has no nodes)')
            return 1
        headers = ('NAME', 'READY', 'GPUS', 'GPU PRODUCT', 'GPU MEMORY')
        print(f'{headers[0]:<28} {headers[1]:<7} {headers[2]:>4}  {headers[3]:<36} {headers[4]}')
        for row in rows:
            memory = row['gpu_memory_gib']
            memory_text = f'{memory:g} GiB' if memory is not None else '-'
            print(
                f"{row['name']:<28} "
                f"{('yes' if row['ready'] else 'NO'):<7} "
                f"{row['gpu_count']:>4}  "
                f"{str(row['gpu_product'] or '-'):<36} {memory_text}"
            )
        return 0


class KubeSetupCLI(_PathOverridesMixin):
    """Inspect or reconcile the Kubernetes capabilities infer-stack needs.

    The default is read-only.  ``--apply`` installs/reconciles only components
    infer-stack owns: the NVIDIA device-plugin/GFD convenience path (when an
    NVIDIA RuntimeClass says the host runtime is ready), and KubeAI with
    discovered resource profiles.  Existing working GPU integrations and
    operator-customized named profiles are preserved.
    """

    __command__ = 'setup'
    apply = kw.Value(False, isflag=True, help='Apply the displayed managed changes.')
    namespace = kw.Value(None, type=str, help='KubeAI namespace (default: configured namespace or kubeai).')
    release = kw.Value('kubeai', type=str, help='KubeAI Helm release name.')
    gpu = kw.Value(
        'auto', type=str, choices=['auto', 'nvidia', 'none'],
        help='GPU integration: auto-detect, require/manage NVIDIA, or skip GPU checks.',
    )
    nvidia_plugin_version = kw.Value(
        NVIDIA_DEVICE_PLUGIN_VERSION, type=str,
        help='NVIDIA device-plugin chart version used when infer-stack installs it.',
    )
    kubeai_version = kw.Value(
        None, type=str,
        help='Optional KubeAI chart version pin (existing/default chart version otherwise).',
    )
    values = kw.Value(
        None, type=str,
        help='Optional operator KubeAI Helm values file. Its settings win; discovered profiles fill missing names.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        namespace = _namespace(config)
        manager = KubeManager()
        try:
            operator_values = manager.load_values_file(config.values)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        if not config.apply:
            try:
                plan = manager.plan_setup(
                    namespace=namespace, release=config.release, gpu=config.gpu,
                    operator_values=operator_values,
                )
            except RuntimeError as ex:
                raise SystemExit(str(ex)) from ex
            _print_plan(plan)
            print('\nNo changes made. Re-run with --apply to reconcile managed components.')
            return 1 if plan.failed else 0

        try:
            plan, values_path = manager.apply_setup(
                namespace=namespace,
                release=config.release,
                gpu=config.gpu,
                nvidia_plugin_version=config.nvidia_plugin_version,
                kubeai_version=config.kubeai_version,
                operator_values=operator_values,
            )
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        _print_plan(plan)
        if values_path is not None:
            print(f'\nKubeAI values authority: {values_path}')
        return 1 if plan.failed else 0


class K3sBootstrapCLI(kw.Config):
    """Install/start a local K3s server (explicit convenience path)."""

    __command__ = 'bootstrap'
    version = kw.Value(
        None, type=str,
        help='Optional exact K3s version, e.g. v1.34.3+k3s1. Omit to use the K3s install channel.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        from ..kube.k3s import bootstrap

        try:
            default_kubeconfig_ready = bootstrap(version=config.version)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        print('K3s server is active and all nodes are Ready.')
        print('K3s kubeconfig: /etc/rancher/k3s/k3s.yaml')
        if not default_kubeconfig_ready:
            print(
                'Existing ~/.kube/config was left unchanged. Make sure its '
                'current context selects this K3s cluster (or set KUBECONFIG) '
                'before running setup.'
            )
        print('Next: infer-stack kube setup')
        return 0


class K3sJoinCLI(kw.Config):
    """Join this machine to a K3s server as an agent."""

    __command__ = 'join'
    server = kw.Value(None, type=str, required=True, help='K3s server URL, e.g. https://host:6443.')
    token_file = kw.Value(
        None, type=str, required=True,
        help='File containing the K3s node token (keeps the secret out of argv/history).',
    )
    node_name = kw.Value(None, type=str, help='Optional Kubernetes node name.')
    version = kw.Value(None, type=str, help='Exact K3s version; normally match the server.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        from ..kube.k3s import join

        try:
            join(
                server=config.server,
                token_file=Path(config.token_file),
                node_name=config.node_name,
                version=config.version,
            )
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        print('K3s agent is active.')
        return 0


class K3sModalCLI(kw.ModalCLI):
    """Create/join K3s clusters; not required for existing Kubernetes clusters."""

    __command__ = 'k3s'
    bootstrap = K3sBootstrapCLI
    join = K3sJoinCLI


class KubeModalCLI(kw.ModalCLI):
    """Inspect/setup the Kubernetes capabilities used by infer-stack."""

    __command__ = 'kube'
    nodes = KubeNodesCLI
    setup = KubeSetupCLI
    k3s = K3sModalCLI
