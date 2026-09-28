"""Kubernetes integration commands.

The top-level ``kube`` surface is distribution-neutral and exposes only the
capabilities infer-stack needs.  Provisioning integrations live in explicitly
scoped subcommands such as ``kube k3s``.  Use ``kubectl``/``helm`` for arbitrary
cluster administration.
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


def _print_node_lifecycle(plan) -> None:
    state = 'schedulable' if plan.schedulable else 'cordoned'
    if plan.detached_for_compose:
        if plan.schedulable:
            state += ' (CONFLICT: Compose detach marker is active)'
        else:
            state += ' (detached for Compose)'
    elif plan.detach_state == 'requested':
        state += ' (detach incomplete; retry detach or attach)'
    role = 'control-plane + worker' if plan.control_plane else 'worker'
    print(f'Node: {plan.name}')
    print(f'Role: {role}')
    print(f'Ready: {"yes" if plan.ready else "NO"}')
    print(f'Scheduling: {state}')
    print(f'Workload pods drain would evict: {len(plan.workload_pods)}')
    for name in plan.workload_pods:
        print(f'  - {name}')
    print(f'DaemonSet/static pods retained: {len(plan.retained_pods)}')
    if plan.unmanaged_pods:
        print('Unmanaged pods block safe detach:')
        for name in plan.unmanaged_pods:
            print(f'  - {name}')


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
        headers = ('NAME', 'READY', 'SCHED', 'GPUS', 'GPU PRODUCT', 'GPU MEMORY')
        print(
            f'{headers[0]:<28} {headers[1]:<7} {headers[2]:<8} '
            f'{headers[3]:>4}  {headers[4]:<36} {headers[5]}'
        )
        for row in rows:
            memory = row['gpu_memory_gib']
            memory_text = f'{memory:g} GiB' if memory is not None else '-'
            if row['detached_for_compose'] and row['schedulable']:
                sched = 'CONFLICT'
            elif row['detached_for_compose']:
                sched = 'compose'
            elif row.get('detach_state') == 'requested':
                sched = 'pending'
            else:
                sched = 'yes' if row['schedulable'] else 'NO'
            print(
                f"{row['name']:<28} "
                f"{('yes' if row['ready'] else 'NO'):<7} "
                f'{sched:<8} '
                f"{row['gpu_count']:>4}  "
                f"{str(row['gpu_product'] or '-'):<36} {memory_text}"
            )
        return 0


class KubeNodeStatusCLI(_PathOverridesMixin):
    """Inspect one node's temporary Compose/Kubernetes scheduling state."""

    __command__ = 'status'
    name = kw.Value(None, position=1, type=str, required=True, help='Kubernetes node name.')
    json = kw.Value(False, isflag=True, help='Emit machine-readable JSON.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        try:
            plan = manager.node_lifecycle_plan(config.name)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        if config.json:
            payload = {
                **plan.__dict__,
                'can_detach': plan.can_detach,
                'detach_owned': plan.detach_owned,
                'detached_for_compose': plan.detached_for_compose,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_node_lifecycle(plan)
        return 0


class KubeNodeDetachCLI(_PathOverridesMixin):
    """Temporarily drain/cordon a node so its host can use Compose directly.

    This does not stop kubelet/K3s or remove the node from the cluster.  The
    default is a read-only preview; ``--yes`` performs the cordon + drain and
    marks the cordon as infer-stack-owned so ``node attach`` can safely undo it.
    """

    __command__ = 'detach'
    name = kw.Value(None, position=1, type=str, required=True, help='Kubernetes node name.')
    yes = kw.Value(False, isflag=True, help='Apply the detach after reviewing the preview.')
    timeout = kw.Value(300, type=int, help='kubectl drain timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        try:
            plan = manager.node_lifecycle_plan(config.name)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        _print_node_lifecycle(plan)
        if (
            plan.detached_for_compose
            and not plan.schedulable
            and not plan.workload_pods
        ):
            print('\nNode is already detached for temporary Compose use.')
            return 0
        if not plan.schedulable and not plan.detach_owned:
            raise SystemExit(
                'detach blocked: node is already cordoned by another operator; '
                'infer-stack will not adopt that maintenance cordon'
            )
        if not plan.can_detach:
            raise SystemExit(
                'detach blocked: unmanaged pods are present; move/remove them '
                'explicitly rather than force-deleting them'
            )
        if plan.control_plane:
            print(
                '\nNote: this is a control-plane node. infer-stack will keep the '
                'Kubernetes control plane running; only scheduled workloads are drained.'
            )
        print(
            '\nDetach keeps cluster membership and the Kubernetes agent running. '
            'After the drain, this host may run the Compose backend directly.'
        )
        if not config.yes:
            print(f'Re-run with `infer-stack kube node detach {config.name} --yes` to apply.')
            return 0
        try:
            fresh = manager.detach_node_for_compose(
                config.name, timeout_seconds=config.timeout,
            )
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        print('\nDetached successfully:')
        _print_node_lifecycle(fresh)
        return 0


class KubeNodeAttachCLI(_PathOverridesMixin):
    """Return a Compose-detached node to Kubernetes scheduling.

    The operator must first stop/release local Compose GPU workloads.  The
    default is read-only; ``--yes`` is the explicit assertion that local GPU
    ownership has been handed back to Kubernetes.
    """

    __command__ = 'attach'
    name = kw.Value(None, position=1, type=str, required=True, help='Kubernetes node name.')
    yes = kw.Value(
        False, isflag=True,
        help='Confirm local Compose GPU workloads are stopped and uncordon the node.',
    )
    timeout = kw.Value(180, type=int, help='Ready wait timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        try:
            plan = manager.node_lifecycle_plan(config.name)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        _print_node_lifecycle(plan)
        if plan.schedulable and not plan.detach_owned:
            print('\nNode is already attached to Kubernetes scheduling.')
            return 0
        if not plan.detach_owned:
            raise SystemExit(
                'attach blocked: this cordon is not marked as an infer-stack '
                'temporary Compose detach; refusing to undo operator maintenance'
            )
        print(
            '\nBefore attaching, stop/release every local Compose GPU workload '
            f'on {config.name}. Kubernetes may schedule onto it immediately after uncordon.'
        )
        if not config.yes:
            print(f'Re-run with `infer-stack kube node attach {config.name} --yes` to apply.')
            return 0
        try:
            fresh = manager.attach_node_from_compose(
                config.name, timeout_seconds=config.timeout,
            )
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        print('\nAttached successfully:')
        _print_node_lifecycle(fresh)
        return 0


class KubeNodeModalCLI(kw.ModalCLI):
    """Temporarily hand a cluster node between Kubernetes and local Compose."""

    __command__ = 'node'
    status = KubeNodeStatusCLI
    detach = KubeNodeDetachCLI
    attach = KubeNodeAttachCLI


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
        from ..kube.k3s import K3S_NODE_TOKEN, bootstrap

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
        print(f'K3s join token: sudo cat {K3S_NODE_TOKEN}')
        print('Join workers with: infer-stack kube k3s join --server=... --token-file=...')
        print('Then: infer-stack kube setup')
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
    """First supported provisioning target: create/join K3s clusters.

    This namespace is intentionally distribution-specific.  Existing clusters
    and future provisioning integrations use the same generic ``kube setup``.
    """

    __command__ = 'k3s'
    bootstrap = K3sBootstrapCLI
    join = K3sJoinCLI


class KubeModalCLI(kw.ModalCLI):
    """Inspect/setup infer-stack capabilities on any Kubernetes distribution."""

    __command__ = 'kube'
    nodes = KubeNodesCLI
    node = KubeNodeModalCLI
    setup = KubeSetupCLI
    k3s = K3sModalCLI
