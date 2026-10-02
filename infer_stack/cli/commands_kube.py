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
from .commands_kube_worker import K3sExportCLI, K3sOnboardCLI, KubeNodeTestCLI
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
    test = KubeNodeTestCLI
    detach = KubeNodeDetachCLI
    attach = KubeNodeAttachCLI


class KubeSetupCLI(_PathOverridesMixin):
    """Deprecated compatibility alias of kube install.

    Use kube bootstrap for local K3s prerequisites. Setup delegates KubeAI
    planning/apply to the install command and has no independent orchestration.
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
        print('Deprecated: kube setup is an alias of kube install; use kube bootstrap for local K3s prerequisites.')
        if config.nvidia_plugin_version != NVIDIA_DEVICE_PLUGIN_VERSION:
            raise SystemExit('Use the pinned NVIDIA bootstrap path; setup no longer owns plugin installation')
        return KubeInstallCLI.main(argv=False, namespace=_namespace(config), release=config.release,
                                   gpu='none' if config.gpu == 'none' else 'nvidia',
                                   apply=config.apply, values=config['values'], version=config.kubeai_version)


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
        from ..kube.k3s import K3S_NODE_TOKEN, bootstrap, user_kubeconfig

        try:
            default_kubeconfig_ready = bootstrap(version=config.version)
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex
        print('K3s server is active and all nodes are Ready.')
        print('Root admin kubeconfig: /etc/rancher/k3s/k3s.yaml (0600)')
        print(f'Private user kubeconfig: {user_kubeconfig()} (0600); rerun bootstrap to refresh certificates')
        if not default_kubeconfig_ready:
            print(
                'Existing ~/.kube/config was left unchanged. Make sure its '
                'current context selects this K3s cluster before installation.'
                f' Select it explicitly with: export KUBECONFIG={user_kubeconfig()}'
            )
        print(f'K3s join token: sudo cat {K3S_NODE_TOKEN}')
        print('Join workers with: infer-stack kube k3s join --server=... --token-file=...')
        print('Then: infer-stack kube inventory; infer-stack kube bootstrap; infer-stack kube install')
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
        import socket

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
        print('K3s agent is active; this does not change the selected admin kubeconfig.')
        print('Inspect local membership: infer-stack kube k3s status')
        manager = KubeManager()
        if manager.command_exists('nvidia-smi'):
            try:
                products = manager.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).strip()
            except RuntimeError:
                print('Local GPU inspection failed; verify driver/runtime preparation and cluster-side GPU availability from the control plane.')
            else:
                if products:
                    print('Local GPUs: ' + ', '.join(products.splitlines()))
                    print('Local GPU detected, but cluster-side GPU availability must be verified from the control plane.')
                    print('Join does not install NVIDIA drivers/toolkit or prove GPU runtime readiness.')
                    print('Verify this node is Ready, exposes nvidia.com/gpu, has GFD product/memory labels, and passes a fresh runtime canary via infer-stack kube install --apply.')
        print(f'On the control plane: infer-stack kube node status {config.node_name or socket.gethostname().lower()}')
        return 0


class K3sStatusCLI(kw.Config):
    """Local K3s server/agent membership, independent of selected admin context."""
    __command__ = 'status'
    json = kw.Value(False, isflag=True, help='Emit membership facts without credentials.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..kube.k3s import local_status
        config = cls.cli(argv=argv, data=kwargs)
        report = local_status()
        if config.json:
            print(json.dumps(report, indent=2))
        else:
            print('local Kubernetes membership (not selected admin context)')
            for key, value in report.items():
                print(f'  {key}: {value}')
        return int(bool(report['errors']))


class K3sModalCLI(kw.ModalCLI):
    """First supported provisioning target: create/join K3s clusters.

    This namespace is intentionally distribution-specific.  Existing clusters
    and future provisioning integrations use generic inventory/doctor/install.
    """

    __command__ = 'k3s'
    bootstrap = K3sBootstrapCLI
    join = K3sJoinCLI
    onboard = K3sOnboardCLI
    export = K3sExportCLI
    status = K3sStatusCLI


def _inventory_for_config(config, manager):
    from ..kube.inspect import inventory
    from ..paths import get_setting

    return inventory(manager, namespace=_namespace(config), release=config.release,
                     base_url=config.base_url or get_setting('kubeai_base_url') or
                     'http://127.0.0.1:8000/openai/v1')


def _print_inventory(report):
    import yaml

    if report.get('local_membership') is not None:
        print('local Kubernetes membership')
        for key, value in report['local_membership'].items():
            print(f'  {key}: {value}')
    print('selected admin cluster')
    print('  RuntimeClass objects: ' + ', '.join(report['runtime_classes'] or []))
    for key, value in report['cluster'].items():
        print(f'  {key}: {value}')
    print(f"host GPUs: {report['host_gpus']}")
    print('tools')
    for key, value in report['tools'].items():
        print(f'  {key}: {"available" if value else "missing"}')
    for node in report['nodes']:
        print(f"node {node['name']}")
        for key in ('ready', 'nvidia_runtime_verified', 'nvidia_runtime_evidence_state', 'nvidia_runtime_observed_at',
                    'nvidia_runtime_evidence', 'gpu_count', 'gpu_product',
                    'gpu_memory_gib', 'gfd_labels'):
            print(f'  {key}: {node[key]}')
    plugins = report['device_plugin']
    print('NVIDIA device plugin: ' + ('unknown' if plugins is None else 'missing' if not plugins else
          ', '.join(f"{p['name']} ({str(p['ready']) + '/' + str(p['desired']) + ' ready' if p['desired'] > 0 else 'N/A: zero scheduled'})" for p in plugins)))
    kubeai = report['kubeai']
    print('KubeAI')
    print(f"  Model CRD: {'installed' if kubeai['crd'] else 'missing' if kubeai['crd'] is False else 'unknown'}")
    print(f"  namespace {kubeai['namespace']}: {'present' if kubeai['namespace_exists'] else 'missing' if kubeai['namespace_exists'] is False else 'unknown'}")
    release = kubeai['release']
    print(f"  Helm release: {release.get('chart')} ({release.get('status')})" if release else
          '  Helm release: missing/unknown')
    api = kubeai['api']
    print(f"  API: {api['base_url']} — {'available' if api['reachable'] else 'unavailable'}")
    if not api['reachable']:
        print(f"    {api.get('error') or api.get('blocked') or api.get('status_code') or ''}")
    for key in ('pods', 'services', 'models'):
        items = kubeai[key]
        print(f"  {key}: {len(items) if items is not None else 'unknown/blocked'}")
        for item in items or []:
            name = item.get('metadata', {}).get('name', '?')
            status = item.get('status', {})
            if key == 'pods':
                containers = status.get('containerStatuses', [])
                summary = f"{status.get('phase', '?')}; {sum(bool(c.get('ready')) for c in containers)}/{len(containers)} containers ready"
            elif key == 'services':
                spec = item.get('spec', {})
                summary = f"{spec.get('type', '?')} {spec.get('clusterIP', '')}"
            else:
                from ..backends.kubeai import model_replica_counts
                total, ready = model_replica_counts(item)
                summary = f"{ready if ready is not None else '?'}/{total if total is not None else '?'} ready replicas"
            print(f'    {name}: {summary}')
    print('resource profiles proposed')
    if report['resource_profiles']['proposed']:
        print(yaml.safe_dump(report['resource_profiles']['proposed'], sort_keys=False), end='')
    else:
        print('  none: discover allocatable GPUs with GFD product/memory labels through a reachable admin context')
    print('resource profiles installed: ' + (', '.join(report['resource_profiles']['installed']) or 'none'))
    for key, error in report['errors'].items():
        print(f'  probe {key}: {error}')


def _print_checks(checks):
    for check in checks:
        mark = {'ok': 'ok  ', 'warn': 'WARN', 'fail': 'FAIL', 'blocked': 'SKIP'}[check['status']]
        print(f"[{mark}] {check['name']} — {check['detail']}")
        if check['fix']:
            print(f"       fix: {check['fix']}")


class KubeInventoryCLI(_PathOverridesMixin):
    """Describe Kubernetes, GPU and KubeAI facts, including partial installations."""

    __command__ = 'inventory'
    namespace = kw.Value(None, type=str, help='Configured KubeAI namespace override.')
    release = kw.Value('kubeai', type=str, help='KubeAI Helm release name.')
    base_url = kw.Value(None, type=str, help='Configured KubeAI OpenAI API URL override.')
    json = kw.Value(False, isflag=True, help='Emit structured facts and probe errors.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        report = _inventory_for_config(config, KubeManager())
        if config.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_inventory(report)
        return 0


class KubeStatusCLI(KubeInventoryCLI):
    """Show cluster GPUs, KubeAI release, pods, services and managed Models."""

    __command__ = 'status'


class KubeDoctorCLI(KubeInventoryCLI):
    """Check Kubernetes/GPU prerequisites and KubeAI readiness in dependency order."""

    __command__ = 'doctor'
    gpu = kw.Value('nvidia', type=str, choices=['nvidia', 'none'], help='Use none for explicitly configured CPU KubeAI.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..kube.inspect import failed, readiness

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        report = _inventory_for_config(config, KubeManager())
        checks = readiness(report, gpu=config.gpu != 'none')
        if config.json:
            print(json.dumps({'inventory': report, 'checks': checks}, indent=2))
        else:
            _print_checks(checks)
        return int(failed(checks))


class KubeInstallCLI(KubeInventoryCLI):
    """Plan KubeAI values; --apply installs/upgrades the chart and checks readiness."""

    __command__ = 'install'
    gpu = kw.Value('nvidia', type=str, choices=['nvidia', 'none'], help='Use none with explicit CPU resource profiles.')
    apply = kw.Value(False, isflag=True, alias=['yes'], help='Apply the displayed Helm installation plan.')
    dry_run = kw.Value(False, isflag=True, alias=['plan'], help='Only inspect values, even with --apply.')
    values = kw.Value(None, type=str, help='Additional Helm values; custom named profiles take precedence.')
    chart = kw.Value('kubeai/kubeai', type=str, help='KubeAI chart reference.')
    version = kw.Value(None, type=str, help='Chart version; preserves installed version by default.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import yaml

        from ..kube.inspect import failed, readiness
        from ..kube.operations import install, install_plan

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        try:
            operator = manager.load_values_file(config['values'])
            report = _inventory_for_config(config, manager)
            if config.gpu == 'none':
                report['resource_profiles']['proposed'] = {}
            values = install_plan(manager, report, operator_values=operator)
            checks = readiness(report, installation=False, gpu=config.gpu != 'none')
            if config.json:
                if not config.apply or config.dry_run:
                    print(json.dumps({'values': values, 'checks': checks}, indent=2))
            else:
                _print_checks(checks)
                print(yaml.safe_dump(values, sort_keys=False), end='')
            if not config.apply or config.dry_run:
                if not config.json:
                    print('No changes made. Run infer-stack kube install --apply to install/upgrade.')
                return int(failed(checks))
            install(manager, report, operator_values=operator, chart=config.chart, version=config.version, gpu=config.gpu != 'none')
            checks = readiness(_inventory_for_config(config, manager), gpu=config.gpu != 'none')
            if config.json:
                print(json.dumps({'values': values, 'checks': checks}, indent=2))
            else:
                _print_checks(checks)
            return int(failed(checks))
        except RuntimeError as ex:
            raise SystemExit(str(ex)) from ex


class KubeBootstrapCLI(KubeInventoryCLI):
    """Plan a local K3s server and GPU prerequisites; never targets unrelated selected contexts."""

    __command__ = 'bootstrap'
    provider = kw.Value('k3s', choices=['k3s'], type=str, help='Provision local K3s, regardless of selected admin context.')
    version = kw.Value(None, type=str, help='Optional exact K3s version; no implicit upgrades.')
    apply = kw.Value(False, isflag=True, alias=['yes'], help='Authorize host/cluster bootstrap changes.')
    dry_run = kw.Value(False, isflag=True, alias=['plan'], help='Only display the plan.')
    timeout = kw.Value(180, type=int, help='Ready/GPU discovery wait timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..kube.inspect import failed, readiness
        from ..kube.operations import K3sProvider

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = KubeManager()
        provider = K3sProvider()
        report = _inventory_for_config(config, manager)
        actions = provider.plan(manager, report)
        if not config.apply or config.dry_run:
            if config.json:
                print(json.dumps({'inventory': report, 'actions': actions}, indent=2))
            else:
                for action in actions:
                    print(f'[plan] {action}')
                print('No changes made. Authenticate with sudo -v if needed; then run infer-stack kube bootstrap --apply.')
            return 0
        try:
            report = provider.apply(manager, report, version=config.version, timeout=config.timeout)
        except RuntimeError as ex:
            raise SystemExit(f'{ex}. If elevation failed, authenticate with sudo -v and retry.') from ex
        checks = readiness(report, installation=False)
        if not config.json:
            from ..kube.k3s import user_kubeconfig
            print(f'Local K3s reconciled. For inventory/install: export KUBECONFIG={user_kubeconfig()}')
        if config.json:
            print(json.dumps({'inventory': report, 'checks': checks}, indent=2))
        else:
            _print_checks(checks)
        return int(failed(checks))


class KubeModalCLI(kw.ModalCLI):
    """Inspect/setup infer-stack capabilities on any Kubernetes distribution."""

    __command__ = 'kube'
    inventory = KubeInventoryCLI
    doctor = KubeDoctorCLI
    bootstrap = KubeBootstrapCLI
    install = KubeInstallCLI
    status = KubeStatusCLI
    nodes = KubeNodesCLI
    node = KubeNodeModalCLI
    setup = KubeSetupCLI
    k3s = K3sModalCLI
