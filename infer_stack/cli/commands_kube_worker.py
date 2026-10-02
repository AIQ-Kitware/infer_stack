"""Reviewable named-worker onboarding and acceptance through kwconf."""
from __future__ import annotations

import json
import subprocess
import uuid

import kwconf as kw

from ..kube.manage import KubeManager
from ..kube.worker import (
    DEFAULT_MODEL,
    acceptance,
    cleanup,
    onboard,
    onboarding_plan,
    plan,
    private_file,
    scoped_manager,
)
from .context import _apply_path_overrides
from .options import _PathOverridesMixin


def _show_result(result, machine=False):
    if machine:
        print(json.dumps(result, indent=2))
        return
    print(f"worker: {result.get('node') or result.get('facts', {}).get('node')}")
    for key in ('context', 'namespace', 'server', 'version', 'profile_name', 'profile', 'run_id'):
        if key in result and result[key] is not None:
            print(f'  {key}: {result[key]}')
    facts = result.get('facts') or {}
    if facts:
        print(f"  Ready: {facts['ready']}; allocatable GPUs: {facts['gpu_count']}; GFD: {facts['product']}, {facts['memory_mib']} MiB")
    for message in facts.get('errors') or []:
        print(f'[FAIL] {message}')
    for device in result.get('devices') or []:
        print(f"  GPU {device['uuid']}: {device['product']}, {device['memory_mib'] / 1024:g} GiB")
    for action in result.get('actions') or []:
        print(f'[plan] {action}')
    for message in result.get('limitations') or []:
        print(f'[info] {message}')
    for device in result.get('serving_devices') or []:
        print(f"  serving GPU: {device['uuid']} ({device['product']})")
    if result.get('generation_verified'):
        print('[ok] GPU runtime, exact node placement, one-GPU serving replica and real generation')
        print('Temporary test resources removed; reusable node profile retained.')
    if 'cleanup_run' in result:
        print(f"Cleanup {result['cleanup_run']}: {'complete' if result['applied'] else 'planned'}")


class KubeNodeTestCLI(_PathOverridesMixin):
    """Test this exact worker's GPUs, model placement and real generation.

    Plans by default. Apply briefly reserves ALL expected GPUs for a device
    query, then ONE GPU for a temporary serving replica. Leaves a reusable
    node profile, deletes only this test's resources, and preserves leases.
    NVIDIA drivers/container toolkit must already be installed on the worker.
    """

    __command__ = 'test'
    node = kw.Value(None, type=str, position=1, required=True, help='Exact Kubernetes node name.')
    expected_gpus = kw.Value(None, type=int, help='Required exact count, e.g. 1 for namek, 2 for yardrat, 4 for aiq-gpu2.')
    kubeconfig = kw.Value(None, type=str, help='Explicit admin kubeconfig; otherwise the selected context.')
    namespace = kw.Value('kubeai', type=str, help='Existing KubeAI namespace (use default on aiq-gpu).')
    release = kw.Value('kubeai', type=str, help='Existing KubeAI Helm release.')
    resource_profile = kw.Value(None, type=str, help='Optional installed one-GPU base profile to constrain to this node.')
    base_url = kw.Value(None, type=str, help='Direct KubeAI OpenAI URL; default: temporary loopback service forwarding.')
    model = kw.Value(DEFAULT_MODEL, type=str, help='Hugging Face model for real generation.')
    timeout = kw.Value(900, type=int, help='GPU/startup wait budget in seconds (first image pull can take minutes).')
    run_id = kw.Value(None, type=str, help='Acceptance run ID; generated unless supplied. Required for explicit cleanup.')
    cleanup = kw.Value(False, isflag=True, help='Remove only a named interrupted test run; needs --run-id --apply.')
    apply = kw.Value(False, isflag=True, alias=['yes'], help='Authorize the printed test/installation changes.')
    dry_run = kw.Value(False, isflag=True, alias=['plan'], help='Only display the plan, even with --apply.')
    json = kw.Value(False, isflag=True, help='Emit structured plan/result; progress goes to stderr.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import sys
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        manager = scoped_manager(KubeManager(), config.kubeconfig)
        def progress(message):
            print(message, file=sys.stderr if config.json else sys.stdout, flush=True)
        try:
            if config.cleanup:
                if not config.run_id:
                    raise RuntimeError('--cleanup requires --run-id; no broad cleanup is performed')
                if config.apply and not config.dry_run:
                    cleanup(manager, node=config.node, namespace=config.namespace, run_id=config.run_id)
                result = {'cleanup_run': config.run_id, 'node': config.node,
                          'applied': bool(config.apply and not config.dry_run)}
            else:
                if config.expected_gpus is None:
                    raise RuntimeError('--expected-gpus is required; cluster totals cannot prove this worker is configured')
                result = plan(manager, node=config.node, namespace=config.namespace,
                              expected_gpus=config.expected_gpus, resource_profile=config.resource_profile)
                if config.apply and not config.dry_run:
                    result = acceptance(manager, node=config.node, namespace=config.namespace,
                                        release=config.release, expected_gpus=config.expected_gpus,
                                        base_url=config.base_url, kubeconfig=config.kubeconfig,
                                        run_id=config.run_id or uuid.uuid4().hex[:12],
                                        timeout=config.timeout, model=config.model,
                                        resource_profile=config.resource_profile, progress=progress)
            _show_result(result, config.json)
            if not config.apply or config.dry_run:
                print('No changes made. Review the target node and rerun with --apply.', file=sys.stderr if config.json else sys.stdout)
            facts = result.get('facts')
            return int(isinstance(facts, dict) and bool(facts.get('errors')))
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as ex:
            raise SystemExit(f'worker test: {ex}') from ex


class K3sOnboardCLI(KubeNodeTestCLI):
    """Join this GPU host and test it through explicit control-plane access.

    Plans by default. Checks host driver/runtime and admin target, infers the
    server's K3s version, joins/resumes membership, then verifies every local
    device plus node-specific model placement/generation. No kubeconfig or
    catalog/ledger is replaced. Driver/toolkit installation is a prerequisite.
    """

    __command__ = 'onboard'
    kubeconfig = kw.Value(None, type=str, required=True, help='Private admin kubeconfig selecting the SAME --server; never inferred from stale context.')
    server = kw.Value(None, type=str, required=True, help='Existing K3s server URL, e.g. https://aiq-gpu:6443.')
    token_file = kw.Value(None, type=str, required=True, help='Private file containing the join token; never pass token text.')
    version = kw.Value(None, type=str, help='Exact K3s version; otherwise infer from this cluster control plane.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import sys
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        local = KubeManager()
        cluster = scoped_manager(local, config.kubeconfig)
        def progress(message):
            print(message, file=sys.stderr if config.json else sys.stdout, flush=True)
        try:
            if config.cleanup:
                raise RuntimeError('Use infer-stack kube node test NODE --cleanup --run-id=... --apply for interrupted acceptance resources')
            private_file(config.kubeconfig, 'Admin kubeconfig')
            private_file(config.token_file, 'Join token')
            prepared = onboarding_plan(local, cluster, server=config.server, node=config.node, version=config.version)
            if config.expected_gpus is not None and config.expected_gpus != prepared['expected_gpus']:
                raise RuntimeError('--expected-gpus differs from locally detected hardware')
            result = prepared
            if config.apply and not config.dry_run:
                result = onboard(local, cluster, prepared, token_file=config.token_file,
                                 namespace=config.namespace, release=config.release,
                                 base_url=config.base_url, kubeconfig=config.kubeconfig,
                                 run_id=config.run_id or uuid.uuid4().hex[:12], timeout=config.timeout,
                                 model=config.model, resource_profile=config.resource_profile,
                                 progress=progress)
            _show_result(result, config.json)
            if not config.apply or config.dry_run:
                print('No changes made. Authenticate with sudo -v, review this plan, then rerun with --apply.', file=sys.stderr if config.json else sys.stdout)
            return 0
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as ex:
            raise SystemExit(f'worker onboarding: {ex}') from ex


class K3sExportCLI(kw.Config):
    """Export private token/admin kubeconfig files for reviewed worker onboarding.

    Run on the K3s server. Plans by default; --apply creates a mode-0700 bundle
    with mode-0600 files and rewrites only its copy's server URL. Existing owned
    bundles can refresh; unrelated directories/clusters are refused.
    """

    __command__ = 'export'
    server = kw.Value(None, type=str, required=True, help='Reachable HTTPS server URL as workers should use it.')
    directory = kw.Value(None, type=str, required=True, help='Private destination directory for token/kubeconfig/manifest.')
    apply = kw.Value(False, isflag=True, alias=['yes'], help='Authorize writing/refreshing the private join bundle.')
    dry_run = kw.Value(False, isflag=True, alias=['plan'], help='Only display paths; never read/write credentials.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from pathlib import Path

        from ..kube.k3s import export_join_info
        config = cls.cli(argv=argv, data=kwargs)
        try:
            directory = Path(config.directory).expanduser()
            if config.apply and not config.dry_run:
                result = export_join_info(server=config.server, directory=directory)
                print(json.dumps(result, indent=2))
                print('Private join bundle exported. Transfer it securely to the worker; then run kube k3s onboard NAME using these files. Keep admin credentials private.')
            else:
                print(f'[plan] Export private join bundle for {config.server} into {directory} (directory 0700, files 0600).')
                print('[plan] Files: token, kubeconfig.yaml (reachable server URL), manifest.json. Original kubeconfigs remain unchanged.')
                print('No credentials read or written. Authenticate with sudo -v, then rerun with --apply.')
            return 0
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as ex:
            raise SystemExit(f'join export: {ex}') from ex
