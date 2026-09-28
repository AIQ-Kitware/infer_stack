"""Capability-driven Kubernetes setup and inspection.

The important boundary here is between *what infer-stack needs* and *how a
cluster happened to provide it*. Existing GPU Operator / managed-cluster
installations are accepted when they already expose the resources and labels
we need; the NVIDIA device-plugin chart is only a convenience remediation.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ..backends.kubeai import (
    GPU_MEMORY_LABEL,
    GPU_PRODUCT_LABEL,
    node_gpus,
    resource_profiles_for_nodes,
)
from ..paths import data_root

NVIDIA_DEVICE_PLUGIN_VERSION = '0.17.1'
NVIDIA_DEVICE_PLUGIN_REPO = 'https://nvidia.github.io/k8s-device-plugin'
KUBEAI_REPO = 'https://www.kubeai.org'

RunFunc = Callable[..., str]


def default_run(
    args: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command and return stdout while preserving useful stderr errors."""
    proc = subprocess.run(
        args,
        input=input_text,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or '').strip()
        message = f'`{" ".join(args)}` failed with exit code {proc.returncode}'
        if detail:
            message += f': {detail}'
        raise RuntimeError(message)
    return proc.stdout


@dataclass
class Check:
    """One read-only capability observation."""

    name: str
    ok: bool
    detail: str = ''
    level: str = 'required'  # required | action | advisory


@dataclass
class SetupPlan:
    """What ``kube setup`` found and what an apply would reconcile."""

    checks: list[Check] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    resource_profiles: dict[str, Any] = field(default_factory=dict)
    namespace: str = 'kubeai'
    context: str = ''

    @property
    def failed(self) -> bool:
        return any(not c.ok and c.level == 'required' for c in self.checks)


class KubeManager:
    """Inspect/reconcile the small slice of Kubernetes infer-stack depends on."""

    def __init__(self, *, run: RunFunc | None = None) -> None:
        self.run = run or default_run

    def _json(self, args: list[str]) -> dict[str, Any]:
        return json.loads(self.run(args) or '{}')

    def kubectl_json(self, args: list[str]) -> dict[str, Any]:
        return self._json(['kubectl', *args])

    def nodes(self) -> list[dict[str, Any]]:
        result = self.kubectl_json(['get', 'nodes', '-o', 'json'])
        return list(result.get('items') or [])

    @staticmethod
    def node_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact node inventory for people and JSON output."""
        gpu = node_gpus(nodes)
        rows = []
        for node in nodes:
            meta = node.get('metadata') or {}
            status = node.get('status') or {}
            conditions = status.get('conditions') or []
            ready = any(
                c.get('type') == 'Ready' and c.get('status') == 'True'
                for c in conditions
            )
            name = str(meta.get('name') or '')
            facts = gpu.get(name, {})
            rows.append({
                'name': name,
                'ready': ready,
                'gpu_count': facts.get('count', 0),
                'gpu_product': facts.get('product'),
                'gpu_memory_gib': facts.get('memory_gib'),
            })
        return rows

    def command_exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def _exists(self, args: list[str]) -> bool:
        try:
            self.run(args)
        except Exception:
            return False
        return True

    def runtime_class_exists(self, name: str = 'nvidia') -> bool:
        return self._exists(
            ['kubectl', 'get', 'runtimeclass', name, '-o', 'name']
        )

    def crd_exists(self, name: str = 'models.kubeai.org') -> bool:
        return self._exists(['kubectl', 'get', 'crd', name, '-o', 'name'])

    def namespace_exists(self, namespace: str) -> bool:
        return self._exists(
            ['kubectl', 'get', 'namespace', namespace, '-o', 'name']
        )

    def kubeai_service_exists(self, namespace: str) -> bool:
        return self._exists([
            'kubectl', '-n', namespace, 'get', 'service', 'kubeai', '-o', 'name'
        ])

    def current_context(self) -> str:
        try:
            return self.run(['kubectl', 'config', 'current-context']).strip()
        except Exception:
            return ''

    def cluster_reachable(self) -> tuple[bool, str]:
        try:
            self.run(['kubectl', 'version', '--client=false', '-o', 'json'])
        except Exception as ex:  # noqa: BLE001 - diagnostic boundary
            return False, str(ex)
        return True, ''

    def helm_releases(self) -> list[dict[str, Any]]:
        value = json.loads(self.run(['helm', 'list', '-A', '-o', 'json']) or '[]')
        return value if isinstance(value, list) else []

    def kubeai_release(self, release: str = 'kubeai') -> dict[str, Any] | None:
        if not self.command_exists('helm'):
            return None
        try:
            for item in self.helm_releases():
                if item.get('name') == release:
                    return item
        except Exception:
            return None
        return None

    @staticmethod
    def resource_profiles(
        nodes: list[dict[str, Any]], *, runtime_class_name: str | None = 'nvidia',
    ) -> dict[str, Any]:
        """One KubeAI profile per discovered GPU product."""
        return resource_profiles_for_nodes(
            nodes, runtime_class_name=runtime_class_name,
        )

    @staticmethod
    def gpu_summary(nodes: list[dict[str, Any]]) -> tuple[bool, bool, str]:
        """``(resource exposed, discovery labels complete, detail)``."""
        facts = node_gpus(nodes)
        gpu_nodes = {name: f for name, f in facts.items() if f.get('count')}
        resource_ok = bool(gpu_nodes)
        labels_ok = bool(gpu_nodes) and all(
            f.get('product') and f.get('memory_gib')
            for f in gpu_nodes.values()
        )
        if not gpu_nodes:
            return False, False, 'no node exposes allocatable nvidia.com/gpu'
        total = sum(int(f.get('count') or 0) for f in gpu_nodes.values())
        missing = [
            name for name, f in gpu_nodes.items()
            if not f.get('product') or not f.get('memory_gib')
        ]
        if missing:
            return True, False, (
                f'{total} GPU(s) allocatable, but product/memory labels are '
                'missing on: ' + ', '.join(sorted(missing))
            )
        products = sorted({str(f['product']) for f in gpu_nodes.values()})
        return True, True, f'{total} GPU(s); ' + ', '.join(products)

    def _kubeai_capability(
        self, namespace: str,
    ) -> tuple[bool, bool, bool, bool]:
        crd = self.crd_exists()
        namespace_ok = self.namespace_exists(namespace)
        service_ok = namespace_ok and self.kubeai_service_exists(namespace)
        return crd and namespace_ok and service_ok, crd, namespace_ok, service_ok

    def _existing_kubeai_values(
        self, release: str, namespace: str,
    ) -> dict[str, Any]:
        found = self.kubeai_release(release)
        if found is None or str(found.get('namespace') or '') != namespace:
            return {}
        try:
            text = self.run([
                'helm', 'get', 'values', release,
                '-n', namespace, '-o', 'yaml',
            ])
        except Exception:
            return {}
        return yaml.safe_load(text or '') or {}

    @staticmethod
    def load_values_file(path: str | Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        source = Path(path).expanduser()
        if not source.is_file():
            raise RuntimeError(f'KubeAI values file does not exist: {source}')
        value = yaml.safe_load(source.read_text(encoding='utf-8')) or {}
        if not isinstance(value, dict):
            raise RuntimeError(f'KubeAI values file must contain a mapping: {source}')
        return value

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def plan_setup(
        self,
        *,
        namespace: str = 'kubeai',
        release: str = 'kubeai',
        gpu: str = 'auto',
        operator_values: dict[str, Any] | None = None,
    ) -> SetupPlan:
        """Inspect a cluster without mutating it."""
        if gpu not in {'auto', 'nvidia', 'none'}:
            raise ValueError("gpu must be 'auto', 'nvidia', or 'none'")
        plan = SetupPlan(namespace=namespace)

        if not self.command_exists('kubectl'):
            plan.checks.append(Check(
                'kubectl available', False,
                'install kubectl and select a Kubernetes context; cluster '
                'provisioning is distribution-specific',
            ))
            return plan
        plan.checks.append(Check('kubectl available', True))
        plan.context = self.current_context()

        reachable, detail = self.cluster_reachable()
        plan.checks.append(Check(
            'cluster reachable', reachable,
            detail or 'current kubeconfig context answers',
        ))
        if not reachable:
            return plan

        nodes = self.nodes()
        rows = self.node_rows(nodes)
        not_ready = [row['name'] for row in rows if not row['ready']]
        plan.checks.append(Check(
            'nodes Ready', bool(rows) and not not_ready,
            (f'{len(rows)} node(s)' if rows and not not_ready
             else ('not Ready: ' + ', '.join(not_ready)
                   if not_ready else 'cluster has no nodes')),
        ))

        resource_ok, labels_ok, gpu_detail = self.gpu_summary(nodes)
        runtime_ok = self.runtime_class_exists('nvidia')
        if gpu == 'none':
            plan.checks.append(Check(
                'NVIDIA GPU support', True, 'skipped (--gpu=none)', 'advisory'
            ))
        else:
            wants_nvidia = gpu == 'nvidia' or runtime_ok or resource_ok
            if not wants_nvidia:
                plan.checks.append(Check(
                    'NVIDIA GPU support', False,
                    'no NVIDIA RuntimeClass or GPU resource detected; configure '
                    'the NVIDIA driver/container runtime on GPU nodes, then '
                    'rerun (or use --gpu=none)',
                    'advisory',
                ))
            elif resource_ok and labels_ok:
                # Existing GPU Operator / managed-cluster integration wins.
                plan.checks.append(Check(
                    'nvidia.com/gpu allocatable', True, gpu_detail
                ))
                plan.checks.append(Check(
                    'GPU product + memory labels', True, gpu_detail
                ))
                plan.checks.append(Check(
                    'RuntimeClass nvidia', True,
                    ('available' if runtime_ok else
                     'not present; external GPU runtime accepted and generated '
                     'profiles omit runtimeClassName'),
                    'advisory',
                ))
            else:
                # Our NVIDIA chart path needs a runtime the node can actually
                # select. Missing RuntimeClass is host setup, not something a
                # Kubernetes Helm chart can repair.
                plan.checks.append(Check(
                    'RuntimeClass nvidia', runtime_ok,
                    '' if runtime_ok else (
                        'Kubernetes has no nvidia RuntimeClass; configure the '
                        'NVIDIA container runtime for this cluster/distribution '
                        'before rerunning setup'
                    ),
                ))
                level = 'action' if runtime_ok else 'required'
                plan.checks.append(Check(
                    'nvidia.com/gpu allocatable', resource_ok, gpu_detail, level
                ))
                plan.checks.append(Check(
                    'GPU product + memory labels', labels_ok,
                    gpu_detail if labels_ok else (
                        f'{gpu_detail}; GPU Feature Discovery must provide '
                        f'{GPU_PRODUCT_LABEL} and {GPU_MEMORY_LABEL}'
                    ),
                    level,
                ))
                if runtime_ok:
                    plan.actions.append(
                        'install/reconcile NVIDIA device plugin '
                        f'{NVIDIA_DEVICE_PLUGIN_VERSION} with GPU Feature Discovery'
                    )

        capability_ok, crd, namespace_ok, service_ok = \
            self._kubeai_capability(namespace)
        runtime_class = 'nvidia' if runtime_ok else None
        plan.resource_profiles = (
            {} if gpu == 'none' else
            self.resource_profiles(nodes, runtime_class_name=runtime_class)
        )

        helm_ok = self.command_exists('helm')
        if not helm_ok:
            needs_helm = bool(plan.actions) or not capability_ok
            if needs_helm:
                helm_detail = (
                    'install Helm before `infer-stack kube setup --apply`'
                )
                helm_level = 'required'
            else:
                helm_detail = (
                    'not installed; existing KubeAI/GPU capabilities are '
                    'externally managed'
                )
                helm_level = 'advisory'
            plan.checks.append(Check(
                'helm available', not needs_helm, helm_detail, helm_level,
            ))
            plan.checks.append(Check(
                'KubeAI Model CRD', crd,
                '' if crd else 'models.kubeai.org is not installed',
            ))
            plan.checks.append(Check(
                f'namespace {namespace!r}', namespace_ok,
                '' if namespace_ok else 'configured KubeAI namespace is missing',
            ))
            plan.checks.append(Check(
                f'KubeAI service in namespace {namespace!r}', service_ok,
                '' if service_ok else 'service/kubeai is not available',
            ))
            return plan
        plan.checks.append(Check('helm available', True))

        found_release = self.kubeai_release(release)
        managed_here = False
        if found_release:
            found_ns = str(found_release.get('namespace') or '')
            same_ns = found_ns == namespace
            managed_here = same_ns
            plan.checks.append(Check(
                f'KubeAI Helm release {release!r}', same_ns,
                (f'installed in namespace {found_ns}' if same_ns else
                 f'installed in namespace {found_ns}; use --namespace={found_ns} '
                 'or choose another release'),
            ))
            if same_ns and not capability_ok:
                plan.actions.append(
                    f'reconcile KubeAI release {release!r} in namespace {namespace}'
                )
        elif capability_ok:
            plan.checks.append(Check(
                f'KubeAI Helm release {release!r}', True,
                'not owned by this Helm release; existing KubeAI capability is '
                'accepted as externally managed',
                'advisory',
            ))
        else:
            managed_here = True
            plan.checks.append(Check(
                f'KubeAI Helm release {release!r}', False,
                f'not installed in namespace {namespace}', 'action',
            ))
            plan.actions.append(
                f'install KubeAI release {release!r} in namespace {namespace}'
            )

        component_level = 'action' if managed_here else 'required'
        plan.checks.append(Check(
            'KubeAI Model CRD', crd,
            '' if crd else 'models.kubeai.org will be installed with KubeAI',
            component_level if not crd else 'required',
        ))
        plan.checks.append(Check(
            f'namespace {namespace!r}', namespace_ok,
            '' if namespace_ok else 'will be created with KubeAI',
            component_level if not namespace_ok else 'required',
        ))
        plan.checks.append(Check(
            f'KubeAI service in namespace {namespace!r}', service_ok,
            '' if service_ok else 'service/kubeai will be installed with KubeAI',
            component_level if not service_ok else 'required',
        ))

        if plan.resource_profiles and managed_here:
            effective_values = self._deep_merge(
                self._existing_kubeai_values(release, namespace),
                operator_values or {},
            )
            existing_profiles = dict(effective_values.get('resourceProfiles') or {})
            missing = sorted(set(plan.resource_profiles) - set(existing_profiles))
            if missing:
                plan.actions.append(
                    f'reconcile {len(missing)} discovered GPU resource profile(s) '
                    f'into KubeAI values: {", ".join(missing)}'
                )
        return plan

    def install_nvidia_device_plugin(
        self,
        *,
        version: str = NVIDIA_DEVICE_PLUGIN_VERSION,
    ) -> None:
        self.run([
            'helm', 'repo', 'add', 'nvdp', NVIDIA_DEVICE_PLUGIN_REPO,
            '--force-update',
        ])
        self.run(['helm', 'repo', 'update'])
        self.run([
            'helm', 'upgrade', '--install', 'nvdp',
            'nvdp/nvidia-device-plugin',
            '--version', version,
            '--namespace', 'nvidia-device-plugin', '--create-namespace',
            '--set', 'gfd.enabled=true',
            '--set', 'runtimeClassName=nvidia',
            '--wait',
        ])

    @staticmethod
    def _split_secret_values(
        values: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Keep Helm ``secrets`` out of the persistent generated values file."""
        public_values = copy.deepcopy(values or {})
        secrets = public_values.pop('secrets', None)
        secret_values = {'secrets': secrets} if secrets else {}
        return public_values, secret_values

    @staticmethod
    def _merge_profiles(
        values: dict[str, Any], generated: dict[str, Any],
    ) -> dict[str, Any]:
        values = dict(values or {})
        existing = dict(values.get('resourceProfiles') or {})
        # Existing named profiles are operator authority. Add only missing
        # discovered products rather than rewriting customized scheduling rules.
        merged = dict(generated)
        merged.update(existing)
        values['resourceProfiles'] = merged
        return values

    @staticmethod
    def _release_chart_version(
        release: dict[str, Any] | None,
    ) -> str | None:
        chart = str((release or {}).get('chart') or '')
        prefix = 'kubeai-'
        return chart[len(prefix):] if chart.startswith(prefix) else None

    def install_kubeai(
        self,
        *,
        namespace: str,
        release: str,
        resource_profiles: dict[str, Any],
        version: str | None = None,
        operator_values: dict[str, Any] | None = None,
    ) -> Path:
        self.run([
            'helm', 'repo', 'add', 'kubeai', KUBEAI_REPO, '--force-update'
        ])
        self.run(['helm', 'repo', 'update'])
        base_values = self._deep_merge(
            self._existing_kubeai_values(release, namespace),
            operator_values or {},
        )
        public_values, secret_values = self._split_secret_values(base_values)
        values = self._merge_profiles(public_values, resource_profiles)
        values_path = data_root() / 'generated' / 'kube' / 'kubeai-values.yaml'
        values_path.parent.mkdir(parents=True, exist_ok=True)
        values_path.write_text(
            yaml.safe_dump(values, sort_keys=False), encoding='utf-8'
        )

        cmd = [
            'helm', 'upgrade', '--install', release, 'kubeai/kubeai',
            '-n', namespace, '--create-namespace',
            '-f', str(values_path), '--wait',
        ]
        if version:
            cmd.extend(['--version', version])

        # Keep chart secrets out of persistent generated YAML and process argv.
        # Existing/operator secret values are preserved in-memory; HF_TOKEN, if
        # exported, deliberately overrides only the Hugging Face token.
        token = os.environ.get('HF_TOKEN', '').strip()
        if token:
            huggingface = secret_values.setdefault('secrets', {}).setdefault(
                'huggingface', {}
            )
            huggingface['token'] = token
        secret_path: str | None = None
        try:
            if secret_values:
                with tempfile.NamedTemporaryFile(
                    'w', prefix='infer-stack-kubeai-secret-',
                    suffix='.yaml', delete=False,
                ) as file:
                    os.chmod(file.name, 0o600)
                    yaml.safe_dump(secret_values, file, sort_keys=False)
                    secret_path = file.name
                cmd.extend(['-f', secret_path])
            self.run(cmd)
        finally:
            if secret_path:
                Path(secret_path).unlink(missing_ok=True)
        return values_path

    def apply_setup(
        self,
        *,
        namespace: str = 'kubeai',
        release: str = 'kubeai',
        gpu: str = 'auto',
        nvidia_plugin_version: str = NVIDIA_DEVICE_PLUGIN_VERSION,
        kubeai_version: str | None = None,
        operator_values: dict[str, Any] | None = None,
    ) -> tuple[SetupPlan, Path | None]:
        """Reconcile managed prerequisites, then return the fresh plan."""
        plan = self.plan_setup(
            namespace=namespace, release=release, gpu=gpu,
            operator_values=operator_values,
        )
        if any(
            c.name == 'cluster reachable' and not c.ok for c in plan.checks
        ):
            raise RuntimeError('Kubernetes cluster is not reachable')
        if plan.failed:
            blockers = '; '.join(
                c.name for c in plan.checks
                if not c.ok and c.level == 'required'
            )
            raise RuntimeError(f'cannot apply while required checks fail: {blockers}')
        if not self.command_exists('helm'):
            # Reaching here means every needed capability was accepted as
            # externally managed, so there is nothing for apply to mutate.
            return plan, None

        if any(
            action.startswith('install/reconcile NVIDIA')
            for action in plan.actions
        ):
            self.install_nvidia_device_plugin(
                version=nvidia_plugin_version,
            )

        # Re-read after the DaemonSet/GFD rollout: those labels are the source
        # of truth for generated profiles.
        nodes = self.nodes()
        if any(
            action.startswith('install/reconcile NVIDIA')
            for action in plan.actions
        ):
            resource_ok, labels_ok, detail = self.gpu_summary(nodes)
            if not (resource_ok and labels_ok):
                raise RuntimeError(
                    'NVIDIA device-plugin/GFD reconciliation completed but GPU '
                    f'capability is still incomplete: {detail}'
                )
        runtime_ok = self.runtime_class_exists('nvidia')
        profiles = (
            {} if gpu == 'none' else self.resource_profiles(
                nodes,
                runtime_class_name='nvidia' if runtime_ok else None,
            )
        )
        found = self.kubeai_release(release)
        if found and str(found.get('namespace') or '') != namespace:
            raise RuntimeError(
                f'KubeAI release {release!r} already exists in namespace '
                f'{found.get("namespace")!r}; use that namespace or another '
                'release name'
            )

        values_path: Path | None = None
        kubeai_actions = [
            action for action in plan.actions
            if 'KubeAI release' in action or 'resource profile' in action
        ]
        if kubeai_actions:
            version = kubeai_version or self._release_chart_version(found)
            values_path = self.install_kubeai(
                namespace=namespace,
                release=release,
                resource_profiles=profiles,
                version=version,
                operator_values=operator_values,
            )

        fresh = self.plan_setup(
            namespace=namespace, release=release, gpu=gpu,
            operator_values=operator_values,
        )
        return fresh, values_path
