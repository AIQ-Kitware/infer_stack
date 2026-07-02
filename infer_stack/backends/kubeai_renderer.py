from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import KUBEAI_GENERATED_SUBDIR, normalized_output
from ..diff_prompt import confirm_writes
from ..profile_runtime import vllm_args


def _resource_profile_values(plan: dict[str, Any]) -> dict[str, Any]:
    values_doc = plan.get('deployment', {}).get('resource_profiles_values', {})
    return values_doc or {'resourceProfiles': {}}


def _kubeai_resource_profile(service: dict[str, Any]) -> str:
    profile = str(service.get('resource_profile', ''))
    if not profile or ':' in profile:
        return profile
    gpu_count = max(
        1,
        len(service.get('gpu_indices', [])),
        int(service.get('tensor_parallel_size', 1) or 1)
        * int(service.get('pipeline_parallel_size', 1) or 1)
        * int(service.get('data_parallel_size', 1) or 1),
    )
    return f'{profile}:{gpu_count}'


def _kubeai_args(service: dict[str, Any]) -> list[str]:
    kubeai_service = dict(service)
    kubeai_service['served_model_name'] = service['profile_public_name']
    return vllm_args(kubeai_service)


def _model_doc(service: dict[str, Any]) -> dict[str, Any]:
    doc = {
        'apiVersion': 'kubeai.org/v1',
        'kind': 'Model',
        'metadata': {
            'name': service['kubernetes_name'],
            'annotations': {
                'infer-stack/profile-name': service['profile_name'],
                'infer-stack/public-name': service['profile_public_name'],
                'infer-stack/logical-model-name': service['logical_model_name'],
                'infer-stack/protocol-mode': service['protocol_mode'],
            },
        },
        'spec': {
            'features': service.get('features', ['TextGeneration']),
            'url': service['model_url'],
            'engine': service.get('engine', 'VLLM'),
            'resourceProfile': _kubeai_resource_profile(service),
            'minReplicas': int(service.get('min_replicas', 0)),
            'maxReplicas': int(service.get('max_replicas', 1)),
            'args': _kubeai_args(service),
        },
    }
    if service.get('priority_class_name'):
        doc['spec']['priorityClassName'] = service['priority_class_name']
    return doc


def render_kubeai_artifacts(
    lock_data: dict, *, assume_yes: bool = True
) -> None:
    """Render the KubeAI backend artifacts.

    When ``assume_yes`` is False, all rendered YAML files are diffed against
    their on-disk versions and the user is prompted via Rich before any file
    is written.
    """
    deployment = lock_data.get('deployment', {})
    cluster = deployment.get('cluster', {})
    namespace = cluster.get('namespace', 'kubeai')
    output_root = Path(
        normalized_output(deployment.get('output'))['generated_dir']
    )
    generated = output_root / KUBEAI_GENERATED_SUBDIR
    generated.mkdir(parents=True, exist_ok=True)

    namespace_doc = {
        'apiVersion': 'v1',
        'kind': 'Namespace',
        'metadata': {'name': namespace},
    }
    namespace_text = yaml.safe_dump(namespace_doc, sort_keys=False)

    values_doc = _resource_profile_values(lock_data)
    values_text = yaml.safe_dump(values_doc, sort_keys=False)

    model_docs = [
        _model_doc(service)
        for service in (
            deployment.get('providers', {}).get('vllm', {}).get('runtimes', {})
            or {}
        ).values()
    ]
    model_text = '---\n'.join(
        yaml.safe_dump(doc, sort_keys=False) for doc in model_docs
    )

    ingress = cluster.get('ingress', {}) or {}
    ingress_path = generated / 'ingress.yaml'
    ingress_text: str | None = None
    if ingress.get('enabled'):
        path_prefix = ingress.get('path_prefix', '/') or '/'
        ingress_doc: dict[str, Any] = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': cluster.get('service_name', 'kubeai'),
                'namespace': namespace,
            },
            'spec': {
                'ingressClassName': ingress.get('class_name', 'traefik'),
                'rules': [
                    {
                        'http': {
                            'paths': [
                                {
                                    'path': path_prefix,
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': cluster.get(
                                                'service_name', 'kubeai'
                                            ),
                                            'port': {'number': 80},
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
        }
        if ingress.get('host'):
            ingress_doc['spec']['rules'][0]['host'] = ingress['host']
        if ingress.get('tls_secret_name') and ingress.get('host'):
            ingress_doc['spec']['tls'] = [
                {
                    'hosts': [ingress['host']],
                    'secretName': ingress['tls_secret_name'],
                }
            ]
        ingress_text = yaml.safe_dump(ingress_doc, sort_keys=False)

    readme = f"""# Generated KubeAI artifacts

Namespace: `{namespace}`
Release: `{cluster.get('kubeai_release_name', 'kubeai')}`
Chart: `{cluster.get('kubeai_chart', 'kubeai/kubeai')}`

Files:
- `namespace.yaml`: namespace to apply before the chart and models
- `kubeai-values.yaml`: custom resource profiles for the KubeAI chart
- `models.yaml`: KubeAI `Model` objects derived intentionally from the selected serving profile(s)
- `ingress.yaml`: optional ingress for one stable hostname

Typical flow:

```bash
kubectl apply -f {generated}/namespace.yaml
helm repo add kubeai https://www.kubeai.org --force-update
helm repo update
helm upgrade --install {cluster.get('kubeai_release_name', 'kubeai')} {cluster.get('kubeai_chart', 'kubeai/kubeai')} \
  -n {namespace} --create-namespace \
  -f {generated}/kubeai-values.yaml \
  --wait
kubectl apply -f {generated}/models.yaml
```
"""

    namespace_path = generated / 'namespace.yaml'
    values_path = generated / 'kubeai-values.yaml'
    models_path = generated / 'models.yaml'
    readme_path = generated / 'README.md'

    planned: dict[Path, str] = {
        namespace_path: namespace_text,
        values_path: values_text,
        models_path: model_text,
        readme_path: readme,
    }
    if ingress_text is not None:
        planned[ingress_path] = ingress_text

    if not confirm_writes(
        planned, assume_yes=assume_yes, title='Pending KubeAI render'
    ):
        raise SystemExit('Aborted by user; no files were written.')

    namespace_path.write_text(namespace_text, encoding='utf-8')
    values_path.write_text(values_text, encoding='utf-8')
    models_path.write_text(model_text, encoding='utf-8')
    readme_path.write_text(readme, encoding='utf-8')
    if ingress_text is not None:
        ingress_path.write_text(ingress_text, encoding='utf-8')
    elif ingress_path.exists():
        ingress_path.unlink()
