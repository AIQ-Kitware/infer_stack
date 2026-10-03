"""The LiteLLM gateway inside the cluster, for the kubeai backend.

The default front door for a cluster is a one-service Compose project on the
host running infer-stack, so that host is in every request's path. This is
the other placement: the same gateway (the same image, config, key and route
registry, through :class:`~infer_stack.leasing.gateway.Gateway`) rendered as
Kubernetes objects in the KubeAI namespace and reached on a NodePort of any
node, or an ingress URL.

It answers the calls the kubeai backend makes on its gateway (keys, routes,
preview / converge / apply, down, instances), so the backend does not know
which placement it has. Static routes only: dynamic routing and Open WebUI
need Postgres and a UI in the cluster, which stay with the host placement.

Objects, all named ``infer-stack-gateway``: a ConfigMap (the LiteLLM config),
a Secret (the master key, applied from the state dir's ``.env`` and never
shown in a diff), a Deployment whose pod template carries the config's and
the key's hashes (so either change rolls the pods), and a NodePort Service.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import PINNED_IMAGES
from ..leasing.backend import ConvergeScaffold
from ..leasing.routes import GatewayRoute, route_table
from ..leasing.gateway import (
    API_KEY_ENV,
    LITELLM_CONTAINER_PORT,
    Gateway,
    remembered_rows,
    render_front_door,
    route_key_envs,
)

NAME = 'infer-stack-gateway'
#: The pod label the Service selects and ``instances`` lists by.
APP_LABEL = 'app.kubernetes.io/name'
MANIFESTS_FILENAME = 'gateway.yaml'
SECRET_FILENAME = 'gateway-secret.yaml'      # 0600, like the .env it copies
STATE_FILENAME = 'gateway-state.json'
DEFAULT_NODE_PORT = 30442


class ClusterGateway(ConvergeScaffold):
    """The LiteLLM gateway as a Deployment + NodePort Service in the cluster."""

    _approve_title = 'infer-stack will update the in-cluster gateway'
    _state_noun = 'gateway manifests'
    #: Where this gateway runs, for the kubeai backend's routing decisions.
    in_cluster = True
    #: The settings a host gateway has and this one does not (see module doc).
    litellm = True
    ui = False
    reverse_proxy = False
    dynamic_routing = False

    def __init__(
        self,
        *,
        state_dir: str | Path,
        namespace: str,
        run: Callable[[list[str]], str],
        node_port: int = DEFAULT_NODE_PORT,
        url: str | None = None,
        images: dict[str, str] | None = None,
        http: Any = None,
        assume_yes: bool = True,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.run = run
        self.node_port = int(node_port)
        self.url = url
        self.images = {**PINNED_IMAGES, **(images or {})}
        self.assume_yes = assume_yes
        self.gateway = Gateway(self.state_dir, ports={'litellm': self.node_port},
                               litellm=True, ui=False, http=http,
                               base_url=self.base_url)
        self.catalog = None
        #: Render inputs from the kubeai backend (see ComposeBackend's).
        self.extra_routes: list[GatewayRoute] = []
        self.extra_dynamic_routes: list[GatewayRoute] = []
        self._node_address: str | None = None

    # -- where it is ---------------------------------------------------------

    @property
    def http(self) -> Any:
        return self.gateway.http

    @http.setter
    def http(self, value: Any) -> None:
        self.gateway.http = value

    @property
    def manifests_file(self) -> Path:
        return self.state_dir / MANIFESTS_FILENAME

    rendered_file = manifests_file

    @property
    def _state_file(self) -> Path:
        return self.state_dir / STATE_FILENAME

    @property
    def litellm_port(self) -> int:
        return self.node_port

    def _kubectl(self, args: list[str]) -> str:
        return self.run(['kubectl', '-n', self.namespace, *args])

    def base_url(self) -> str:
        """Where clients reach it: the ``url`` setting, else a node's NodePort.

        A NodePort answers on every node, so any node's address works; the
        first node's InternalIP is used.
        """
        if self.url:
            return self.url.rstrip('/')
        if self._node_address is None:
            nodes = json.loads(self._kubectl(['get', 'nodes', '-o', 'json']) or '{}')
            addresses = [a['address'] for n in nodes.get('items') or []
                         for a in (n.get('status') or {}).get('addresses') or []
                         if a.get('type') == 'InternalIP']
            self._node_address = addresses[0] if addresses else '127.0.0.1'
        return f'http://{self._node_address}:{self.node_port}'

    def compose_project(self):
        """No Compose project: this gateway is Kubernetes objects."""
        return None

    def front_door(self):
        return self

    # -- keys and routes: the one Gateway ------------------------------------

    def master_key(self) -> str:
        return self.gateway.master_key()

    def rotate_master_key(self):
        return self.gateway.rotate_master_key()

    def restore_env(self, values) -> None:
        self.gateway.restore_env(values)

    def gateway_accepts(self, key: str, *, wait: float = 0.0):
        return self.gateway.gateway_accepts(key, wait=wait)

    @property
    def env_path(self) -> Path:
        return self.gateway.env_path

    def require_route_keys(self, routes) -> None:
        self.gateway.require_route_keys(routes)

    def registry_routes(self):
        return self.gateway.registry_routes()

    def route_entries(self) -> dict[str, dict[str, Any]]:
        return self.gateway.route_entries()

    def replace_route_entries(self, entries: dict[str, dict[str, Any]]) -> None:
        self.gateway.replace_route_entries(entries)

    def urls(self) -> tuple[str | None, str | None]:
        return self.gateway.urls()

    def connection_info(self):
        return self.gateway.connection_info()

    def catalog_routes(self, catalog) -> list[GatewayRoute]:
        return []              # the kubeai backend supplies its own routes

    def _secret_values(self, routes: list[GatewayRoute]) -> dict[str, str]:
        """What the Secret holds: the master key, and each routed external
        key that has a value (never written into the manifests)."""
        env = self.gateway.managed_env()
        keys = {n: env[n] for n in route_key_envs(routes) if env.get(n)}
        return {API_KEY_ENV: self.master_key(), **keys}

    def static_routes(self) -> list[GatewayRoute]:
        """The static route table: the registry under the backend's routes."""
        return route_table(self.gateway.registry_routes(), self.extra_routes)

    # -- render / apply -----------------------------------------------------

    def _render_documents(self) -> dict[Path, str]:
        """The planned files, in memory: the one render."""
        front = render_front_door(
            self.static_routes(), vllm_v1_urls=[], ollama_native_urls=[],
            images=self.images, state={}, litellm=True,
            litellm_port=LITELLM_CONTAINER_PORT, litellm_master_key=None,
            litellm_salt_key=False, ui=False, ui_port=0, reverse_proxy=False,
            reverse_proxy_port=0, reverse_proxy_config=None, aux_dir=self.state_dir,
            dynamic_routing=False,
        )
        config = front.litellm_config or ''
        routes = self.static_routes()
        self.gateway.require_route_keys(routes)
        key_hash = secret_hash(self._secret_values(routes))
        docs = gateway_manifests(
            namespace=self.namespace, image=self.images['litellm'], config=config,
            key_hash=key_hash, node_port=self.node_port)
        text = '---\n'.join(yaml.safe_dump(d, sort_keys=False) for d in docs)
        return {self.manifests_file: text}

    def preview(self, desired=(), placement=None, *, approve: bool = False):
        """Render without writing; with ``approve``, show the diff now."""
        with self.gateway.staging_secrets():
            planned = self._render_documents()
        self._preview_approval(planned, approve=approve)
        return None, None

    def converge(self, desired=(), *, apply: bool = True, placement=None):
        """Render the gateway's objects, and write them after approval."""
        with self._converge_lock():
            planned = self._render_documents()
            self.last_planned_digest = self._planned_digest(planned)
            self._approve_changes(planned)
            self.gateway.remember(remembered_rows(
                [], {}, defined=set(getattr(self.catalog, 'endpoints', None) or {}),
                extra=self.extra_routes))
            for path, text in planned.items():
                self._atomic_write(path, text)
        if apply:
            self.apply()

    def apply(self) -> None:
        """Apply the key's Secret, then the objects, and wait for the rollout."""
        import os

        from .._log import logger

        if not self.manifests_file.exists():
            return
        secret = yaml.safe_dump(secret_manifest(
            self.namespace, self._secret_values(self.static_routes())))
        path = self.state_dir / SECRET_FILENAME
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as handle:
            handle.write(secret)
        logger.info('kubectl apply (the in-cluster gateway)')
        self._kubectl(['apply', '-f', str(path)])
        self._kubectl(['apply', '-f', str(self.manifests_file)])
        self._kubectl(['rollout', 'status', f'deployment/{NAME}', '--timeout=300s'])

    def settle_snapshot(self) -> None:
        """Nothing local: this gateway is Kubernetes objects, applied declaratively."""
        return None

    def retire_routes(self) -> bool:
        """Nothing to retire: this gateway's routes are static and point at
        KubeAI's one upstream, which no Model's removal takes away."""
        return True

    def down(self) -> None:
        self._kubectl(['delete', 'deployment,service,configmap,secret', NAME,
                       '--ignore-not-found'])

    def instances(self):
        from ..leasing.instances import KUBERNETES, from_residency
        from ..leasing.residency import ResidencyUnknown, residency_from_pods

        try:
            raw = self._kubectl(['get', 'pods', '-l', f'{APP_LABEL}={NAME}', '-o', 'json'])
        except Exception as ex:  # noqa: BLE001 - any failure is "unknown"
            raise ResidencyUnknown(f'kubectl get pods failed: {ex}') from ex
        return from_residency(residency_from_pods(raw), runtime=KUBERNETES,
                              namespace=self.namespace)

    def doctor_check(self) -> tuple[str, bool, str]:
        """``(check, ok, detail)``: does the gateway answer with the managed key?"""
        base = self.base_url()
        accepted = self.gateway_accepts(self.master_key())
        if accepted:
            return (f'in-cluster gateway at {base}/v1', True, '')
        return (f'in-cluster gateway at {base}/v1', False,
                'not answering; `infer-stack apply` renders and starts it, and '
                '`kubectl -n <namespace> get pods -l '
                f'{APP_LABEL}={NAME}` shows why it is not up')

    # -- the recovery profile ------------------------------------------------

    def render_profile(self) -> dict[str, Any]:
        return {'placement': 'cluster', 'node_port': self.node_port, 'url': self.url,
                'image': self.images['litellm']}

    def use_profile(self, profile: dict[str, Any]) -> None:
        if profile.get('placement') != 'cluster':
            from ..leasing.profile import ProfileMismatch

            raise ProfileMismatch(
                'the active recovery snapshot has the gateway on this host; '
                '`infer-stack stack down` it before moving it into the cluster')
        self.node_port = int(profile.get('node_port') or self.node_port)
        self.url = profile.get('url') or self.url
        self.images['litellm'] = profile.get('image') or self.images['litellm']


def secret_manifest(namespace: str, values: dict[str, str]) -> dict[str, Any]:
    """The gateway's secrets as a Secret (applied, never rendered into a diff):
    the master key, and the external keys its routes send by name."""
    return {'apiVersion': 'v1', 'kind': 'Secret',
            'metadata': {'name': NAME, 'namespace': namespace},
            'type': 'Opaque', 'stringData': dict(values)}


def secret_hash(values: dict[str, str]) -> str:
    """The pod-template annotation that rolls the gateway when a Secret value
    changes. With only the master key it is that key's hash, as before.

    >>> secret_hash({API_KEY_ENV: 'k'}) == hashlib.sha256(b'k').hexdigest()[:12]
    True
    >>> secret_hash({API_KEY_ENV: 'k', 'X': '1'}) != secret_hash({API_KEY_ENV: 'k', 'X': '2'})
    True
    """
    extra = {k: v for k, v in values.items() if k != API_KEY_ENV}
    material = values.get(API_KEY_ENV, '')
    if extra:
        material += '\x00' + json.dumps(extra, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def gateway_manifests(*, namespace: str, image: str, config: str, key_hash: str,
                      node_port: int) -> list[dict[str, Any]]:
    """ConfigMap, Deployment and NodePort Service for the in-cluster gateway.

    >>> docs = gateway_manifests(namespace='kubeai', image='litellm:x',
    ...                          config='model_list: []\\n', key_hash='abc', node_port=30442)
    >>> [d['kind'] for d in docs]
    ['ConfigMap', 'Deployment', 'Service']
    >>> docs[2]['spec']['ports'][0]['nodePort']
    30442
    """
    config_hash = hashlib.sha256(config.encode()).hexdigest()[:12]
    labels = {APP_LABEL: NAME}
    meta = {'name': NAME, 'namespace': namespace, 'labels': labels}
    return [
        {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': meta,
         'data': {'config.yaml': config}},
        {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': meta,
         'spec': {
             'replicas': 1,
             'selector': {'matchLabels': labels},
             'template': {
                 'metadata': {'labels': labels, 'annotations': {
                     # LiteLLM reads both once at startup: a change must roll.
                     'infer-stack/config-hash': config_hash,
                     'infer-stack/key-hash': key_hash}},
                 'spec': {'containers': [{
                     'name': 'litellm',
                     'image': image,
                     'args': ['--config', '/etc/litellm/config.yaml',
                              '--port', str(LITELLM_CONTAINER_PORT)],
                     'envFrom': [{'secretRef': {'name': NAME}}],
                     'ports': [{'containerPort': LITELLM_CONTAINER_PORT}],
                     'readinessProbe': {
                         'httpGet': {'path': '/health/liveliness',
                                     'port': LITELLM_CONTAINER_PORT},
                         'periodSeconds': 5},
                     'volumeMounts': [{'name': 'config', 'mountPath': '/etc/litellm',
                                       'readOnly': True}],
                 }],
                     'volumes': [{'name': 'config', 'configMap': {'name': NAME}}]},
             },
         }},
        {'apiVersion': 'v1', 'kind': 'Service', 'metadata': meta,
         'spec': {'type': 'NodePort', 'selector': labels,
                  'ports': [{'port': LITELLM_CONTAINER_PORT,
                             'targetPort': LITELLM_CONTAINER_PORT,
                             'nodePort': node_port}]}},
    ]
