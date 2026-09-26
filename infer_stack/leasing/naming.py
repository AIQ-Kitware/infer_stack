"""The names infer-stack gives the things it runs, in one place.

A compose service, a Kubernetes object, and the upstream host a gateway
route points at are all derived from the served model name (or an Ollama
host). The engine side and the gateway side must agree on these exactly, so
both import them from here.
"""

from __future__ import annotations

import re

from .models import Deployment, served_name

#: The port every engine container serves on inside the Compose network.
VLLM_CONTAINER_PORT = 8000
OLLAMA_CONTAINER_PORT = 11434


def dns_slug(text: str) -> str:
    """A lowercase ``[a-z0-9-]`` label safe as a compose service / DNS /
    Kubernetes object name (shared by the compose and kubeai backends)."""
    out = re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')
    return out or 'model'


_dns_slug = dns_slug  # historical internal name


def vllm_service_name_for(served: str) -> str:
    """Deterministic compose/DNS service name for a vLLM upstream: ``vllm-<served>``.

    Derived purely from the served model name (vLLM's ``--served-model-name``,
    the Open WebUI label, the alias the user chose), so it is identical whether
    computed from a live :class:`Deployment` or from a catalog endpoint. That
    stability is what lets the LiteLLM gateway carry a *static* route table (one
    per catalog endpoint) whose upstream hosts match the containers when they
    come up — so adding/removing models does not rewrite the gateway's config and
    the gateway is never recreated (no "blip"); see :func:`_litellm_model_list`.
    """
    return f'vllm-{_dns_slug(served)}'


def _unique_vllm_service_name(served: str, deployment_id: str) -> str:
    """Per-deployment vLLM service/DNS name: ``vllm-<served>-<id-tail>``.

    The static-superset gateway needs a name derivable from the served model
    *alone* (so a catalog route can address it without knowing the live
    deployment) — but that deliberately drops the deployment id, which
    **collapses every** ``--dedicated`` **deployment of one model onto a single
    container** (hence one GPU). Dynamic routing manages the gateway's routes
    live via the admin API, so the upstream host no longer has to be predictable
    from the catalog. That frees us to give each deployment its **own** service,
    so N dedicated deployments of one model become N containers on N GPUs. The
    suffix is the deployment id's hex tail, keeping the name short and DNS-safe.
    """
    return f'{vllm_service_name_for(served)}-{deployment_tail(deployment_id)}'


def deployment_tail(deployment_id: str) -> str:
    """The short, DNS-safe suffix that makes a per-deployment name unique.

    >>> deployment_tail('grp-0123456789ab')
    '01234567'
    """
    return _dns_slug(deployment_id.rsplit('-', 1)[-1][:8] or 'x')


def vllm_service_name(deployment: Deployment, *, unique: bool = False) -> str:
    """Compose service name for a vLLM deployment (see :func:`vllm_service_name_for`).

    Default (``unique=False``, static-superset mode): deterministic from the
    served model name only — *no* deployment-id suffix — so it matches the
    gateway's pre-rendered route for that endpoint. Trade-off: two
    *simultaneously desired* deployments that share a served name collide on this
    name; under the static gateway the catalog endpoint is the unit, so that case
    (including same-model ``--dedicated``) is unsupported.

    ``unique=True`` (dynamic-routing mode): append the deployment-id tail
    (:func:`_unique_vllm_service_name`) so same-model dedicated deployments get
    distinct containers/GPUs; the admin-API route table addresses each by name.

    Either way the container carries the ``infer-stack.deployment`` label.
    :meth:`ComposeBackend.residency` correlates containers to deployments by that
    label, so the choice of suffix does not affect it. (The lenient
    :meth:`ComposeBackend.observe` still maps service names through the render
    sidecar; it is for reporting, not for decisions that touch a GPU.)
    """
    served = served_name(deployment)
    if unique:
        return _unique_vllm_service_name(served, deployment.id)
    return vllm_service_name_for(served)


def ollama_service_name_for(host: str) -> str:
    """Deterministic service name for an Ollama daemon: ``ollama-<host>``.

    One daemon per host (Ollama coalesces tags onto it), so the host is the
    stable key — matching :func:`vllm_service_name_for`'s role for vLLM so the
    gateway's static route table addresses it regardless of which tags are live.
    """
    return f'ollama-{_dns_slug(host)}'


def ollama_service_name(deployment: Deployment) -> str:
    host = deployment.spec.get('host') or deployment.id
    return ollama_service_name_for(host)
