"""Lease ledger: the stateful controller half of infer-stack.

See ``dev/infer-stack-redesign-critique.md`` (aiq-eval-runner) for the design.
This subpackage is backend-agnostic bookkeeping — ``acquire``/``release``/
``renew`` over a shared sqlite store with demand reference-counting, same-model
coalescing, and soft-TTL reclaim. The reconciler/backend layer (compose,
kubeai) and the CLI verbs are built on top of this in later phases.
"""

from __future__ import annotations

from .backend import Backend, MemoryBackend, NullBackend, Readiness
from .catalog import (
    Catalog,
    CatalogError,
    EndpointSpec,
    ModelSpec,
    RuntimeHostSpec,
)
from .controller import (
    AcquireOutcome,
    Controller,
    ReconcileResult,
    ReleaseOutcome,
    WaitResult,
)
from .ledger import (
    AcquireResult,
    Ledger,
    ReleaseResult,
    SweepResult,
    default_ledger_path,
)
from .models import (
    DeploymentGroup,
    EndpointRequest,
    GroupState,
    Lease,
    LeaseState,
    Sharing,
    capacity_satisfies,
    compatibility_key,
    ollama_structural,
    vllm_structural,
)
from .placement import (
    GpuPlan,
    available_indices,
    plan_placement,
    required_gpu_count,
)
from .store import SqliteStore

__all__ = [
    'AcquireOutcome',
    'AcquireResult',
    'Backend',
    'Catalog',
    'CatalogError',
    'Controller',
    'DeploymentGroup',
    'EndpointRequest',
    'EndpointSpec',
    'GpuPlan',
    'GroupState',
    'Ledger',
    'Lease',
    'LeaseState',
    'MemoryBackend',
    'ModelSpec',
    'NullBackend',
    'Readiness',
    'ReconcileResult',
    'ReleaseOutcome',
    'ReleaseResult',
    'RuntimeHostSpec',
    'Sharing',
    'SqliteStore',
    'SweepResult',
    'WaitResult',
    'available_indices',
    'capacity_satisfies',
    'compatibility_key',
    'default_ledger_path',
    'ollama_structural',
    'plan_placement',
    'required_gpu_count',
    'vllm_structural',
]
