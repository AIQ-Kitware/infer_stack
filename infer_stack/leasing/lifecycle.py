"""Lifecycle evidence without equating a ready pod with successful generation."""
from __future__ import annotations


def pod_phase(pod: dict) -> tuple[str, str]:
    spec, status = pod.get('spec') or {}, pod.get('status') or {}
    reason, message = status.get('reason', ''), status.get('message', '')
    if status.get('phase') == 'Failed':
        return 'FAILED', ': '.join(part for part in [reason, message] if part) or 'Pod failed'
    scheduled = next((c for c in status.get('conditions') or [] if c.get('type') == 'PodScheduled'), {})
    if not spec.get('nodeName'):
        detail = scheduled.get('message') or scheduled.get('reason') or 'Waiting for scheduler'
        if 'Insufficient ' in detail:
            return 'WAITING_FOR_CAPACITY', detail
        if scheduled.get('reason') == 'Unschedulable':
            return 'BLOCKED', detail
        return 'SCHEDULING', detail
    containers = (status.get('initContainerStatuses') or []) + (status.get('containerStatuses') or [])
    for container in containers:
        state = container.get('state') or {}
        waiting = state.get('waiting') or {}
        terminated = state.get('terminated') or {}
        reason = waiting.get('reason') or terminated.get('reason') or ''
        detail = ': '.join(part for part in [reason, waiting.get('message') or terminated.get('message')] if part)
        if reason in {'CrashLoopBackOff', 'OOMKilled', 'Error'}:
            return 'FAILED', detail
        if reason in {'ImagePullBackOff', 'ErrImagePull', 'InvalidImageName', 'CreateContainerConfigError', 'CreateContainerError'}:
            return 'BLOCKED', detail
        if reason in {'ContainerCreating', 'PodInitializing'}:
            # ContainerCreating alone does not establish an image pull.
            return 'STARTING', detail
    return 'STARTING', 'Container running; generation has not yet been verified'


def kube_view(model: dict | None, pods: list[dict], *, generation_ready=False) -> dict:
    """Normalize observed state; phase labels require supporting runtime facts."""
    from ..backends.kubeai import model_replica_counts

    observed, ready = model_replica_counts(model or {})
    rows = []
    for pod in pods:
        phase, detail = pod_phase(pod)
        metadata = pod.get('metadata') or {}
        rows.append({'id': metadata.get('uid'), 'name': metadata.get('name'),
                     'node': (pod.get('spec') or {}).get('nodeName'), 'phase': phase, 'detail': detail,
                     'started_at': (pod.get('status') or {}).get('startTime')})
    spec = (model or {}).get('spec') or {}
    minimum, maximum = spec.get('minReplicas'), spec.get('maxReplicas')
    if ready is not None and ready > 0 and any(row['phase'] in {'FAILED', 'BLOCKED'} for row in rows):
        phase, action = 'DEGRADED', 'Inspect the unhealthy replica; other replicas report readiness'
    elif generation_ready:
        phase, action = 'READY', ''
    elif any(row['phase'] == 'FAILED' for row in rows):
        phase, action = 'FAILED', 'Inspect the failed replica and its previous container logs'
    elif any(row['phase'] == 'BLOCKED' for row in rows):
        phase, action = 'BLOCKED', 'Resolve the reported image, configuration, or scheduler constraint'
    elif any(row['phase'] == 'WAITING_FOR_CAPACITY' for row in rows):
        phase, action = 'WAITING_FOR_CAPACITY', 'Release idle capacity or add an eligible worker'
    elif not rows and minimum == 0 and observed == 0:
        phase, action = 'ON_DEMAND', 'A generation request wakes this logical Model'
    elif not rows:
        phase, action = 'SCHEDULING', 'Wait for KubeAI to create and schedule a serving pod'
    elif observed is not None and ready is not None and observed > ready:
        phase, action = 'SCALING', 'Wait for the remaining replicas; inspect individual startup phases'
    else:
        phase, action = 'STARTING', 'Verify readiness with a generation request'
    return {'phase': phase, 'next_action': action, 'min_replicas': minimum, 'max_replicas': maximum,
            'observed_replicas': observed, 'ready_replicas': ready,
            'generation_ready': generation_ready, 'replicas': rows}
