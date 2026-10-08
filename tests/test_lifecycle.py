from infer_stack.leasing.lifecycle import kube_view


def test_logical_presence_at_zero_is_on_demand_without_generation_claim():
    model = {'spec': {'minReplicas': 0, 'maxReplicas': 4}, 'status': {'replicas': {'all': 0, 'ready': 0}}}
    view = kube_view(model, [])
    assert view['phase'] == 'ON_DEMAND' and not view['generation_ready']
    assert view['observed_replicas'] == view['ready_replicas'] == 0


def test_terminal_pod_evidence_precedes_unscheduled_replacement():
    pods = [{'metadata': {'uid': 'old'}, 'status': {'phase': 'Failed', 'reason': 'Evicted', 'message': 'nodefs.available crossed threshold'}},
            {'metadata': {'uid': 'new'}, 'status': {'phase': 'Pending'}}]
    view = kube_view({}, pods)
    assert view['phase'] == 'FAILED'
    assert 'nodefs.available' in view['replicas'][0]['detail']


def test_capacity_and_disk_pressure_preserve_different_actions():
    def view(message):
        return kube_view({}, [{'status': {'conditions': [{'type': 'PodScheduled', 'reason': 'Unschedulable', 'message': message}]}}])
    assert view('Insufficient nvidia.com/gpu')['phase'] == 'WAITING_FOR_CAPACITY'
    pressured = view('untolerated taint node.kubernetes.io/disk-pressure')
    assert pressured['phase'] == 'BLOCKED'
    assert 'disk-pressure' in pressured['replicas'][0]['detail']


def test_unknown_counts_are_not_zero_and_ready_pods_are_not_generation():
    pod = {'spec': {'nodeName': 'worker'}, 'status': {'containerStatuses': [{'ready': True, 'state': {'running': {}}}]}}
    view = kube_view({}, [pod])
    assert view['observed_replicas'] is view['ready_replicas'] is None
    assert view['phase'] != 'READY' and not view['generation_ready']
    assert kube_view({}, [pod], generation_ready=True)['phase'] == 'READY'


def test_failed_replica_does_not_hide_healthy_replica_capacity():
    model = {'status': {'replicas': {'all': 2, 'ready': 1}}}
    pod = {'status': {'phase': 'Failed', 'reason': 'Evicted'}}
    view = kube_view(model, [pod], generation_ready=True)
    assert view['phase'] == 'DEGRADED'
    assert view['ready_replicas'] == 1 and view['generation_ready']


def test_mixed_node_memory_uses_smallest_verified_device():
    import json

    from infer_stack.backends.kubeai import DEVICE_FACTS_ANNOTATION, node_gpus

    inventory = {'version': 1, 'devices': [{'uuid': 'GPU-big', 'product': 'big', 'memory_mib': 49152},
                                          {'uuid': 'GPU-small', 'product': 'small', 'memory_mib': 16384}]}
    node = {'metadata': {'name': 'mixed', 'labels': {'nvidia.com/gpu.memory': '49152'},
                         'annotations': {DEVICE_FACTS_ANNOTATION: json.dumps(inventory)}},
            'status': {'allocatable': {'nvidia.com/gpu': '2'}}}
    facts = node_gpus([node])['mixed']
    assert facts['memory_gib'] == 16 and facts['heterogeneous']
    node['status']['allocatable']['nvidia.com/gpu'] = '3'
    incomplete = node_gpus([node])['mixed']
    assert incomplete['memory_gib'] is None and incomplete['inventory_error']
