"""
The checks that tell a wedged GPU from a lying gauge.

Every case here is one that actually happened on aiq-gpu, twice. The one that
matters most is ``test_unprivileged_holder_check_is_unknown_not_clear``:
unprivileged ``lsof``/``fuser`` see only the caller's own processes, so they
reported "nothing holds it" while ``nvidia-smi -r`` said ``In use by another
client``, and a reset was recommended on that basis.
"""
from infer_stack.gpu_doctor import GpuSample, Holder, gpu_checks


def _checks(**kw):
    return {name: (ok, detail) for name, ok, detail in gpu_checks(**kw)}


def test_busy_card_with_memory_allocated_is_not_flagged():
    """A card doing real work is busy for a reason."""
    res = _checks(_sample=lambda: [GpuSample(0, 100, 80_000)], _apps=lambda: [123],
                  _holders=lambda: [])
    assert res['utilization explained'][0] is True


def test_busy_card_with_nothing_allocated_is_flagged():
    """100% with 2 MiB is the phantom-busy signature."""
    res = _checks(_sample=lambda: [GpuSample(0, 100, 2)], _apps=lambda: [],
                  _holders=lambda: [])
    ok, detail = res['utilization explained']
    assert ok is False
    assert 'GPU0' in detail
    # It must not tell anyone to reset: the card may compute perfectly well,
    # which is what happened both times.
    assert 'reset' not in detail.lower()
    assert 'computes' in detail


def test_idle_cards_are_quiet():
    res = _checks(_sample=lambda: [GpuSample(i, 0, 2) for i in range(4)],
                  _apps=lambda: [], _holders=lambda: [])
    assert res['utilization explained'][0] is True


def test_unprivileged_holder_check_is_unknown_not_clear():
    """The bug this module exists for.

    device_holders returns None without root. That must not read as "nobody
    holds it" -- a false all-clear is what led to recommending a reset that
    could never succeed.
    """
    res = _checks(_sample=lambda: [GpuSample(0, 0, 2)], _apps=lambda: [],
                  _holders=lambda: None)
    ok, detail = res['device holders']
    assert ok is True                      # not a failure, but...
    assert 'not checked' in detail         # ...explicitly not an all-clear
    assert '--sudo' in detail


def test_persistenced_alone_is_expected():
    """It holds every device by design; flagging it would cry wolf always."""
    h = Holder(4036525, '/dev/nvidia0',
               '/usr/bin/nvidia-persistenced --user nvidia-persistenced', '/')
    res = _checks(_sample=lambda: [GpuSample(0, 0, 2)], _apps=lambda: [],
                  _holders=lambda: [h])
    assert res['device holders'][0] is True
    # ...but it is still why a reset gets refused, so say so.
    assert 'reset would be blocked' in res
    assert '-pm 0' in res['reset would be blocked'][1]


def test_a_kubernetes_pod_holding_a_device_is_named():
    """`pid 9030` is useless; the pod it lives in is the answer."""
    cg = ('0::/kubepods.slice/kubepods-besteffort.slice/'
          'kubepods-besteffort-pod00397bb3_dbeb_412e_8a60_731e5d34b4a7.slice/'
          'cri-containerd-4471688cb2a43ab78441da05bebe492e712d64fedbcfcc02c176d5e16273c8f9.scope')
    h = Holder(9030, '/dev/nvidia0', 'gpu-feature-discovery', cg)
    assert 'kubernetes pod' in h.where
    res = _checks(_sample=lambda: [GpuSample(0, 100, 2)], _apps=lambda: [],
                  _holders=lambda: [h])
    ok, detail = res['device holders']
    assert ok is False
    assert '9030' in detail and 'kubernetes pod' in detail
    assert 'refuse' in detail          # explains why a reset fails
    assert 'Stop the owning container' in detail


def test_a_docker_container_holding_a_device_is_named():
    h = Holder(555, '/dev/nvidia1', 'python -m vllm',
               '0::/system.slice/docker-4471688cb2a4deadbeef.scope')
    assert 'docker container' in h.where


def test_sampling_takes_the_minimum_so_one_spike_is_not_load():
    """Utilization is a windowed average; a spike right after a process exits
    is not sustained load."""
    from infer_stack import gpu_doctor
    seq = iter([
        '0, 100, 2, P0\n',
        '0, 0, 2, P8\n',
        '0, 0, 2, P8\n',
    ])
    gpu_doctor._run = lambda *a, **k: next(seq)
    try:
        got = gpu_doctor.sample_gpus(samples=3, interval=0, _sleep=lambda s: None)
    finally:
        import importlib
        importlib.reload(gpu_doctor)
    assert len(got) == 1
    assert got[0].util == 0, 'a single 100% sample must not read as busy'


def test_no_gpus_is_a_failure_not_a_pass():
    res = _checks(_sample=lambda: [], _apps=lambda: [], _holders=lambda: [])
    assert res['GPUs visible'][0] is False
