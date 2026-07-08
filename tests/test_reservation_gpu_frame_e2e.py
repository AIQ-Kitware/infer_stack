"""E2E probe: does THIS machine satisfy the reserved-GPU index-frame assumption?

The reserved-GPU lease (``acquire --reserve-gpus N``) works by handing an external
process the GPU **index** infer-stack's placement assigned, via the env-file's
``CUDA_VISIBLE_DEVICES``. A consumer (e.g. eval_audit's docker node) then pins its
container with ``docker run --gpus "device=<index>"``. That is only correct if
**infer-stack's inventory index and docker's ``--gpus device=`` index refer to the
same physical GPU** — the "nuance" flagged in
``docs/planning/huggingface-in-process-reserved-gpu-plan.md``. Cgroup renumbering
(SLURM), MIG, or an unusual ``CUDA_DEVICE_ORDER`` could break that equivalence.

Unlike the rest of the suite (FakeDocker mocks), this ACTUALLY runs docker + GPUs,
so it is **opt-in / environment-probing**: run it on the target host to confirm the
assumption holds *there*. It SKIPS when the probe can't run (no docker, no
nvidia-smi, no GPUs, GPU-docker not wired, image unavailable) and only FAILS on a
genuine frame mismatch — a real "this machine violates our assumption" signal.

    # on the target host (the image must have nvidia-smi; override if needed):
    INFER_STACK_E2E_GPU_IMAGE=nvidia/cuda:12.4.1-base-ubuntu22.04 \
      pytest tests/test_reservation_gpu_frame_e2e.py -o addopts='' -v

Set ``INFER_STACK_E2E_GPU_IMAGE`` to any CUDA image already present on the host
(e.g. the vLLM serving image) to avoid a pull.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from infer_stack.hardware import detect_inventory
from infer_stack.leasing import plan_placement, reservation_request
from infer_stack.leasing.models import Deployment, DeploymentState

# Any image carrying nvidia-smi. Overridable so an operator can point at an image
# already cached on the host (a pull inside CI-less GPU boxes is often undesirable).
_GPU_IMAGE = os.environ.get(
    "INFER_STACK_E2E_GPU_IMAGE", "nvidia/cuda:12.4.1-base-ubuntu22.04"
)
_DOCKER_TIMEOUT = int(os.environ.get("INFER_STACK_E2E_DOCKER_TIMEOUT", "180"))


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=_DOCKER_TIMEOUT
    )


def _container_gpu_uuids(device_arg: str) -> list[str]:
    """UUIDs of the GPUs a container sees for ``docker run --gpus device=<arg>``.

    Queries ``uuid`` (a stable physical identifier), NOT ``index`` — the index is
    renumbered to 0..k-1 inside the container, so it can't verify the frame; the
    UUID can.
    """
    proc = _run(
        [
            "docker", "run", "--rm", "--gpus", f"device={device_arg}",
            _GPU_IMAGE,
            "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker --gpus device={device_arg} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def inventory() -> dict:
    if shutil.which("docker") is None:
        pytest.skip("docker not installed — cannot probe the GPU index frame")
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not available — no GPU host to probe")
    inv = detect_inventory()
    if not inv.get("gpu_count"):
        pytest.skip("no GPUs detected — nothing to probe")
    # Baseline: is GPU-docker even wired here? If `--gpus all` can't start the
    # probe image, that is an environment gap (nvidia-container-toolkit, image
    # pull, driver), NOT an assumption violation — skip rather than fail.
    try:
        uuids = _container_gpu_uuids("all")
    except Exception as ex:  # noqa: BLE001
        pytest.skip(
            f"GPU-docker not runnable here ({ex}); set INFER_STACK_E2E_GPU_IMAGE "
            "to an image present on this host, or check the nvidia container runtime"
        )
    if not uuids:
        pytest.skip("container saw no GPUs under --gpus all; nvidia runtime not wired")
    return inv


def test_docker_device_index_matches_inventory_frame(inventory: dict) -> None:
    """``docker --gpus device=N`` must select the GPU infer-stack calls index N.

    This is THE reserved-GPU assumption: the index placement assigns (and writes to
    CUDA_VISIBLE_DEVICES) is the physical GPU the consumer's ``--gpus device=<idx>``
    then pins. A UUID mismatch here means the machine renumbers between the two
    frames (SLURM cgroup / MIG / CUDA_DEVICE_ORDER) and the reserved-GPU path would
    run on the WRONG card — fail loudly so we find out before trusting it.
    """
    mismatches = []
    for gpu in inventory["gpus"]:
        idx = gpu["index"]
        expected_uuid = gpu["uuid"]
        seen = _container_gpu_uuids(str(idx))
        if seen != [expected_uuid]:
            mismatches.append(
                f"index {idx}: inventory UUID {expected_uuid!r}, but "
                f"`docker --gpus device={idx}` saw {seen!r}"
            )
    assert not mismatches, (
        "GPU index frame mismatch — infer-stack's inventory index does NOT agree "
        "with docker's `--gpus device=` index on this machine, so the reserved-GPU "
        "path (CUDA_VISIBLE_DEVICES -> --gpus device=) would pin the wrong card:\n"
        + "\n".join(mismatches)
    )


def test_reserved_index_is_dockerable_and_correctly_framed(inventory: dict) -> None:
    """The exact index a *reservation* would assign is dockerable and frame-correct.

    Exercises the feature's own path: placement picks a GPU for a reservation
    against the real inventory (honoring any INFER_STACK_ALLOWED_GPUS the host set,
    e.g. $SLURM_JOB_GPUS), and the index it hands out — the value that lands in the
    env-file's CUDA_VISIBLE_DEVICES — maps under docker to the physical GPU
    infer-stack believes it reserved.
    """
    req = reservation_request(1)
    reservation = Deployment(
        "res-e2e", req.compat_key, req.engine, req.sharing, dict(req.capacity),
        dict(req.spec), {req.endpoint: {}}, DeploymentState.LIVE, 0.0, 0.0,
    )
    plan = plan_placement([reservation], inventory)
    assert not plan.errors, f"reservation could not be placed: {plan.errors}"
    reserved_idx = plan.assignments["res-e2e"][0]

    by_index = {g["index"]: g["uuid"] for g in inventory["gpus"]}
    expected_uuid = by_index[reserved_idx]
    seen = _container_gpu_uuids(str(reserved_idx))
    assert seen == [expected_uuid], (
        f"reservation assigned GPU index {reserved_idx} (UUID {expected_uuid!r}), "
        f"but `docker --gpus device={reserved_idx}` saw {seen!r} — the reserved-GPU "
        "lease would run the in-process model on the wrong card on this machine"
    )
