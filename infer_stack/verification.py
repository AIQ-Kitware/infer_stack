from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import KUBEAI_GENERATED_SUBDIR, normalized_output
from .profile_runtime import default_base_url


def verify_profile(deployment: dict[str, Any]) -> dict[str, Any]:
    service = deployment.get("services", [])[0] if deployment.get("services") else {}
    output_root = Path(normalized_output(deployment.get("output"))["generated_dir"])
    expected = {
        "public_name": deployment["serving_profile"]["public_name"],
        "logical_model_name": service.get("logical_model_name", ""),
        "protocol_mode": service.get("protocol_mode", deployment["serving_profile"].get("protocol_mode", "")),
        "served_model_name": service.get("served_model_name", deployment["serving_profile"].get("served_model_name", "")),
        "endpoint_base_url": default_base_url(deployment),
        "generated_artifacts": {
            "compose": str(output_root / "docker-compose.yml"),
            "kubeai_models": str(output_root / KUBEAI_GENERATED_SUBDIR / "models.yaml"),
        },
    }
    checks = {
        "has_services": bool(deployment.get("services")),
        "has_public_name": bool(expected["public_name"]),
        "has_logical_model_name": bool(expected["logical_model_name"]),
        "has_protocol_mode": bool(expected["protocol_mode"]),
    }
    return {
        "ok": all(checks.values()),
        "profile": expected,
        "checks": checks,
    }
