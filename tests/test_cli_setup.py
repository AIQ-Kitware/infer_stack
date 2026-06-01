from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from infer_stack import cli as cli_mod


MANAGE_PY = Path(__file__).resolve().parents[1] / "manage.py"


def _anchor_paths(tmp_path: Path, monkeypatch) -> Path:
    """Point both config_root() and data_root() at ``tmp_path``.

    ``config_root()`` / ``data_root()`` read these env vars fresh on every
    call, so this is enough for in-process ``cli_mod.cmd_*`` invocations to
    see the same layout as the subprocess ones launched via ``run_cli``.
    ``monkeypatch.setenv`` auto-restores at test teardown.
    """
    monkeypatch.setenv("INFER_STACK_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("INFER_STACK_DATA_DIR", str(tmp_path))
    return tmp_path


def run_cli(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    # Anchor config + rendered artifacts under tmp_path so tests are
    # independent of the user's ~/.config and ~/.cache directories.
    full_env.setdefault("INFER_STACK_CONFIG_DIR", str(tmp_path))
    full_env.setdefault("INFER_STACK_DATA_DIR", str(tmp_path))
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(MANAGE_PY), *args],
        env=full_env,
        text=True,
        capture_output=True,
        check=True,
    )


def write_kubeai_values_file(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "resourceProfiles": {
                    "gpu-single-default": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        },
                        "extraField": "keep-me",
                    },
                    "gpu-tp2-balanced": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        }
                    },
                    "gpu-tp2-maxctx": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_setup_compose_then_render_without_manual_file_edits(tmp_path: Path) -> None:
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", "qwen2-5-7b-instruct-turbo-default")
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "compose"
    assert cfg["active_profile"] == "qwen2-5-7b-instruct-turbo-default"
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    assert (tmp_path / "generated" / "docker-compose.yml").exists()


def test_setup_kubeai_then_render_without_manual_file_edits(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-72b-instruct-tp2-balanced",
        "--namespace",
        "demo-llm",
        "--ingress",
        "--ingress-host",
        "llm.example.test",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "kubeai"
    assert cfg["cluster"]["namespace"] == "demo-llm"
    assert cfg["cluster"]["ingress"]["enabled"] is True
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "2x96")
    assert (tmp_path / "generated" / "kubeai" / "models.yaml").exists()
    assert (tmp_path / "generated" / "kubeai" / "ingress.yaml").exists()


def test_setup_kubeai_with_resource_profiles_file_syncs_canonical_values(tmp_path: Path) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
        "--resource-profiles-file",
        str(source),
    )
    values_doc = yaml.safe_load((tmp_path / "kubeai-values.local.yaml").read_text())
    assert "gpu-single-default" in values_doc["resourceProfiles"]
    assert values_doc["resourceProfiles"]["gpu-single-default"]["extraField"] == "keep-me"


def test_render_overrides_backend_and_profile_without_persisting_config(tmp_path: Path) -> None:
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", "gpt-oss-20b-chat")
    run_cli(
        tmp_path,
        "render",
        "--yes",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-72b-instruct-tp2-balanced",
        "--namespace",
        "override-ns",
        "--simulate-hardware",
        "2x96",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "compose"
    assert cfg["active_profile"] == "gpt-oss-20b-chat"
    models = list(yaml.safe_load_all((tmp_path / "generated" / "kubeai" / "models.yaml").read_text()))
    namespace_doc = yaml.safe_load((tmp_path / "generated" / "kubeai" / "namespace.yaml").read_text())
    assert models[0]["metadata"]["name"] == "qwen2-72b-instruct-tp2-balanced"
    assert namespace_doc["metadata"]["name"] == "override-ns"


def test_setup_supports_environment_fallbacks(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        "setup",
        env={
            "INFER_STACK_BACKEND": "kubeai",
            "INFER_STACK_PROFILE": "gpt-oss-20b-chat",
            "INFER_STACK_NAMESPACE": "env-ns",
            "INFER_STACK_INGRESS_ENABLED": "true",
            "INFER_STACK_INGRESS_HOST": "env.example.test",
        },
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "kubeai"
    assert cfg["active_profile"] == "gpt-oss-20b-chat"
    assert cfg["cluster"]["namespace"] == "env-ns"
    assert cfg["cluster"]["ingress"]["enabled"] is True
    assert cfg["cluster"]["ingress"]["host"] == "env.example.test"


def test_kubeai_sync_resource_profiles_then_validate_without_config_duplication(tmp_path: Path) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    cfg.pop("resource_profiles", None)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_cli(tmp_path, "kubeai-sync-resource-profiles", "--from-file", str(source))
    run_cli(tmp_path, "validate", "--simulate-hardware", "1x96")


def test_kubeai_config_fallback_still_works_before_and_after_render_without_sync(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
    )
    run_cli(tmp_path, "validate", "--simulate-hardware", "1x96")
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    assert not (tmp_path / "kubeai-values.local.yaml").exists()
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    cfg["resource_profiles"]["gpu-single-default"]["node_selector"] = {"example.com/profile-source": "config-after-render"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_cli(tmp_path, "validate", "--simulate-hardware", "1x96")
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    rendered_values = yaml.safe_load((tmp_path / "generated" / "kubeai" / "kubeai-values.yaml").read_text())
    assert rendered_values["resourceProfiles"]["gpu-single-default"]["nodeSelector"]["example.com/profile-source"] == "config-after-render"


def test_kubeai_synced_source_is_preferred_over_config_and_preserves_unknown_fields(tmp_path: Path) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    cfg["resource_profiles"]["gpu-single-default"]["node_selector"] = {"example.com/profile-source": "config"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_cli(tmp_path, "kubeai-sync-resource-profiles", "--from-file", str(source))
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    generated_values = yaml.safe_load((tmp_path / "generated" / "kubeai" / "kubeai-values.yaml").read_text())
    assert generated_values["resourceProfiles"]["gpu-single-default"]["nodeSelector"]["nvidia.com/gpu.product"] == "NVIDIA_TEST_GPU"
    assert generated_values["resourceProfiles"]["gpu-single-default"]["extraField"] == "keep-me"
    assert "example.com/profile-source" not in generated_values["resourceProfiles"]["gpu-single-default"].get("nodeSelector", {})


def test_kubeai_local_values_change_marks_render_stale(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
        "--resource-profiles-file",
        str(source),
    )
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    _anchor_paths(tmp_path, monkeypatch)
    assert cli_mod.render_is_stale() is False
    time.sleep(1.1)
    values_doc = yaml.safe_load((tmp_path / "kubeai-values.local.yaml").read_text())
    values_doc["resourceProfiles"]["gpu-single-default"]["nodeSelector"]["example.com/profile-source"] = "updated-local"
    (tmp_path / "kubeai-values.local.yaml").write_text(yaml.safe_dump(values_doc, sort_keys=False), encoding="utf-8")
    assert cli_mod.render_is_stale() is True


def test_kubeai_deploy_rerenders_from_updated_local_values(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
        "--resource-profiles-file",
        str(source),
    )
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    _anchor_paths(tmp_path, monkeypatch)

    values_doc = yaml.safe_load((tmp_path / "kubeai-values.local.yaml").read_text())
    time.sleep(1.1)
    values_doc["resourceProfiles"]["gpu-single-default"]["nodeSelector"]["example.com/profile-source"] = "deploy-rerender"
    (tmp_path / "kubeai-values.local.yaml").write_text(yaml.safe_dump(values_doc, sort_keys=False), encoding="utf-8")

    observed: dict[str, object] = {}

    def fake_deploy_rendered_artifacts(deployment: dict) -> None:
        observed["source"] = deployment["resource_profiles_source"]

    monkeypatch.setattr(cli_mod.commands_runtime, "deploy_rendered_artifacts", fake_deploy_rendered_artifacts)
    cli_mod.DeployCLI.main(
        argv=False,
        detach=False,
        simulate_hardware="1x96",
        yes=True,
    )
    generated_values = yaml.safe_load((tmp_path / "generated" / "kubeai" / "kubeai-values.yaml").read_text())
    assert observed["source"] == str((tmp_path / "kubeai-values.local.yaml"))
    assert generated_values["resourceProfiles"]["gpu-single-default"]["nodeSelector"]["example.com/profile-source"] == "deploy-rerender"


def test_switch_apply_persists_only_active_profile_and_uses_transient_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "compose",
        "--profile",
        "gpt-oss-20b-chat",
        "--compose-cmd",
        "docker compose",
    )
    _anchor_paths(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def fake_deploy_rendered_artifacts(deployment: dict) -> None:
        observed["backend"] = deployment["backend"]
        observed["namespace"] = deployment["cluster"]["namespace"]
        observed["profile"] = deployment["serving_profile"]["public_name"]

    monkeypatch.setattr(cli_mod.commands_profile, "deploy_rendered_artifacts", fake_deploy_rendered_artifacts)
    cli_mod.SwitchCLI.main(
        argv=False,
        profile="qwen2-5-7b-instruct-turbo-default",
        backend="kubeai",
        compose_cmd="podman compose",
        namespace="override-ns",
        apply=True,
        simulate_hardware="1x96",
        yes=True,
    )

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["active_profile"] == "qwen2-5-7b-instruct-turbo-default"
    assert cfg["backend"] == "compose"
    assert cfg["runtime"]["compose_cmd"] == "docker compose"
    assert cfg["cluster"]["namespace"] != "override-ns"

    assert observed["backend"] == "kubeai"
    assert observed["namespace"] == "override-ns"
    assert observed["profile"] == "qwen2-5-7b-instruct-turbo-default"


def test_switch_apply_compose_recreates_router_when_config_changes(
    tmp_path: Path, monkeypatch
) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "compose",
        "--profile",
        "gpt-oss-20b-chat",
    )
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    _anchor_paths(tmp_path, monkeypatch)

    invocations: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        invocations.append(list(cmd))

    import infer_stack.docker_utils as du
    monkeypatch.setattr(du, "run", fake_run)

    cli_mod.SwitchCLI.main(
        argv=False,
        profile="qwen2-5-7b-instruct-turbo-default",
        apply=True,
        simulate_hardware="1x96",
        yes=True,
    )

    flat = [" ".join(c) for c in invocations]
    assert not any("down -v" in c or "--volumes" in c for c in flat)
    assert any(" up " in c and "-d" in c.split() for c in flat)
    # When live router refresh isn't reachable (no real LiteLLM under test),
    # we fall back to force-recreating just the litellm container — open-webui
    # is deliberately left running so chat sessions don't drop on a switch.
    recreate_cmds = [c for c in flat if "--force-recreate" in c]
    assert recreate_cmds, "expected a force-recreate fallback to fire"
    assert any("litellm" in c for c in recreate_cmds)
    assert all("open-webui" not in c for c in recreate_cmds), (
        "open-webui should not be force-recreated during a profile switch"
    )


def test_openwebui_state_paths_are_not_deleted_or_reinitialized_on_switch(
    tmp_path: Path, monkeypatch
) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "compose",
        "--profile",
        "gpt-oss-20b-chat",
    )
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")
    _anchor_paths(tmp_path, monkeypatch)

    state_root = tmp_path / "state"
    sentinel_open_webui = state_root / "open-webui" / "sentinel.txt"
    sentinel_postgres_owui = state_root / "postgres-open-webui" / "sentinel.txt"
    sentinel_postgres_litellm = state_root / "postgres-litellm" / "sentinel.txt"
    for sentinel in (sentinel_open_webui, sentinel_postgres_owui, sentinel_postgres_litellm):
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("keep", encoding="utf-8")

    invocations: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        invocations.append(list(cmd))

    import infer_stack.docker_utils as du
    monkeypatch.setattr(du, "run", fake_run)

    cli_mod.SwitchCLI.main(
        argv=False,
        profile="qwen2-5-7b-instruct-turbo-default",
        apply=True,
        simulate_hardware="1x96",
        yes=True,
    )

    flat = [" ".join(c) for c in invocations]
    assert not any("down -v" in c or "--volumes" in c for c in flat)
    assert not any("rm -rf" in c for c in flat)
    for sentinel in (sentinel_open_webui, sentinel_postgres_owui, sentinel_postgres_litellm):
        assert sentinel.exists() and sentinel.read_text() == "keep"


def test_kubeai_status_namespace_error_is_actionable(tmp_path: Path, monkeypatch) -> None:
    run_cli(tmp_path, "setup", "--backend", "kubeai", "--profile", "qwen2-5-7b-instruct-turbo-default", "--namespace", "default")
    _anchor_paths(tmp_path, monkeypatch)

    def fake_status(namespace: str) -> None:
        raise cli_mod.CommandError(f"Command failed in namespace {namespace}")

    monkeypatch.setattr(cli_mod.commands_runtime, "kubeai_print_status", fake_status)
    try:
        cli_mod.StatusCLI.main(argv=False)
    except SystemExit as ex:
        text = str(ex)
    else:
        raise AssertionError("expected StatusCLI.main to raise SystemExit")
    assert "namespace 'default'" in text
    assert "setup --backend kubeai --namespace default" in text


def test_status_reports_initialization_and_profile(tmp_path: Path) -> None:
    # Before setup: status reports it is uninitialized and exits cleanly
    # (no Docker / network / hardware probe required).
    pre = run_cli(tmp_path, "status")
    assert "Not initialized" in pre.stdout

    # After setup: status reports the active profile and that it is not yet
    # rendered, still without touching Docker.
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", "smollm2-135m-single")
    post = run_cli(tmp_path, "status")
    assert "Not initialized" not in post.stdout
    assert "active profile: smollm2-135m-single" in post.stdout
    assert "rendered:       no" in post.stdout


class _FakeResponse:
    """Minimal stand-in matching what ``_ready_openai_probe`` reads from a response."""

    def __init__(self, payload: dict[str, object], *, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


def _smoke_test_kwargs(**overrides) -> dict:
    base = dict(
        base_url="http://127.0.0.1:14042/v1",
        prompt="Say hello in one sentence.",
        max_tokens=64,
        simulate_hardware="1x96",
    )
    base.update(overrides)
    return base


def _setup_compose(tmp_path: Path, profile: str) -> None:
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", profile)
    run_cli(tmp_path, "render", "--yes", "--simulate-hardware", "1x96")


def test_smoke_test_uses_chat_endpoint_for_chat_profiles(tmp_path: Path, monkeypatch) -> None:
    _setup_compose(tmp_path, "qwen2-5-7b-instruct-turbo-default")
    _anchor_paths(tmp_path, monkeypatch)
    posted: dict[str, object] = {}

    def fake_get(url, **kwargs):
        return _FakeResponse({"data": [{"id": "qwen2-5-7b-instruct-turbo-default"}]})

    def fake_post(url, headers=None, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(cli_mod.requests, "get", fake_get)
    monkeypatch.setattr(cli_mod.requests, "post", fake_post)
    cli_mod.SmokeTestCLI.main(argv=False, **_smoke_test_kwargs(model="qwen2-5-7b-instruct-turbo-default"))
    assert posted["url"].endswith("/chat/completions")
    assert "messages" in posted["json"]
    assert "prompt" not in posted["json"]


def test_smoke_test_uses_completions_endpoint_for_completions_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    _setup_compose(tmp_path, "helm-pythia-1b-v0")
    _anchor_paths(tmp_path, monkeypatch)
    posted: dict[str, object] = {}

    def fake_get(url, **kwargs):
        return _FakeResponse({"data": [{"id": "eleutherai/pythia-1b-v0"}]})

    def fake_post(url, headers=None, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse({"choices": [{"text": "hi"}]})

    monkeypatch.setattr(cli_mod.requests, "get", fake_get)
    monkeypatch.setattr(cli_mod.requests, "post", fake_post)
    cli_mod.SmokeTestCLI.main(argv=False, **_smoke_test_kwargs(model="eleutherai/pythia-1b-v0"))
    assert posted["url"].endswith("/completions")
    assert not posted["url"].endswith("/chat/completions")
    assert "prompt" in posted["json"]
    assert "messages" not in posted["json"]


def test_smoke_test_protocol_override_forces_completions(
    tmp_path: Path, monkeypatch
) -> None:
    _setup_compose(tmp_path, "qwen2-5-7b-instruct-turbo-default")
    _anchor_paths(tmp_path, monkeypatch)
    posted: dict[str, object] = {}

    def fake_get(url, **kwargs):
        return _FakeResponse({"data": [{"id": "qwen2-5-7b-instruct-turbo-default"}]})

    def fake_post(url, headers=None, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse({"choices": [{"text": "hi"}]})

    monkeypatch.setattr(cli_mod.requests, "get", fake_get)
    monkeypatch.setattr(cli_mod.requests, "post", fake_post)
    cli_mod.SmokeTestCLI.main(
        argv=False,
        **_smoke_test_kwargs(model="qwen2-5-7b-instruct-turbo-default", protocol="completions"),
    )
    assert posted["url"].endswith("/completions")
    assert "prompt" in posted["json"]
