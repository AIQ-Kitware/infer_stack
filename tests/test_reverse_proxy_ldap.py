from __future__ import annotations

from pathlib import Path

import yaml

from infer_stack.backends.compose_renderer import render_compose_artifacts
from infer_stack.config import initial_config
from infer_stack.hardware import simulate_inventory
from infer_stack.resolver import resolve
from infer_stack.validator import validate_resolved


def _cfg(tmp_path: Path, profile: str) -> dict:
    cfg = initial_config()
    cfg["backend"] = "compose"
    cfg["active_profile"] = profile
    cfg["output"]["generated_dir"] = str(tmp_path / "generated")
    for key in cfg["state"]:
        cfg["state"][key] = str(tmp_path / "state" / key)
    return cfg


def _cfg_with_profile(tmp_path: Path, name: str, profile: dict) -> dict:
    """Build a config whose active profile is a caller-supplied custom stack."""
    cfg = _cfg(tmp_path, name)
    cfg["profiles"][name] = profile
    return cfg


def _render(cfg: dict, tmp_path: Path, hardware: str = "1x24") -> dict:
    """Resolve + validate + render and return the parsed compose document."""
    dep = resolve(cfg, inventory=simulate_inventory(hardware))
    report = validate_resolved(dep)
    assert report["ok"] is True, report["errors"]
    render_compose_artifacts({"deployment": dep})
    return yaml.safe_load((tmp_path / "generated" / "docker-compose.yml").read_text())


def test_openwebui_tls_ldap_profile_renders_ldap_and_reverse_proxy(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "openwebui-tls-ldap"), inventory=simulate_inventory("2x24"))
    report = validate_resolved(dep)
    assert report["ok"] is True
    assert dep["frontends"]["open_webui"]["publish_port"] is False
    assert dep["providers"]["ollama"]["publish_port"] is False
    assert dep["frontends"]["reverse_proxy"]["enabled"] is True

    render_compose_artifacts({"deployment": dep})
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    compose_doc = yaml.safe_load(compose_text)
    services = compose_doc["services"]

    assert "reverse-proxy" in services
    assert "ports" not in services["open-webui"]
    assert "ports" not in services["ollama"]
    assert services["ollama"]["environment"]["OLLAMA_FLASH_ATTENTION"] == "1"
    assert services["ollama"]["environment"]["OLLAMA_NO_CLOUD"] == "true"
    assert services["open-webui"]["environment"]["ENABLE_LDAP"] == "true"
    assert services["open-webui"]["environment"]["LDAP_SERVER_HOST"] == "${LDAP_HOST}"
    assert services["reverse-proxy"]["ports"] == [
        "${INFER_STACK_REVERSE_PROXY_HTTP_PORT:-80}:80",
        "${INFER_STACK_REVERSE_PROXY_HTTPS_PORT:-443}:443",
    ]

    env_text = (tmp_path / "generated" / ".env").read_text()
    assert "LDAP_PORT=636" in env_text
    assert "LDAP_ATTRIBUTE_FOR_USERNAME=uid" in env_text

    nginx_text = (tmp_path / "state" / "runtime" / "nginx.conf").read_text()
    assert "server_name openwebui.example.com;" in nginx_text
    assert "proxy_pass         http://infer_stack_target;" in nginx_text
    assert "client_max_body_size 1G;" in nginx_text


def test_reverse_proxy_can_use_manual_nginx_config_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "ollama-direct")
    cfg["profiles"]["manual-nginx"] = {
        "description": "manual nginx config path smoke test",
        "providers": {
            "ollama": {
                "enabled": True,
                "gpu_indices": [0],
                "publish_port": False,
            }
        },
        "gateways": {"litellm": {"enabled": False}},
        "frontends": {
            "open_webui": {"enabled": True, "provider": "ollama", "publish_port": False},
            "reverse_proxy": {
                "enabled": True,
                "target": "open_webui",
                "ssl": {"enabled": False},
                "config_path": str(tmp_path / "manual-nginx.conf"),
            },
        },
        "routes": {},
    }
    cfg["active_profile"] = "manual-nginx"
    (tmp_path / "manual-nginx.conf").write_text("server { listen 80; }\n", encoding="utf-8")

    dep = resolve(cfg, inventory=simulate_inventory("1x24"))
    assert validate_resolved(dep)["ok"] is True
    render_compose_artifacts({"deployment": dep})
    compose_doc = yaml.safe_load((tmp_path / "generated" / "docker-compose.yml").read_text())
    volumes = compose_doc["services"]["reverse-proxy"]["volumes"]
    assert f"{tmp_path / 'manual-nginx.conf'}:/etc/nginx/conf.d/default.conf:ro" in volumes
    assert not (tmp_path / "state" / "runtime" / "nginx.conf").exists()


# --- Base profiles reused by the focused tests below ---------------------------

def _ollama_openwebui_profile(reverse_proxy: dict | None = None, open_webui: dict | None = None) -> dict:
    frontends: dict = {"open_webui": {"enabled": True, "provider": "ollama", "publish_port": False, **(open_webui or {})}}
    if reverse_proxy is not None:
        frontends["reverse_proxy"] = reverse_proxy
    return {
        "description": "ollama + open-webui test stack",
        "providers": {"ollama": {"enabled": True, "gpu_indices": [0], "publish_port": False}},
        "gateways": {"litellm": {"enabled": False}},
        "frontends": frontends,
        "routes": {},
    }


def test_http_only_reverse_proxy_has_no_tls_block(tmp_path: Path) -> None:
    cfg = _cfg_with_profile(
        tmp_path,
        "http-only",
        _ollama_openwebui_profile(reverse_proxy={"enabled": True, "target": "open_webui", "ssl": {"enabled": False}}),
    )
    compose_doc = _render(cfg, tmp_path)

    # No TLS means HTTPS must not be published, and port 80 alone is mapped.
    assert compose_doc["services"]["reverse-proxy"]["ports"] == ["${INFER_STACK_REVERSE_PROXY_HTTP_PORT:-80}:80"]
    nginx_text = (tmp_path / "state" / "runtime" / "nginx.conf").read_text()
    assert "listen 443" not in nginx_text
    assert "return 301 https" not in nginx_text  # no force_https without TLS
    assert "proxy_pass         http://infer_stack_target;" in nginx_text


def test_force_https_redirect_present_with_tls(tmp_path: Path) -> None:
    rp = {"enabled": True, "target": "open_webui", "server_name": "host.example.com",
          "ssl": {"enabled": True, "certificate": "cert.crt", "certificate_key": "key.key"}}
    cfg = _cfg_with_profile(tmp_path, "tls", _ollama_openwebui_profile(reverse_proxy=rp))
    _render(cfg, tmp_path)
    nginx_text = (tmp_path / "state" / "runtime" / "nginx.conf").read_text()
    assert "return 301 https://$host$request_uri;" in nginx_text
    assert "listen 443 ssl;" in nginx_text


def test_ldap_disabled_omits_ldap_env(tmp_path: Path) -> None:
    cfg = _cfg_with_profile(tmp_path, "no-ldap", _ollama_openwebui_profile())
    compose_doc = _render(cfg, tmp_path)
    env = compose_doc["services"]["open-webui"]["environment"]
    assert not any(key.startswith("LDAP_") or key == "ENABLE_LDAP" for key in env)


def test_reverse_proxy_target_litellm(tmp_path: Path) -> None:
    profile = {
        "description": "litellm target",
        "providers": {"ollama": {"enabled": True, "gpu_indices": [0]}},
        "gateways": {"litellm": {"enabled": True}},
        "frontends": {
            "open_webui": {"enabled": True, "provider": "litellm", "publish_port": False},
            "reverse_proxy": {"enabled": True, "target": "litellm", "ssl": {"enabled": False}},
        },
        "routes": {"chat": {"provider": "ollama", "model": "llama3"}},
    }
    cfg = _cfg_with_profile(tmp_path, "rp-litellm", profile)
    compose_doc = _render(cfg, tmp_path)
    assert "litellm" in compose_doc["services"]["reverse-proxy"]["depends_on"]
    nginx_text = (tmp_path / "state" / "runtime" / "nginx.conf").read_text()
    assert "server litellm:4000;" in nginx_text


def test_generic_service_escape_hatches_render(tmp_path: Path) -> None:
    open_webui = {
        "extra_env": {"WEBUI_NAME": "Demo", "CUSTOM_FLAG": "1"},
        "env_file": ["./openwebui.env"],
        "extra_volumes": ["./branding:/app/branding:ro"],
        "extra_hosts": ["registry.internal:10.0.0.5"],
        "labels": {"com.example.team": "infra"},
        "additional_ports": ["127.0.0.1:3001:8080"],
        "gpus": "all",
    }
    cfg = _cfg_with_profile(tmp_path, "hatches", _ollama_openwebui_profile(open_webui=open_webui))
    compose_doc = _render(cfg, tmp_path)
    svc = compose_doc["services"]["open-webui"]
    assert svc["environment"]["WEBUI_NAME"] == "Demo"
    assert svc["environment"]["CUSTOM_FLAG"] == "1"
    assert svc["env_file"] == ["./openwebui.env"]
    assert "./branding:/app/branding:ro" in svc["volumes"]
    assert svc["extra_hosts"] == ["registry.internal:10.0.0.5"]
    assert svc["labels"]["com.example.team"] == "infra"
    assert "127.0.0.1:3001:8080" in svc["ports"]
    assert svc["gpus"] == "all"


def test_structured_gpus_renders_as_yaml_list(tmp_path: Path) -> None:
    device_request = [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
    profile = _ollama_openwebui_profile()
    profile["providers"]["ollama"]["gpus"] = device_request
    cfg = _cfg_with_profile(tmp_path, "structured-gpus", profile)
    compose_doc = _render(cfg, tmp_path)
    ollama = compose_doc["services"]["ollama"]
    # The structured form round-trips as real YAML, not a quoted repr, and the
    # device-reservation deploy block is not also emitted.
    assert ollama["gpus"] == device_request
    assert "deploy" not in ollama


def test_extra_config_injected_into_nginx(tmp_path: Path) -> None:
    rp = {
        "enabled": True,
        "target": "open_webui",
        "ssl": {"enabled": True, "certificate": "cert.crt", "certificate_key": "key.key"},
        "extra_config": "location /healthz { return 200 'ok'; }",
    }
    cfg = _cfg_with_profile(tmp_path, "extra-config", _ollama_openwebui_profile(reverse_proxy=rp))
    _render(cfg, tmp_path)
    nginx_text = (tmp_path / "state" / "runtime" / "nginx.conf").read_text()
    assert "location /healthz { return 200 'ok'; }" in nginx_text


def test_no_watchtower_service_is_rendered(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "openwebui-tls-ldap"), inventory=simulate_inventory("2x24"))
    render_compose_artifacts({"deployment": dep})
    compose_doc = yaml.safe_load((tmp_path / "generated" / "docker-compose.yml").read_text())
    assert all("watchtower" not in name for name in compose_doc["services"])
    assert not any((svc or {}).get("image", "").startswith("nickfedor/watchtower") for svc in compose_doc["services"].values())


# --- Validator guards ----------------------------------------------------------

def test_validator_errors_when_target_disabled(tmp_path: Path) -> None:
    profile = _ollama_openwebui_profile(reverse_proxy={"enabled": True, "target": "litellm", "ssl": {"enabled": False}})
    cfg = _cfg_with_profile(tmp_path, "bad-target", profile)
    dep = resolve(cfg, inventory=simulate_inventory("1x24"))
    report = validate_resolved(dep)
    assert report["ok"] is False
    assert any("target=litellm" in e for e in report["errors"])


def test_validator_warns_on_missing_certs(tmp_path: Path) -> None:
    rp = {"enabled": True, "target": "open_webui",
          "ssl": {"enabled": True, "certificate": "", "certificate_key": "missing.key"}}
    cfg = _cfg_with_profile(tmp_path, "missing-certs", _ollama_openwebui_profile(reverse_proxy=rp))
    dep = resolve(cfg, inventory=simulate_inventory("1x24"))
    report = validate_resolved(dep)
    assert report["ok"] is True  # warnings, not errors
    joined = " ".join(report["warnings"])
    assert "ssl.certificate is empty" in joined
    assert "ssl.certificate_key" in joined and "not found" in joined


def test_validator_errors_on_publish_https_without_ssl_in_plan(tmp_path: Path) -> None:
    # Simulate a hand-edited plan where the resolver's gating was bypassed.
    cfg = _cfg_with_profile(
        tmp_path, "http-only",
        _ollama_openwebui_profile(reverse_proxy={"enabled": True, "target": "open_webui", "ssl": {"enabled": False}}),
    )
    dep = resolve(cfg, inventory=simulate_inventory("1x24"))
    dep["frontends"]["reverse_proxy"]["publish_https"] = True
    report = validate_resolved(dep)
    assert report["ok"] is False
    assert any("publishes HTTPS" in e for e in report["errors"])


def test_validator_warns_on_missing_config_path(tmp_path: Path) -> None:
    rp = {"enabled": True, "target": "open_webui", "ssl": {"enabled": False},
          "config_path": str(tmp_path / "does-not-exist.conf")}
    cfg = _cfg_with_profile(tmp_path, "missing-config", _ollama_openwebui_profile(reverse_proxy=rp))
    dep = resolve(cfg, inventory=simulate_inventory("1x24"))
    report = validate_resolved(dep)
    assert report["ok"] is True
    assert any("config_path" in w and "not found" in w for w in report["warnings"])
