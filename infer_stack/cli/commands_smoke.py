from __future__ import annotations

from ..benchmark import run_benchmark
from ..env_utils import parse_env_file
from ..kubeai_ops import print_status as kubeai_print_status
from ..paths import config_root
from ..profile_runtime import default_base_url
from pathlib import Path
from typing import Any
import json
import requests
import scriptconfig as scfg

from .context import _apply_path_overrides, _as_mapping, backend_name, build_plan, config_for_runtime, effective_allow_unsupported, effective_inventory, plan_path, runtime_env_path
from .probes import _default_model_for_deployment, _ready_ollama_probe, _ready_openai_probe, _resolve_smoke_protocol_from_deployment
from .compose import _explain_readiness_message, _print_compose_diagnostics, _print_gateway_diagnostics
from .options import _AllowUnsupportedMixin, _BackendOverrideMixin, _ClusterOverridesMixin, _ComposeOverrideMixin, _PathOverridesMixin, _PortOverridesMixin, _ProfileOverrideMixin, _SimulateHardwareMixin

# ---------------------------------------------------------------------------
# Smoke-test / benchmark commands
# ---------------------------------------------------------------------------





def _wait_until_ready(
    cfg: dict[str, Any],
    config: Any,
    *,
    model: str | None = None,
    timeout: float = 600.0,
    interval: float = 5.0,
    prompt: str = "Reply with ready.",
    max_tokens: int = 1,
    require_generation: bool = True,
    quiet: bool = False,
) -> str:
    """Wait until the active profile can serve a real request.

    Docker Compose health only tells us that a process/container passed its
    healthcheck.  For vLLM, the API can exist before the model path is fully
    ready through LiteLLM.  This probes the user-facing access surface and, by
    default, requires a tiny generation/completion to succeed.
    """
    import time

    plan = _smoke_plan(cfg, config)
    deployment = plan.get("deployment", {})
    access = deployment.get("access", {}).get("default", {}) or {}
    access_kind = str(access.get("kind") or "openai-compatible")
    base_url = _infer_default_base_url(cfg, config, deployment=deployment)
    model_name = _default_model_for_deployment(deployment, explicit=model)
    deadline = time.monotonic() + float(timeout)
    last_message = "not probed yet"
    attempt = 0

    env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
    headers = {"Content-Type": "application/json"}
    if access_kind != "ollama-native":
        auth_env_name = str(access.get("auth_env_name") or "LITELLM_MASTER_KEY")
        api_key = getattr(config, "api_key", None) or env.get(auth_env_name, "") or env.get("LITELLM_MASTER_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    protocol = _resolve_smoke_protocol_from_deployment(deployment, model_name)
    while True:
        attempt += 1
        if access_kind == "ollama-native":
            ok, message = _ready_ollama_probe(
                base_url=base_url,
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                require_generation=require_generation,
            )
        else:
            ok, message = _ready_openai_probe(
                base_url=base_url,
                headers=headers,
                model=model_name,
                protocol=protocol,
                prompt=prompt,
                max_tokens=max_tokens,
                require_generation=require_generation,
            )
        last_message = message
        if ok:
            if not quiet:
                print(f"Ready: {message}")
            return message
        now = time.monotonic()
        if now >= deadline:
            raise SystemExit(
                "Timed out waiting for the active stack to serve requests.\n"
                f"Last probe: {_explain_readiness_message(last_message)}\n"
                "Useful diagnostics:\n"
                "  infer-stack diagnose --logs --tail 80\n"
                "  infer-stack ps\n"
                "  infer-stack logs vllm-* litellm open-webui"
            )
        if not quiet and (attempt == 1 or attempt % 6 == 0):
            print(f"Waiting for readiness: {_explain_readiness_message(last_message)}")
        time.sleep(float(interval))

def _smoke_plan(cfg: dict[str, Any], config: Any) -> dict[str, Any]:
    overrides = _as_mapping(config)
    return build_plan(
        cfg,
        profile_name=overrides.get("profile"),
        allow_unsupported=effective_allow_unsupported(config, cfg),
        inventory=effective_inventory(config),
    )


def _infer_default_base_url(cfg: dict[str, Any], config: Any, deployment: dict[str, Any] | None = None) -> str:
    explicit = _as_mapping(config).get("base_url")
    if explicit:
        return str(explicit).rstrip("/")
    if deployment is None:
        try:
            deployment = _smoke_plan(cfg, config).get("deployment", {})
        except Exception:
            deployment = {
                "backend": backend_name(cfg),
                "cluster": cfg.get("cluster", {}),
                "ports": cfg.get("ports", {}),
            }
    return default_base_url(deployment)


def _smoke_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    timeout: float = 30,
    retries: int = 1,
    retry_delay: float = 2.0,
) -> requests.Response:
    """Wrapper around ``requests.{get,post}`` that emits actionable errors.

    The smoke test runs against a stack that may be (a) not listening yet,
    (b) listening but with an unhealthy upstream that resets connections, or
    (c) returning HTTP errors during model load. Retry transient startup
    failures so ``switch --apply && smoke-test`` is usable immediately after a
    provider container was recreated.
    """
    last_timeout: requests.exceptions.Timeout | None = None
    last_conn: requests.exceptions.ConnectionError | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            break
        except requests.exceptions.Timeout as ex:
            last_timeout = ex
            if attempt < retries:
                import time

                time.sleep(retry_delay)
                continue
            raise SystemExit(
                f"Request to {url} timed out after {timeout}s.\n"
                "The model may still be loading, or the server is overloaded.\n"
                "  infer-stack logs vllm-*"
            ) from ex
        except requests.exceptions.ConnectionError as ex:
            last_conn = ex
            if attempt < retries:
                import time

                time.sleep(retry_delay)
                continue
            # Two distinct sub-cases inside ConnectionError that warrant different
            # remediation: (a) nothing listening on the port, (b) something is
            # listening but it closed the connection without responding (typical
            # of LiteLLM up but a depended-on vLLM container still loading the
            # model and failing the dependency health-check chain).
            cause = ex.args[0] if ex.args else ex
            cause_str = str(cause)
            if "RemoteDisconnected" in cause_str or "Connection aborted" in cause_str:
                raise SystemExit(
                    f"Connection to {url} was closed before a response arrived.\n"
                    "The router is listening but an upstream service is not ready yet.\n"
                    "Check container status and logs:\n"
                    "  infer-stack ps\n"
                    "  infer-stack logs vllm-*"
                ) from ex
            if "Connection refused" in cause_str or "Failed to establish a new connection" in cause_str:
                raise SystemExit(
                    f"Could not connect to {url}: nothing is listening yet.\n"
                    "If you just ran `infer-stack up`, give the router a few seconds.\n"
                    "  infer-stack ps                # confirm the litellm container is running\n"
                    "  infer-stack logs litellm      # check for startup errors"
                ) from ex
            raise SystemExit(f"Connection error reaching {url}: {cause_str}") from ex
    else:  # pragma: no cover - defensive; loop exits via break or raise
        if last_timeout is not None:
            raise last_timeout
        if last_conn is not None:
            raise last_conn
        raise RuntimeError("smoke request failed without an exception")
    status = getattr(resp, "status_code", 200)
    if status >= 400:
        body = getattr(resp, "text", "") or ""
        body = body.strip()
        if len(body) > 500:
            body = body[:500] + "... [truncated]"
        reason = getattr(resp, "reason", "") or ""
        if status in (401, 403):
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "The auth key didn't match what the running container expects.\n"
                "If you re-rendered after the container started, the key in .env "
                "may have changed. Restart with:\n"
                "  infer-stack down && infer-stack up -d\n"
                f"Response: {body}"
            )
        if status == 503:
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "An upstream service is unavailable (commonly the vLLM engine is still loading).\n"
                "  infer-stack logs vllm-*\n"
                f"Response: {body}"
            )
        raise SystemExit(
            f"HTTP {status} {reason} from {url}.\nResponse: {body}"
        )
    return resp


def _resolve_smoke_test_protocol(
    cfg: dict[str, Any],
    config: Any,
    model_name: str,
) -> str:
    """Pick the OpenAI route for smoke-test based on protocol resolution order.

    1. ``--protocol`` CLI override (``chat`` or ``completions``).
    2. Resolved deployment: if the requested model maps to a service whose
       protocol_mode is known, use that.
    3. Active profile's primary service protocol_mode.
    4. Fallback: ``chat``.
    """
    overrides = _as_mapping(config)
    explicit = overrides.get("protocol")
    if explicit:
        return str(explicit)
    try:
        plan = build_plan(
            cfg,
            profile_name=overrides.get("profile"),
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
    except Exception:
        return "chat"
    deployment = plan.get("deployment", {})
    return _resolve_smoke_protocol_from_deployment(deployment, model_name)


def _ollama_smoke_test(
    base_url: str,
    *,
    model: str | None,
    prompt: str,
    max_tokens: int,
    skip_chat: bool,
) -> int:
    """Smoke-test an Ollama-native endpoint without requiring LiteLLM."""
    tags_resp = _smoke_request("GET", f"{base_url}/api/tags", timeout=30, retries=12, retry_delay=5)
    tags_doc = tags_resp.json()
    print(json.dumps(tags_doc, indent=2))
    if skip_chat:
        return 0
    models = tags_doc.get("models") or []
    model_name = model or (models[0].get("name") if models else None)
    if not model_name:
        raise SystemExit(
            "Ollama is reachable, but no models are installed in its model store.\n"
            "Pull one through the CLI wrapper, for example:\n"
            "  infer-stack ollama-pull smollm2:135m\n"
            "Then rerun:\n"
            "  infer-stack smoke-test --model smollm2:135m"
        )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    resp = _smoke_request("POST", f"{base_url}/api/chat", json_body=payload, timeout=120, retries=3, retry_delay=5)
    print(json.dumps(resp.json(), indent=2))
    return 0



class DiagnoseCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Print targeted diagnostics for the active rendered stack.

    This command is intentionally more specific than ``ps`` or ``logs``.  It
    prints the resolved provider/gateway/frontend graph, rendered compose
    service state, LiteLLM route probes, direct provider probes, and optional
    recent logs.  It helps distinguish these cases:

    * LiteLLM container is actually absent/down.
    * LiteLLM is running but its upstream vLLM process is still booting.
    * Open WebUI is polling a provider that is not present in the active
      profile.
    """

    __command__ = "diagnose"

    model = scfg.Value(None, type=str, help="Model/alias to use for optional generation diagnostics.")
    logs = scfg.Value(False, isflag=True, help="Include recent logs for litellm/open-webui/vllm/ollama services.")
    tail = scfg.Value(80, type=int, help="Number of log lines per service when --logs is set.")
    generation = scfg.Value(False, isflag=True, help="Also run a tiny generation probe through the active access surface.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=_as_mapping(config).get("profile"),
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        deployment = plan.get("deployment", {}) or {}
        print(f"active_profile: {deployment.get('source', {}).get('active_profile') or cfg.get('active_profile')}")
        print(f"backend: {deployment.get('backend') or backend_name(cfg)}")
        print(f"plan: {plan_path(cfg)}")
        access = (deployment.get("access", {}) or {}).get("default", {}) or {}
        if access:
            print("default access:")
            print(f"  kind: {access.get('kind')}")
            print(f"  base_url: {access.get('base_url')}")
            if access.get("auth_env_name"):
                print(f"  auth_env_name: {access.get('auth_env_name')}")

        providers = deployment.get("providers", {}) or {}
        gateways = deployment.get("gateways", {}) or {}
        frontends = deployment.get("frontends", {}) or {}
        print("\nresolved graph:")
        print(f"  providers: {', '.join(k for k, v in providers.items() if (v or {}).get('enabled') or (v or {}).get('runtimes')) or 'none'}")
        print(f"  gateways:  {', '.join(k for k, v in gateways.items() if (v or {}).get('enabled')) or 'none'}")
        print(f"  frontends: {', '.join(k for k, v in frontends.items() if (v or {}).get('enabled')) or 'none'}")
        litellm_routes = ((gateways.get("litellm") or {}).get("routes") or {})
        if litellm_routes:
            print("\nLiteLLM routes:")
            for alias, route in litellm_routes.items():
                print(
                    f"  {alias}: provider={route.get('provider')} "
                    f"runtime={route.get('runtime', '-')} upstream={route.get('upstream_model', route.get('model', '-'))} "
                    f"protocol={route.get('protocol_mode', 'chat')}"
                )

        if backend_name(cfg) == "compose":
            _print_compose_diagnostics(cfg, tail=int(config.tail) if config.logs else 0)
            _print_gateway_diagnostics(cfg, deployment, model=config.model, require_generation=bool(config.generation))
        else:
            namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
            kubeai_print_status(namespace)
        return 0


class WaitReadyCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Wait until the active profile can serve a real request.

    This is stronger than Docker Compose health.  It probes the same access
    surface users will hit (LiteLLM, direct Ollama, or direct vLLM) and, by
    default, requires a tiny generation/completion to succeed.
    """

    __command__ = "wait-ready"

    base_url = scfg.Value(None, type=str, help="Override the resolved base URL.")
    api_key = scfg.Value(None, type=str, help="Override the auth key for OpenAI-compatible surfaces.")
    model = scfg.Value(None, type=str, help="Model/alias to probe. Defaults to the first active route/runtime.")
    prompt = scfg.Value("Reply with ready.", type=str)
    max_tokens = scfg.Value(1, type=int)
    timeout = scfg.Value(600, type=float, help="Maximum seconds to wait.")
    interval = scfg.Value(5, type=float, help="Seconds between probes.")
    skip_generation = scfg.Value(False, isflag=True, help="Only wait for the API model listing/tag endpoint, not generation.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        _wait_until_ready(
            cfg,
            config,
            model=config.model,
            timeout=float(config.timeout),
            interval=float(config.interval),
            prompt=config.prompt,
            max_tokens=int(config.max_tokens),
            require_generation=not bool(config.skip_generation),
            quiet=False,
        )
        return 0


class SmokeTestCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Probe the running router with a single chat/completions request."""

    __command__ = "smoke-test"

    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)
    model = scfg.Value(None, type=str)
    prompt = scfg.Value("Say hello in one sentence.", type=str)
    max_tokens = scfg.Value(128, type=int)
    skip_chat = scfg.Value(False, isflag=True)
    no_wait = scfg.Value(False, isflag=True, help="Do not wait for the active access surface to serve a real request before the smoke request.")
    wait_timeout = scfg.Value(600, type=float, help="Seconds to wait for readiness before the smoke request.")
    wait_interval = scfg.Value(5, type=float, help="Seconds between readiness probes.")
    protocol = scfg.Value(
        None,
        choices=["chat", "completions"],
        help="Force the smoke-test endpoint. Defaults to the resolved profile's protocol_mode.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = _smoke_plan(cfg, config)
        deployment = plan.get("deployment", {})
        access = deployment.get("access", {}).get("default", {}) or {}
        env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
        base_url = _infer_default_base_url(cfg, config, deployment=deployment)

        if not bool(config.no_wait):
            _wait_until_ready(
                cfg,
                config,
                model=config.model,
                timeout=float(config.wait_timeout),
                interval=float(config.wait_interval),
                prompt=config.prompt,
                max_tokens=1,
                require_generation=not bool(config.skip_chat),
                quiet=True,
            )

        access_kind = str(access.get("kind") or "openai-compatible")
        explicit_base_url = bool(_as_mapping(config).get("base_url"))
        if access_kind == "ollama-native" and not explicit_base_url:
            return _ollama_smoke_test(
                base_url,
                model=config.model,
                prompt=config.prompt,
                max_tokens=int(config.max_tokens),
                skip_chat=bool(config.skip_chat),
            )

        headers = {"Content-Type": "application/json"}
        auth_env_name = str(access.get("auth_env_name") or "LITELLM_MASTER_KEY")
        api_key = config.api_key or env.get(auth_env_name, "") or env.get("LITELLM_MASTER_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        models_resp = _smoke_request("GET", f"{base_url}/models", headers=headers, timeout=30, retries=12, retry_delay=5)
        models = models_resp.json().get("data", [])
        print(json.dumps(models_resp.json(), indent=2))
        if config.skip_chat:
            return 0
        if not models:
            raise SystemExit("No models returned from /models")
        model_name = config.model or models[0]["id"]
        protocol = _resolve_smoke_test_protocol(cfg, config, model_name)
        if protocol == "completions":
            payload = {
                "model": model_name,
                "prompt": config.prompt,
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/completions"
        else:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": config.prompt}],
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/chat/completions"
        resp = _smoke_request("POST", endpoint, headers=headers, json_body=payload, timeout=120, retries=3, retry_delay=5)
        print(json.dumps(resp.json(), indent=2))
        return 0


class BenchmarkCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Run benchmark_prompts.json against the router."""

    model = scfg.Value(None, type=str, required=True)
    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        # benchmark_prompts.json is a user-supplied fixture. Look for it
        # first in the config dir, then fall back to CWD so an ad-hoc
        # invocation from a checkout still picks up a sibling file.
        prompts_path = config_root() / "benchmark_prompts.json"
        if not prompts_path.exists():
            prompts_path = Path.cwd() / "benchmark_prompts.json"
        if not prompts_path.exists():
            raise SystemExit(
                f"benchmark_prompts.json not found at {config_root() / 'benchmark_prompts.json'} "
                f"or {Path.cwd() / 'benchmark_prompts.json'}"
            )
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        cfg = config_for_runtime(config)
        env = parse_env_file(runtime_env_path(cfg))
        base_url = config.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
        api_key = config.api_key or env.get("LITELLM_MASTER_KEY", "")
        data = run_benchmark(base_url, api_key, config.model, prompts)
        print(json.dumps(data, indent=2))
        return 0
