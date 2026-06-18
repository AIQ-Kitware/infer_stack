from __future__ import annotations

from ..config import load_yaml
from ..docker_utils import PortInUseError
from ..docker_utils import check_ports_available
from ..docker_utils import compose_recreate_router
from ..docker_utils import compose_up
from ..docker_utils import our_published_ports
from ..env_utils import parse_env_file
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import subprocess

if TYPE_CHECKING:
    import requests

from .context import (
    generated_dir,
    plan_path,
    runtime_env_path,
    runtime_litellm_config_path,
)
from .probes import (
    _default_model_for_deployment,
    _ready_openai_probe,
    _resolve_smoke_protocol_from_deployment,
)

# Backend-specific helpers
# ---------------------------------------------------------------------------


def _compose_base_cmd(cfg: dict[str, Any]) -> list[str]:
    """Build the shared ``docker compose -f ... --env-file ...`` prefix.

    Used by every compose-wrapper subcommand so the user doesn't have to
    cd into the rendered-artifacts directory just to run a one-shot
    ``ps`` / ``restart`` / ``pull`` / ``logs``.
    """
    compose_file = generated_dir(cfg) / 'docker-compose.yml'
    env_file = generated_dir(cfg) / '.env'
    return cfg['runtime']['compose_cmd'].split() + [
        '-f',
        str(compose_file),
        '--env-file',
        str(env_file),
    ]


def _kubeai_stub(cmd_name: str) -> None:
    """Raise for a day-2-ops subcommand that has no kubeai implementation yet.

    These wrappers (logs/ps/restart/pull/start/stop) compose docker-compose
    invocations and have no kubectl equivalent in this CLI. Until somebody
    writes one, surface the gap as ``NotImplementedError`` so callers can
    distinguish "kubeai doesn't do this yet" from a real failure.
    """
    raise NotImplementedError(
        f'`{cmd_name}` is not implemented for the kubeai backend yet. '
        f'Use the equivalent kubectl command in the meantime '
        f'(e.g. `kubectl -n <namespace> ...`).'
    )


def _compose_up_with_router_recreate(
    cfg: dict[str, Any],
    *,
    detach: bool,
) -> None:
    """Run ``compose up`` and refresh LiteLLM's model list to match the new render.

    A compose ``up`` against the rendered stack only restarts vLLM services
    whose specs changed. LiteLLM and Open WebUI keep their existing
    containers and would therefore serve stale model lists until something
    else refreshed them. Two ways to do that:

    1. **Live refresh** (preferred): talk to LiteLLM's admin API
       (``/model/new`` / ``/model/delete``) to diff and apply alias changes
       in-process. LiteLLM and Open WebUI stay up — users hitting unchanged
       models see no disruption. Skipped automatically on cold start or if
       the admin API is unreachable.
    2. **Container recreate** (fallback): force-recreate only the LiteLLM
       container so it reloads the rendered YAML on startup. Open WebUI stays
       up; used only when live refresh fails and Compose did not already
       reload LiteLLM during convergence.
    """
    compose_file = generated_dir(cfg) / 'docker-compose.yml'
    env_file = generated_dir(cfg) / '.env'
    compose_cmd = cfg['runtime']['compose_cmd']

    _preflight_check_ports(cfg)

    litellm_in_render = _compose_has_service(compose_file, 'litellm')
    litellm_before = (
        _compose_service_state(compose_cmd, compose_file, env_file, 'litellm')
        if litellm_in_render
        else {}
    )

    compose_up(
        compose_cmd,
        compose_file,
        env_file,
        detach=detach,
        remove_orphans=True,
    )

    # If `up` failed it would have already raised; only do the router refresh
    # in detached mode (foreground `up` keeps the user attached to logs and
    # leaves cycling decisions to compose).
    if not detach:
        return

    if not runtime_litellm_config_path(cfg).exists():
        return

    if not litellm_in_render:
        # Direct Ollama / raw-server profiles intentionally do not render a
        # LiteLLM service.  A stale litellm_config.yaml may still exist in the
        # runtime directory from a previous gateway profile, but that must not
        # trigger a router refresh/recreate against a service that is no longer
        # present in the active compose file.
        return

    litellm_after = _compose_service_state(
        compose_cmd, compose_file, env_file, 'litellm'
    )
    litellm_reloaded_by_compose = bool(litellm_after) and (
        litellm_after.get('id') != litellm_before.get('id')
        or litellm_after.get('started_at') != litellm_before.get('started_at')
        or litellm_before.get('running') not in {'true', 'True'}
    )
    if litellm_reloaded_by_compose:
        # Compose already created or restarted LiteLLM while converging the
        # stack.  The process has read the freshly rendered YAML, so a second
        # live refresh or forced recreate is redundant churn.
        print(
            'LiteLLM was started/reloaded by compose; skipping extra router refresh.'
        )
        return

    try:
        _litellm_refresh_router_live(cfg)
        return
    except RouterRefreshError as ex:
        print(
            f'Live router refresh skipped ({ex}); '
            'recreating litellm container to reload config from YAML '
            '(open-webui stays up)...'
        )

    compose_recreate_router(
        compose_cmd,
        compose_file,
        env_file,
        detach=True,
    )


def _compose_has_service(compose_file: Path, service_name: str) -> bool:
    """Return true when a rendered compose file contains ``service_name``.

    This is intentionally based on the rendered compose file rather than the
    presence of sidecar artifacts such as ``runtime/litellm_config.yaml``.
    Runtime artifacts are persistent across profile switches, while the compose
    service list is the active source of truth for what ``docker compose up``
    can recreate.
    """
    try:
        doc = load_yaml(compose_file)
    except FileNotFoundError:
        return False
    services = doc.get('services') or {}
    return service_name in services


def _compose_service_container_id(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    service_name: str,
) -> str:
    """Return the current container id for a rendered compose service."""
    return _compose_service_state(
        compose_cmd, compose_file, env_file, service_name
    ).get('id', '')


def _compose_service_state(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    service_name: str,
) -> dict[str, str]:
    """Return a robust best-effort state snapshot for a compose service.

    Use JSON ``docker inspect`` rather than a Go template.  The template form is
    brittle for containers without a healthcheck: missing ``State.Health`` can
    make ``docker inspect --format`` fail, causing diagnostics to show only an
    id and the convergence logic to misclassify a still-running LiteLLM
    container as newly reloaded.

    The returned fields are also used to diagnose ``exit code 137`` cases: when
    a container disappears or restarts, ``oom_killed`` / ``exit_code`` make it
    clear whether Docker killed it, Compose recreated it, or the process exited
    normally.
    """
    if not compose_file.exists():
        return {}
    ps_cmd = compose_cmd.split() + [
        '-f',
        str(compose_file),
        '--env-file',
        str(env_file),
        'ps',
        '-q',
        service_name,
    ]
    try:
        proc = subprocess.run(
            ps_cmd, capture_output=True, text=True, check=False, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    container_id = proc.stdout.strip().splitlines()[0]
    inspect_cmd = ['docker', 'inspect', container_id]
    try:
        proc = subprocess.run(
            inspect_cmd, capture_output=True, text=True, check=False, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return {'id': container_id}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {'id': container_id}
    try:
        payload = json.loads(proc.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return {'id': container_id}

    state = payload.get('State') or {}
    health = state.get('Health') or {}
    return {
        'id': payload.get('Id') or container_id,
        'name': str(payload.get('Name') or '').lstrip('/'),
        'running': str(bool(state.get('Running'))).lower(),
        'status': str(state.get('Status') or ''),
        'health': str(health.get('Status') or 'none'),
        'started_at': str(state.get('StartedAt') or ''),
        'finished_at': str(state.get('FinishedAt') or ''),
        'exit_code': str(
            state.get('ExitCode') if state.get('ExitCode') is not None else ''
        ),
        'oom_killed': str(bool(state.get('OOMKilled'))).lower(),
        'restart_count': str(
            payload.get('RestartCount')
            if payload.get('RestartCount') is not None
            else ''
        ),
    }


def _compose_rendered_service_names(compose_file: Path) -> list[str]:
    """Return service names from the rendered compose file."""
    try:
        doc = load_yaml(compose_file)
    except FileNotFoundError:
        return []
    return sorted((doc.get('services') or {}).keys())


def _short_id(value: str) -> str:
    """Shorten a docker container id for human diagnostics."""
    return value[:12] if value else '-'


def _http_probe_summary(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout: float = 8.0,
) -> str:
    """Return a concise one-line summary for a diagnostic HTTP probe."""
    import requests
    try:
        resp = requests.request(
            method, url, headers=headers, json=json_body, timeout=timeout
        )
    except requests.exceptions.RequestException as ex:
        return f'ERR {type(ex).__name__}: {ex}'
    body = (resp.text or '').strip().replace('\n', ' ')
    if len(body) > 220:
        body = body[:220] + '...'
    return (
        f'HTTP {resp.status_code}: {body}'
        if resp.status_code >= 400
        else f'HTTP {resp.status_code}'
    )


def _print_gateway_diagnostics(
    cfg: dict[str, Any],
    deployment: dict[str, Any],
    *,
    model: str | None = None,
    require_generation: bool = False,
) -> None:
    """Print targeted probes for the active gateway/provider graph."""
    env = (
        parse_env_file(runtime_env_path(cfg))
        if runtime_env_path(cfg).exists()
        else {}
    )
    ports = cfg.get('ports', {}) or {}
    gateways = deployment.get('gateways', {}) or {}
    providers = deployment.get('providers', {}) or {}

    litellm = gateways.get('litellm') or {}
    if litellm.get('enabled'):
        litellm_port = ports.get('litellm')
        base = f'http://127.0.0.1:{litellm_port}'
        key = env.get('LITELLM_MASTER_KEY', '')
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        print('\nLiteLLM probes:')
        print(
            f'  GET {base}/model/info -> {_http_probe_summary("GET", base + "/model/info", headers=headers)}'
        )
        print(
            f'  GET {base}/v1/models  -> {_http_probe_summary("GET", base + "/v1/models", headers=headers)}'
        )
        if require_generation:
            probe_model = model or _default_model_for_deployment(deployment)
            protocol = _resolve_smoke_protocol_from_deployment(
                deployment, probe_model
            )
            ok, msg = _ready_openai_probe(
                base_url=f'{base}/v1',
                headers=headers,
                model=probe_model,
                protocol=protocol,
                prompt='Reply with ready.',
                max_tokens=1,
                require_generation=True,
            )
            status = 'OK' if ok else 'WAIT'
            print(
                f'  generation probe ({probe_model}, {protocol}) -> {status}: {msg}'
            )

    vllm = providers.get('vllm') or {}
    runtimes = vllm.get('runtimes') or {}
    if runtimes:
        print('\nvLLM provider probes:')
        for name, rt in runtimes.items():
            service = rt.get('compose_service_name') or f'vllm-{name}'
            host_port = rt.get('host_port') or ports.get('vllm') or 18000
            print(
                f'  {service}: model={rt.get("served_model_name")} protocol={rt.get("protocol_mode")} gpu={rt.get("gpu_indices")}'
            )
            if host_port:
                url = f'http://127.0.0.1:{host_port}/health'
                print(f'    GET {url} -> {_http_probe_summary("GET", url)}')

    ollama = providers.get('ollama') or {}
    if ollama.get('enabled') and ollama.get('publish_port'):
        port = ports.get('ollama') or 11434
        base = f'http://127.0.0.1:{port}'
        print('\nOllama probes:')
        print(
            f'  GET {base}/api/tags -> {_http_probe_summary("GET", base + "/api/tags")}'
        )


def _print_compose_diagnostics(cfg: dict[str, Any], *, tail: int = 0) -> None:
    """Print compose state and optionally recent logs for diagnostic purposes."""
    compose_file = generated_dir(cfg) / 'docker-compose.yml'
    env_file = runtime_env_path(cfg)
    compose_cmd = cfg['runtime']['compose_cmd']
    services = _compose_rendered_service_names(compose_file)
    if not services:
        print(f'No rendered compose services found at {compose_file}')
        return
    print('\nCompose services:')
    for svc in services:
        state = _compose_service_state(compose_cmd, compose_file, env_file, svc)
        if not state:
            print(f'  {svc:22s} absent')
            continue
        print(
            f'  {svc:22s} id={_short_id(state.get("id", ""))} '
            f'name={state.get("name", "-")} '
            f'running={state.get("running", "-")} status={state.get("status", "-")} '
            f'health={state.get("health", "-")} exit={state.get("exit_code", "-")} '
            f'oom={state.get("oom_killed", "-")} restarts={state.get("restart_count", "-")} '
            f'started_at={state.get("started_at", "-")}'
        )
    if tail:
        log_services = [
            svc
            for svc in services
            if svc == 'litellm'
            or svc == 'open-webui'
            or svc.startswith('vllm-')
            or svc == 'ollama'
        ]
        if log_services:
            print(f'\nRecent logs (--tail {tail}):')
            cmd = _compose_base_cmd(cfg) + [
                'logs',
                '--tail',
                str(tail),
                *log_services,
            ]
            subprocess.run(cmd, check=False)


def _explain_readiness_message(msg: str) -> str:
    """Add operator-facing interpretation to common readiness failures."""
    lower = msg.lower()
    if (
        'cannot connect to host litellm' in lower
        or 'name or service not known' in lower
    ):
        return (
            msg
            + '\n  hint: the frontend or caller cannot resolve/reach the LiteLLM service. '
            'Run `infer-stack diagnose --logs --tail 80` to check whether the active profile renders LiteLLM and whether the container is running. '
            'If diagnose shows `oom=true` or `exit=137`, Docker killed LiteLLM rather than merely waiting on a vLLM upstream.'
        )
    if (
        'connection error' in lower
        or 'connection refused' in lower
        or 'cannot connect to host vllm' in lower
    ):
        return (
            msg
            + '\n  hint: LiteLLM is responding, but the upstream vLLM process is not serving yet. '
            'This is expected while a single vLLM runtime is being replaced; wait-ready will keep polling.'
        )
    return msg


class RouterRefreshError(RuntimeError):
    """Live LiteLLM router refresh did not complete; caller should fall back."""


def _resolve_env_refs(obj: Any, env: dict[str, str]) -> Any:
    """Recursively replace ``os.environ/VAR`` strings with the value from ``env``.

    Mirrors LiteLLM's YAML-load substitution so we can feed the admin API
    literal credentials. If a referenced variable isn't in ``env`` the original
    string is left alone — caller will get the upstream error to debug.
    """
    if isinstance(obj, str):
        if obj.startswith('os.environ/'):
            var = obj.removeprefix('os.environ/')
            return env.get(var, obj)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_refs(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_refs(v, env) for v in obj]
    return obj


def _litellm_delete_missed_config_model(resp: requests.Response) -> bool:
    """Return true for LiteLLM's config-model delete miss response.

    ``GET /model/info`` returns both models loaded from config.yaml and
    models inserted into LiteLLM's DB via ``/model/new``.  In current LiteLLM
    releases, ``POST /model/delete`` only deletes DB-backed models.  When the
    reported id belongs to a config-backed model, the delete endpoint returns a
    400/404 response whose body says the model id was not found in the DB.
    That is not a proxy availability failure, so it should not trigger the
    fallback path that restarts the LiteLLM container.
    """
    try:
        payload = resp.json()
    except ValueError:
        payload = resp.text
    text = str(payload).lower()
    return 'not found' in text and 'db' in text


def _litellm_refresh_router_live(cfg: dict[str, Any]) -> None:
    """Sync the running LiteLLM router's model list to match the rendered YAML.

    Diffs ``GET /model/info`` (current state in the running container) against
    the rendered ``litellm_config.yaml`` (desired state), then applies
    ``POST /model/delete`` and ``POST /model/new`` for the differences. Aliases
    that didn't change keep serving traffic without interruption.

    Raises ``RouterRefreshError`` on any failure (admin API unreachable,
    auth missing, individual call fails); the caller falls back to the
    full container-recreate path.
    """
    import requests
    import yaml as _yaml

    litellm_port = cfg.get('ports', {}).get('litellm')
    if not litellm_port:
        raise RouterRefreshError('litellm port not configured in cfg')
    base = f'http://127.0.0.1:{litellm_port}'

    env = parse_env_file(runtime_env_path(cfg))
    master_key = env.get('LITELLM_MASTER_KEY', '').strip()
    if not master_key:
        raise RouterRefreshError(
            'LITELLM_MASTER_KEY missing from rendered .env'
        )
    headers = {
        'Authorization': f'Bearer {master_key}',
        'Content-Type': 'application/json',
    }

    config_path_ = runtime_litellm_config_path(cfg)
    if not config_path_.exists():
        raise RouterRefreshError(
            f'rendered litellm config not found at {config_path_}'
        )
    try:
        desired_doc = (
            _yaml.safe_load(config_path_.read_text(encoding='utf-8')) or {}
        )
    except _yaml.YAMLError as ex:
        raise RouterRefreshError(
            f'could not parse {config_path_}: {ex}'
        ) from ex
    desired_models = desired_doc.get('model_list') or []

    # The rendered YAML keeps secrets as `os.environ/VAR` references so the
    # file itself isn't sensitive. LiteLLM resolves these only at YAML-load
    # time on container startup — the admin API takes literal values. Inline
    # the actual env values now so /model/new gets a usable upstream.
    desired_models = [_resolve_env_refs(m, env) for m in desired_models]

    last_ex: requests.exceptions.RequestException | None = None
    resp = None
    max_attempts = 20
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                f'{base}/model/info', headers=headers, timeout=5
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as ex:
            last_ex = ex
            if attempt < max_attempts:
                import time

                time.sleep(1.5)
                continue
            raise RouterRefreshError(
                f'GET /model/info failed after {attempt} attempts: {ex}'
            ) from ex
    assert resp is not None
    current_models = resp.json().get('data') or []

    # Key by alias (model_name). Within an alias, the "upstream" identity is
    # litellm_params.model (e.g. "openai/qwen3.5-9b"). If that changes, the
    # alias points to a different service and must be re-added; if it matches,
    # the alias is untouched and continues serving.
    def upstream_of(entry):
        return (entry.get('litellm_params') or {}).get('model')

    current_by_alias = {m['model_name']: m for m in current_models}
    desired_by_alias = {m['model_name']: m for m in desired_models}

    to_delete: list[tuple[str, str]] = []
    to_add: list[dict] = []

    for alias, current in current_by_alias.items():
        desired = desired_by_alias.get(alias)
        if desired is None:
            to_delete.append((alias, current['model_info']['id']))
        elif upstream_of(current) != upstream_of(desired):
            to_delete.append((alias, current['model_info']['id']))
            to_add.append(desired)

    for alias, desired in desired_by_alias.items():
        if alias not in current_by_alias:
            to_add.append(desired)

    if not to_delete and not to_add:
        return

    # Delete-before-add so the same alias can transition to a new upstream
    # without LiteLLM rejecting a duplicate model_name.  LiteLLM distinguishes
    # config-file models from DB-backed models: /model/delete only applies to
    # DB-backed rows.  When an existing model was loaded from config.yaml,
    # LiteLLM may report it in /model/info but return "not found in db" from
    # /model/delete.  Treat that as a non-fatal stale-config alias instead of
    # forcing a LiteLLM container restart; the desired new aliases can still be
    # added live, and a later manual LiteLLM restart will clean up the stale
    # config-backed aliases if the operator cares about /v1/models hygiene.
    stale_config_aliases: set[str] = set()
    deleted_count = 0
    for alias, model_id in to_delete:
        try:
            resp = requests.post(
                f'{base}/model/delete',
                headers=headers,
                json={'id': model_id},
                timeout=5,
            )
            if resp.status_code in {
                400,
                404,
            } and _litellm_delete_missed_config_model(resp):
                stale_config_aliases.add(alias)
                continue
            resp.raise_for_status()
            deleted_count += 1
        except requests.exceptions.RequestException as ex:
            raise RouterRefreshError(
                f'DELETE alias={alias} id={model_id} failed: {ex}'
            ) from ex

    skipped_add_aliases: set[str] = set()
    for model in to_add:
        alias = model.get('model_name', '<unknown>')
        if alias in stale_config_aliases:
            # Same alias, changed upstream, and the old alias is config-backed.
            # Adding would collide and deleting would require a container restart.
            skipped_add_aliases.add(alias)
            continue
        try:
            resp = requests.post(
                f'{base}/model/new',
                headers=headers,
                json=model,
                timeout=5,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as ex:
            raise RouterRefreshError(
                f'POST /model/new alias={alias} failed: {ex}'
            ) from ex

    summary_parts = []
    if deleted_count:
        summary_parts.append(f'removed {deleted_count} alias(es)')
    added_count = len(to_add) - len(skipped_add_aliases)
    if added_count:
        summary_parts.append(f'added {added_count} alias(es)')
    if stale_config_aliases:
        summary_parts.append(
            'left '
            f'{len(stale_config_aliases)} stale config-backed alias(es) live '
            'because LiteLLM would not delete them without a restart'
        )
    if skipped_add_aliases:
        summary_parts.append(
            'skipped '
            f'{len(skipped_add_aliases)} same-name update(s); '
            'restart LiteLLM to replace those aliases'
        )
    print(f'Live LiteLLM router refresh: {", ".join(summary_parts)}.')


def _preflight_check_ports(cfg: dict[str, Any]) -> None:
    """Verify only the host ports the current rendered stack will publish."""
    ports = cfg.get('ports', {})
    candidates: list[tuple[str, int, str]] = []
    deployment: dict[str, Any] = {}
    try:
        if plan_path(cfg).exists():
            deployment = load_yaml(plan_path(cfg)).get('deployment', {})
    except Exception:
        deployment = {}

    frontends = deployment.get('frontends', {}) or {}
    gateways = deployment.get('gateways', {}) or {}
    providers = deployment.get('providers', {}) or {}

    if not deployment:
        # Fallback for very old rendered states; keep this conservative.
        if ports.get('litellm'):
            candidates.append(('litellm', int(ports['litellm']), '0.0.0.0'))
        if ports.get('open_webui'):
            candidates.append(
                ('open-webui', int(ports['open_webui']), '0.0.0.0')
            )
    else:
        if (gateways.get('litellm') or {}).get('enabled') and ports.get(
            'litellm'
        ):
            candidates.append(('litellm', int(ports['litellm']), '0.0.0.0'))
        if (
            (frontends.get('open_webui') or {}).get('enabled')
            and (frontends.get('open_webui') or {}).get('publish_port', True)
            and ports.get('open_webui')
        ):
            candidates.append(
                ('open-webui', int(ports['open_webui']), '0.0.0.0')
            )
        reverse_proxy = frontends.get('reverse_proxy') or {}
        if reverse_proxy.get('enabled'):
            if reverse_proxy.get('publish_http', True):
                candidates.append(
                    (
                        'reverse-proxy-http',
                        int(
                            reverse_proxy.get('http_port')
                            or ports.get('reverse_proxy_http')
                            or 80
                        ),
                        reverse_proxy.get('http_bind_host') or '0.0.0.0',
                    )
                )
            if reverse_proxy.get('publish_https', True):
                candidates.append(
                    (
                        'reverse-proxy-https',
                        int(
                            reverse_proxy.get('https_port')
                            or ports.get('reverse_proxy_https')
                            or 443
                        ),
                        reverse_proxy.get('https_bind_host') or '0.0.0.0',
                    )
                )
        ollama = providers.get('ollama') or {}
        if (
            ollama.get('enabled')
            and ollama.get('publish_port')
            and ports.get('ollama')
        ):
            candidates.append(('ollama', int(ports['ollama']), '127.0.0.1'))
        for name, rt in (
            (providers.get('vllm') or {}).get('runtimes') or {}
        ).items():
            if rt.get('publish_port'):
                candidates.append(
                    (
                        f'vllm-{name}',
                        int(rt.get('host_port') or 18000),
                        '127.0.0.1',
                    )
                )

    owned = our_published_ports(
        cfg['runtime']['compose_cmd'],
        generated_dir(cfg) / 'docker-compose.yml',
        runtime_env_path(cfg),
    )
    to_check = [
        (svc, port, host) for svc, port, host in candidates if port not in owned
    ]

    try:
        check_ports_available(to_check)
    except PortInUseError as ex:
        raise SystemExit(str(ex)) from ex
