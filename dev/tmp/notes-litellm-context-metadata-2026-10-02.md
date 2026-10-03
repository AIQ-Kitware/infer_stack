# Working notes — LiteLLM `model_info.max_input_tokens` from effective `max_model_len`

Task: for every infer-stack-managed vLLM endpoint, advertise the same effective
`max_model_len` to LiteLLM as the one used to launch vLLM:

```text
effective vLLM max_model_len == LiteLLM model_info.max_input_tokens
```

## Where things are today (verified in tree, branch dev/0.7.2)

1. **Effective vLLM max model length is computed in**
   `infer_stack/leasing/compose.py::vllm_service_dict()` (line ~336):
   `runtime = translate_legacy(deployment.spec['runtime'])` then
   `'max_model_len': runtime.get('max_model_len', VLLM_DEFAULTS['max_model_len'])`
   with `VLLM_DEFAULTS` (line 98) = `{'max_model_len': 8192, ...}`.
   The dict feeds `vllm_args()` (`--max-model-len=...`) in BOTH backends:
   compose (`_vllm_service`) and kubeai (`_model_doc` -> `spec.args`).
   So `vllm_service_dict(deployment)['max_model_len']` is the single launch
   value; no other place computes it.

2. **Static gateway entries render in** `gateway.py::render_front_door()`:
   `entries = [r.entry() for r in routes]` -> `litellm_config.yaml model_list`
   (config hash stamped on the service label; a config change recreates the
   gateway once). Routes come from `front_door_routes()` = route_table over
   (registry, catalog_routes, deployment_routes, extra).

3. **Dynamic `/model/new` entries render through the same** `GatewayRoute.entry()`:
   `render_front_door(dynamic_routing=True)` sets `litellm_routes = [r.entry() ...]`
   (written to `litellm_routes.json`); `Gateway._reconcile_routes()` POSTs
   each entry body via `_post_route('/model/new', route, ...)`.

4. **Route semantics compare in** `Gateway._route_semantics(route)`
   (`{'model_name', 'model', 'api_base', 'key_env'}`); desired comes from
   `litellm_routes.json`, current from `_list_managed_routes()` which parses
   `/v1/model/info` and keeps routes whose `model_info.id` starts with `isr-`.
   Drift (same id, different semantics) -> delete + re-add under the id.

5. **Remembered routes live in** `litellm_registry.json`
   (`LITELLM_REGISTRY_FILENAME`, version 1, `entries: {alias: row}`):
   written by `remembered_rows()` (rows: vllm `{'engine','served'}`,
   ollama `{'engine','model','host'}`, upstream `{'engine','served','api_base'}`)
   through `Gateway.remember()`; reconstructed by `registry_route()` /
   `registry_routes()`.

6. **KubeAI gateway routes construct in** `backends/kubeai.py`:
   `catalog_routes()` (per-endpoint -> Model via `_upstream_url()`) and
   `_front_door_inputs()` (static: catalog routes + cluster Models from
   `rendered.request_names`; dynamic: `upstream_route(gid, endpoint, name, base)`
   from `rendered.models`). Both backends' deployments render through
   `vllm_service_dict`, and KubeAI only ever renders vllm deployments.

## Other facts established

- `Deployment.spec['runtime']` is the creating request's raw `runtime` dict
  (ledger.acquire copies `req.spec`); `translate_legacy()` is idempotent.
- Catalog resolver: `spec = {..., 'runtime': dict(rt)}`; `capacity` only gets
  `max_model_len` when explicitly set (catalog.py ~610).
- `GatewayRoute` is a frozen dataclass; all call sites pass at most 4
  positional args, `key_env`/`route_id`/`origin` always by keyword.
- LiteLLM v1.82.3 `/model/info`: config-provided `model_info` wins over its
  bundled cost map; `max_input_tokens` is a first-class `ModelInfo` field;
  `litellm_params.max_tokens` does NOT feed it.
- `VLLM_DEFAULTS` is referenced only inside compose.py (def + vllm_service_dict).

## Plan (smallest coherent patch)

- `launch.py`: move `VLLM_DEFAULTS` here (it is launch defaults; launch.py is
  the shared lowest module, imports nothing -> no cycle for gateway.py);
  add `effective_max_model_len(runtime)` (translate_legacy first, idempotent,
  default from VLLM_DEFAULTS, positive int).
- `compose.py`: import VLLM_DEFAULTS + effective_max_model_len from launch;
  `vllm_service_dict` uses the helper for the launch value (same number as
  before, one source of truth).
- `routes.py`: `GatewayRoute.max_input_tokens: int | None = None` (last
  field, compared); `entry()` emits `model_info.max_input_tokens` alongside
  existing id / key_env metadata, creating `model_info` when needed; never
  null, never under litellm_params, never max_output_tokens.
- `gateway.py`:
  - `compose_catalog_route` vllm branch: `effective_max_model_len(request.spec['runtime'])`
  - `deployment_routes` vllm branch: same from `deployment.spec['runtime']`
  - `upstream_route` gains optional `max_input_tokens` kwarg
  - `remembered_rows`: vllm rows remember the value; upstream rows carry it
    when the route has it
  - `registry_route`: reads the remembered value (guarded: positive int or
    None; a legacy row without it reconstructs with no context — never
    invented)
  - `_route_semantics`: add `model_info.max_input_tokens` (absent==absent)
- `backends/kubeai.py`: `catalog_routes().managed` and `_front_door_inputs`
  (both branches) feed the value through.
- Tests A-I in `tests/test_leasing_context_metadata.py` (+ extend the
  existing `_route_semantics` test file's conventions where natural).
