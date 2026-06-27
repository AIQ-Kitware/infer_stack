# Lessons

Confirmed, reusable, non-obvious lessons only (see `AGENTS.md`). Each cites
evidence; prefer append-only; supersede incorrect entries with a new one.

---

- **Lesson:** LiteLLM's runtime model-management admin API (`/model/new`,
  `/model/delete`, `/model/update`) is **not** DB-less. It returns HTTP 500
  unless **both** a database is connected (`prisma_client is not None`) **and**
  `STORE_MODEL_IN_DB=true`. There is no in-memory runtime model-add — "dynamic
  routing via the admin API" therefore *requires* Postgres, it is not an
  optional durability upgrade.
  - **Evidence:** `litellm v1.82.3` source,
    `litellm/proxy/management_endpoints/model_management_endpoints.py::add_new_model`
    (the `prisma_client is None` → 500 and the `store_model_in_db is True else …
    "Set STORE_MODEL_IN_DB=True"` → 500 guards); same guards on `delete_model`
    /`update_model`. `STORE_MODEL_IN_DB` + `DATABASE_URL` are read from the env
    (`proxy_server.py`). Used by `infer_stack.leasing.compose` dynamic routing.
  - **Applies when:** designing/maintaining runtime LiteLLM route management;
    deciding whether a Postgres dependency is avoidable (it isn't).

- **Lesson:** In a declarative compose project that is re-rendered whole on
  every change, any per-service field derived from the *live set* (not from the
  service's own identity) silently churns survivors. A vLLM/ollama host port
  assigned by enumeration index (`BASE + i` over the sorted live deployments)
  meant removing/adding one deployment renumbered every later one's published
  port → changed its service spec → `docker compose up -d` recreated unrelated,
  in-flight containers. The recreate is invisible in the converge log (it just
  says "up -d N services"); only a GPU-memory timeline showed all upstreams
  dropping at once. Rule: a rendered service spec must depend **only** on that
  deployment (its id, spec, and sticky placement), never on its neighbours —
  that is what makes "re-render all, recreate only what changed" hold. Fix:
  upstreams behind the gateway are internal (compose-network DNS), so they
  publish no host port at all → nothing set-dependent left to churn.
  - **Evidence:** slurm-e2e run `20260627T080504-slurm`: `vllm-smol-135-5e2dcadf`
    (GPU 3) loaded 08:10:51, then an unrelated `gpt2` release at 08:12:26
    renumbered ports and `docker compose up -d` recreated it at 08:12:38; the
    probe got `litellm.InternalServerError: Connection error … Model
    Group=smol-135` until the budget ran out. Fixed in
    `compose.render_compose` / `_vllm_service` / `_ollama_service`; guarded by
    `test_dynamic_upstreams_have_no_host_ports_and_survive_set_change`.
  - **Applies when:** rendering any declarative project (compose/k8s) from a
    mutable working set; auditing for "why did an unrelated thing restart?".

- **Lesson:** `/model/new` honors a **caller-supplied** `model_info.id`, and
  `GET /v1/model/info` returns each model's `model_info.id`. So a reconcile can
  assign each managed route a *deterministic* id and become a pure set-diff
  (add ids not present, delete ids no longer desired) — idempotent, drift-safe,
  and able to leave hand-added models alone by id prefix.
  - **Evidence:** `litellm v1.82.3` `_add_model_to_db` (`if
    model_params.model_info.id is not None: _data["model_id"] = …`) and
    `model_info_v1` response shape (`data[].model_info.id`). Implemented in
    `ComposeBackend._reconcile_routes` (ids prefixed `isr-`).
  - **Applies when:** building idempotent reconcile over LiteLLM's model store.
