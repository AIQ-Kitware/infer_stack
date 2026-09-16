# Investigation: the gateway routes a model's traffic to a different container after recreation

- **Date:** 2026-09-16
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`. The root cause was
  found by a second agent operating the affected run. This author re-verified its
  evidence and wrote this report.
- **Code examined:** `dev/0.7.1` at `7be95ba`; gateway image
  `ghcr.io/berriai/litellm:v1.82.3-stable` (`PINNED_IMAGES`, `config.py`).
- **Status:** investigation only. **No code has been changed.**
- **Related:**
  [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
  and
  [`plan-keep-warm-admission-2026-09-16.md`](plan-keep-warm-admission-2026-09-16.md);
  the final section explains why this bug constrains that plan.

Claims are tagged **[code]**, **[observed]** or **[inferred]** as in the related
investigation.

---

## Summary

A model's container was removed and immediately recreated, with a different
model's container created alongside it. For the next 28 minutes every request
the LiteLLM gateway sent to the recreated model reached **a vLLM server serving a
different model**. vLLM answered each one with
``404 The model `<name>` does not exist``. Readiness probes go through the
gateway, so a model that was running correctly never became ready, and two batch
jobs failed at their 30-minute timeout.

The best-supported mechanism is at the network layer. The gateway addresses each
upstream by its Compose DNS name. The recreated **other** model received the
removed container's IP address. The gateway kept using the old name→IP mapping,
held open by a DNS cache and by continuous keep-alive traffic. A rerun after the
cache had expired came up normally in about two minutes.

---

## The incident

Labels are used instead of model names:

- **A:** a 27B model shared as an auxiliary endpoint, service `vllm-<A>`;
- **B:** a 2B model, service `vllm-<B>`;
- **C:** a 4B model.

Times are local (UTC−4); the gateway log is in UTC.

| time | event | tag |
|---|---|---|
| 15:46:34 | Jobs using A and C finish; their leases release. A's and C's reclaim policy is `stop`. | [observed] |
| 15:46:35 | A's container (GPU 2) and C's container are removed. | [observed] |
| 15:46:39 | A new job's apply: B's container is created, then A's container is created. | [observed] |
| 19:45Z–19:47Z | Gateway requests for A: `Connection error`. | [observed] |
| 19:48:38Z → 20:16Z | Every gateway request for A: vLLM 404 ``The model `A` does not exist``, about 92–96 per minute for 28 minutes. | [observed, re-verified by count] |
| ~15:48:38 | About when B's server would start listening (its startup takes ~1:48). | [inferred from B's startup time] |
| 16:16:42 | Both jobs fail: `run: endpoints not ready: [(..., 'A')]`, the 1800 s readiness timeout. | [observed] |
| 16:16:44 | Rollback stops A's container: **exit code 0**. It was running normally, not crash-looping. | [observed] |
| 16:51 | Rerun, long after any cache expiry: A and B are both ready in ~2 minutes; the jobs complete. | [observed] |

**Why the 404 is decisive.** vLLM returns ``The model `X` does not exist`` when
the requested model name is not what that server serves. A's own container
serves `A`, and it was running. So the requests addressed to `vllm-<A>` were not
reaching A's container. They were reaching another vLLM server, and B's was the
only one started at that moment.

---

## Mechanism

### M1. Upstreams are addressed by Compose DNS name  [code]

Routes are rendered with `api_base = http://{vllm_service_name(deployment)}:8000/v1`
(`compose.py:471`). A and B have distinct service names, so the misroute cannot be
a name collision. It must happen at the address level.

### M2. Readiness is judged through the gateway  [code]

`ComposeBackend.probe_ready` probes via `http://127.0.0.1:{litellm_port}/v1`
when LiteLLM is enabled (see the `probe_ready` docstring and body). A misrouted
gateway therefore reports a healthy model as not ready. It also misroutes real
traffic, not only probes.

### M3. The gateway reused a stale name→IP mapping  [inferred]

The proposed chain:

1. A's old container is removed; its IP returns to the network's pool.
2. B's container, created first, receives that IP.
3. The gateway's HTTP client still maps `vllm-<A>` to the old IP. It sends A's
   requests there, and B answers 404 once it is listening (19:48:38Z).
4. Requests for A arrive every ~2 s, so a keep-alive connection to that IP never
   goes idle long enough to close, and the name is never re-resolved.

**Reported by the discovering agent, not verified here** (LiteLLM is not
installed on the guest): LiteLLM v1.82.3's aiohttp transport uses
`AIOHTTP_TTL_DNS_CACHE = 300` s and `AIOHTTP_KEEPALIVE_TIMEOUT = 120` s
(`litellm/constants.py:204-205`). See V1.

**Consistent with, not proven by, the evidence:** the rerun succeeded after more
than 300 s; the 404s start when B could first answer; the pre-404 errors are
connection errors, when nothing was listening at the old IP yet.

**Not established:** the actual IPs, since the old container's address is no
longer recoverable. Other layers could also cache (Docker's embedded DNS, a
connection pool independent of DNS).

### M4. `reclaim=stop` on a shared model maximises the exposure  [code] + [inferred]

With per-job leases, a shared `stop` model is removed at the end of each job and
recreated at the start of the next, often seconds apart and alongside other
containers being created. That is exactly the window for IP reuse.

---

## Prior incident that fits the pattern

`scripts/set_endpoint_reclaim.sh`, in a downstream harness, documents a 16-job
run where a shared auxiliary model was "created 9 times and destroyed 8", and two
jobs timed out with `endpoints not ready` for it. At the time this was attributed
to a concurrent teardown, and the remedy was to make that model `keep-warm`.

**[inferred]** That failure is consistent with this misroute. It is not proven,
because no gateway log from it was examined. Note the interaction: the
`keep-warm` workaround for *this* bug is what produces the placement starvation
in the related investigation.

---

## Candidate directions: not a plan

| # | direction | trade-off |
|---|---|---|
| 1 | **Lower or disable the gateway client's DNS cache**, via environment on the LiteLLM service or a config setting, if the pinned version exposes one. Every *new* connection then resolves the current IP. | Cheap. Does not help if a keep-alive connection is held to a reused IP, though a removed container's connections are reset, which forces a reconnect. Needs V1 and V2. |
| 2 | **Stable per-service IPs:** a fixed `ipv4_address` for each model service on the Compose network, derived deterministically from the service, so recreation keeps its address and no other service can take it. | Removes the class at the network layer, with no gateway config change. Needs subnet management and collision-free assignment. |
| 3 | **Per-instance hostnames/aliases** that change on each recreation. | Correct by construction, but changes `api_base` on every recreation. That breaks static-superset byte-stability, where the gateway must never be recreated, unless routes are reconciled through the admin API. |
| 4 | **Probe the upstream directly as well** (`GET /v1/models` on the service, checking the served name) and treat a gateway/upstream disagreement as a routing fault. | Detects the fault and names it, instead of a silent 30-minute timeout. Does not fix misrouted *traffic* on its own. |
| 5 | **Drain the gateway's connections to a service when its container is removed**, for example by re-registering the route through the admin API. | Depends on LiteLLM internals: does re-registering a model recreate its HTTP session? Unverified. |

(4) is worth doing regardless, because it turns a silent timeout into a named
fault. (1) or (2) addresses the cause.

---

## Verification (host)

- **V1:** `docker exec infer-stack-litellm-1 python -c "import litellm.constants as c; print(c.AIOHTTP_TTL_DNS_CACHE, c.AIOHTTP_KEEPALIVE_TIMEOUT)"`
  confirms the reported values in the pinned image.
- **V2:** whether those values are configurable through the environment, or
  through a LiteLLM setting, without patching the image.
- **V3 (deliberate reproduction):** serve two small models X and Y through the
  gateway; probe X every 2 s; remove X, create Y, then recreate X within a few
  seconds; check whether X's requests return Y's 404. Repeat with direction (1)
  or (2) applied.
- **V4:** record container IPs (`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'`)
  before and after recreation during V3, to turn M3 from inferred into observed.

---

## Open questions

- **Q1.** Is the stale mapping held by aiohttp's DNS cache, by a pooled
  keep-alive connection, or by both? This decides between direction 1 and
  direction 2.
- **Q2.** Can a *successful* misroute happen? That needs two deployments that
  serve the same model name on different containers, e.g. dedicated copies. Then
  the wrong container would answer 200, and the fault would be invisible even to
  direction 4.
- **Q3.** Does Ollama traffic (`compose.py` Ollama routes) share the exposure?
- **Q4.** Should `reclaim=stop` on a deployment shared by several short leases be
  discouraged, given M4 and the placement plan's handoff barrier (next section)?

---

## Interaction with the keep-warm admission plan

The plan (revision 2) makes this bug **more likely** unless it is addressed
first:

- The **handoff barrier** (plan I10, §4.6) stops and removes a displaced
  container, then starts another deployment on its GPU, seconds apart and within
  one apply. That is the remove-then-create sequence that frees an IP for reuse.
- **Selective apply** creates new containers individually, with no project-wide
  recreation, so IP reuse across *different* services is the normal case.
- **Readiness is the admission signal.** A misrouted probe makes an admitted,
  healthy deployment time out. Under the plan's rules that releases its lease
  through the strict path, which is a correct outcome for a wrong reason.

**Proposed addition to the plan:** an invariant that *a request routed for
deployment D reaches only D's container*, established before the barrier (plan
P6) can run, together with direction (4) as the detector. The addendum in the
plan records this as D12.
