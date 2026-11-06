# HyperQwen reasoning effort through LiteLLM and pi

Qwen3.8's chat template distinguishes **`low`**, **`medium`**, and **`xhigh`**.
`xhigh` is the model template's default if no effort is sent. `medium` inserts
no extra effort instruction, while `low` and `xhigh` insert different
instructions. These settings shape reasoning from the beginning; they are
**not** token-budget cutoffs. `minimal`, `high`, and `max` are *not* native
values in the stock Qwen3.8 template. A client should not silently map them.

## Catalog and gateway

An endpoint opts in by declaring the template's actually supported levels:

```yaml
endpoints:
  qwen3.8-27b-dbirks-hyperqwen:
    engine: vllm
    model: qwen3.8-27b-dbirks-hyperqwen
    reasoning_efforts: [low, medium, xhigh]
    # ...your existing runtime/placement settings...
```

`infer-stack catalog suggest` supplies this declaration for the standard
HyperQwen endpoint and its hardware-specific variants. **Existing catalogs
are not silently rewritten**; add the line to each existing HyperQwen endpoint
if its declaration predates this feature.

The metadata follows the endpoint alias rather than the GPU deployment: it
survives static route rendering, dynamic LiteLLM route publication, and
coalesced deployments. LiteLLM receives:

```yaml
litellm_params:
  allowed_openai_params: [reasoning_effort]
model_info:
  supports_reasoning: true
  reasoning_effort_levels: [low, medium, xhigh]
```

This lets LiteLLM forward the field for a custom model name absent from its
built-in model map, and lets pi's LiteLLM provider discover supported levels.
The actual per-request parameter is OpenAI-compatible, so no
`chat_template_kwargs` conversion or server default is necessary:

```json
{"model":"qwen3.8-27b-dbirks-hyperqwen","messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"medium"}
```

Changes to reasoning effort between client turns **do not restart the engine**.
The per-endpoint capability declaration is not a new vLLM launcher argument,
so no extra vLLM compile-cache variant is created.

## pi

Install both pi packages in the client environment:

```bash
pi install npm:pi-provider-litellm
pi install npm:pi-effort
```

The current `pi-provider-litellm` requires Pi >=0.99.2 (older installations
may use `pi-provider-litellm@3.4.0`, but this route-level reasoning metadata
has not been checked with the older provider). Configure LiteLLM credentials
as normal (`/login litellm`, or your existing infer-stack/pi bridge). Then:

```
/litellm-refresh
/model
/effort
/effort medium
/effort xhigh
/effort low
```

The bare `/effort` opens a picker. These commands affect **subsequent
requests**, not the currently running request. `pi-effort` also has `/effort
default`; on HyperQwen that should leave the effort unspecified, letting the
chat template pick `xhigh`. Do not claim `off`, `minimal`, or `high` work
through this endpoint without a separately verified template adaptation.

## End-to-end smoke test on a running stack

Use the host's actual LiteLLM URL and an acquired HyperQwen endpoint:

```bash
export OPENAI_API_KEY="$(infer-stack env LITELLM_MASTER_KEY)"
export OPENAI_BASE_URL="$(infer-stack env OPENAI_BASE_URL)"
curl -fsS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-dbirks-hyperqwen","messages":[{"role":"user","content":"Reply with OK"}],"reasoning_effort":"medium","max_tokens":128}'
```

Repeat with `low` and `xhigh`, and check LiteLLM `/model/info` for the
capability metadata. A successful completion establishes forwarding; it does
not, by itself, quantify model quality or reasoning-token usage.

Sources: [HyperQwen gotchas](https://github.com/syv-ai/HyperQwen/blob/main/docs/gotchas.md),
[Qwen3.8 effort discussion](https://github.com/QwenLM/Qwen3.8/issues/217),
[pi-provider-litellm](https://pi.dev/packages/pi-provider-litellm),
[pi-effort](https://pi.dev/packages/pi-effort),
[vLLM reasoning outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/).
