# Compose recipe: Qwen3.6-35B-A3B on a 4x96GB host

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../README.md).

This is the shortest end-to-end example for serving **Qwen/Qwen3.6-35B-A3B** on a machine with **4 x 96GB GPUs** using the **Compose** backend.

This profile uses:

- two model instances
- GPUs 0 and 1 for one instance
- GPUs 2 and 3 for one instance
- tensor parallel size 2 per instance
- text-only mode
- native **262,144** token context
- the Qwen reasoning parser
- higher batching and concurrency for more users
- Open WebUI on top of LiteLLM

---

## Quick path: use the built-in profile

This deployment is available out of the box as the built-in profile
`qwen3.6-35b-a3b-dual-tp2-4x96`. To skip writing any local YAML:

```bash
infer-stack setup --backend compose --profile qwen3.6-35b-a3b-dual-tp2-4x96
infer-stack render --yes
infer-stack up -d
```

The rest of this recipe walks through the same deployment manually, so you
can see how the model and profile YAML map to the rendered Compose stack.

---

## 1. Start from the repo root

```bash
cd /path/to/infer_stack
```

---

## 2. Initialize Compose config

```bash
python manage.py setup --backend compose
```

---

## 3. Write the local model and profile definitions

Write provider-specific model definitions and a stack profile in `models.yaml`:

```bash
cat > models.yaml <<'EOF'
vllm_models:
  qwen3.6-35b-a3b-local:
    hf_model_id: Qwen/Qwen3.6-35B-A3B
    tokenizer_name: Qwen/Qwen3.6-35B-A3B
    served_model_name: qwen3.6-35b-a3b
    family: qwen3.6
    modalities: [text]
    memory_class_gib: 80
    min_vram_gib_per_replica: 24
    preferred_gpu_count: 2
    context_window: 262144
    defaults:
      max_model_len: 262144
      gpu_memory_utilization: 0.95
      enable_prefix_caching: false
      max_num_batched_tokens: 8192
      max_num_seqs: 8

profiles:
  qwen3.6-35b-a3b-dual-tp2-262k-local:
    description: "Two Qwen3.6-35B-A3B vLLM runtimes across 4 GPUs at 262k context."
    vllm:
      enable_responses_api_store: false
      logging_level: INFO
    providers:
      vllm:
        runtimes:
          qwen36-35b-gpu01:
            model: qwen3.6-35b-a3b-local
            served_model_name: qwen3.6-35b-a3b-gpu01
            placement:
              strategy: exact
              gpu_indices: [0, 1]
            topology:
              tensor_parallel_size: 2
            runtime:
              max_model_len: 262144
              gpu_memory_utilization: 0.95
              max_num_batched_tokens: 8192
              max_num_seqs: 8
              enable_prefix_caching: false
            extra_args:
              - --language-model-only
              - --reasoning-parser
              - qwen3

          qwen36-35b-gpu23:
            model: qwen3.6-35b-a3b-local
            served_model_name: qwen3.6-35b-a3b-gpu23
            placement:
              strategy: exact
              gpu_indices: [2, 3]
            topology:
              tensor_parallel_size: 2
            runtime:
              max_model_len: 262144
              gpu_memory_utilization: 0.95
              max_num_batched_tokens: 8192
              max_num_seqs: 8
              enable_prefix_caching: false
            extra_args:
              - --language-model-only
              - --reasoning-parser
              - qwen3
    gateways:
      litellm:
        enabled: true
    frontends:
      open_webui:
        enabled: true
        provider: litellm
    routes:
      qwen3.6-35b-a3b-gpu01:
        provider: vllm
        runtime: qwen36-35b-gpu01
      qwen3.6-35b-a3b-gpu23:
        provider: vllm
        runtime: qwen36-35b-gpu23
EOF
```

---

## 4. Switch to the local profile

```bash
python manage.py switch qwen3.6-35b-a3b-dual-tp2-262k-local
```

Optional sanity check before rendering:

```bash
python manage.py describe-profile qwen3.6-35b-a3b-dual-tp2-262k-local --format yaml
```

---

## 5. Render the stack

```bash
python manage.py render
```

This creates:

- `generated/docker-compose.yml`
- `generated/.env`

---

## 6. Set your Hugging Face token

Add your token to `generated/.env`:

```bash
grep '^HF_TOKEN=' generated/.env
```

Then edit the file and set:

```text
HF_TOKEN=your_token_here
```

---

## 7. Start the stack

From the repo root:

```bash
infer-stack up -d
```

---

## 8. Verify the backend

Check that both model endpoints are exposed:

```bash
infer-stack smoke-test
```

You should see:

- `qwen3.6-35b-a3b-gpu01`
- `qwen3.6-35b-a3b-gpu23`

---

## 9. Open the UI

Open:

```text
http://127.0.0.1:13000
```

On first run, create the Open WebUI admin account.

Then select one of the models:

- `qwen3.6-35b-a3b-gpu01`
- `qwen3.6-35b-a3b-gpu23`

---

## 10. Optional repo smoke tests

From the repo root:

```bash
python manage.py smoke-test   --model qwen3.6-35b-a3b-gpu01   --prompt "Say hello in one sentence."

python manage.py smoke-test   --model qwen3.6-35b-a3b-gpu23   --prompt "Say hello in one sentence."
```

---

## Summary

This profile serves:

- `Qwen/Qwen3.6-35B-A3B`
- as two TP2 instances across all 4 x 96GB GPUs
- at 262,144 token context
- with `max_num_batched_tokens: 8192`
- with `max_num_seqs: 8`
- with `--reasoning-parser qwen3`
- through both the direct backend and Open WebUI

Note: with the current repo templates, each backend instance is exposed as its own LiteLLM model name.
