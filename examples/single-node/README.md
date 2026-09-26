# Single-node KubeAI example

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../../README.md).

This example uses the stack-graph schema, but KubeAI currently renders only the
`providers.vllm.runtimes` section. Ollama, LiteLLM, and Open WebUI are disabled
in this example because they are Compose-only for now.

Copy these files into the repo root:

```bash
cp examples/single-node/config.yaml ./config.yaml
cp examples/single-node/models.yaml ./models.yaml
```

Then edit:

- `resource_profiles.blackwell-96gb.node_selector.kubernetes.io/hostname`
- any Hugging Face model ids you want to use
- `cluster.ingress.host` if you want an Ingress on your LAN

Typical flow:

```bash
python manage.py render
bash scripts/bootstrap_k3s.sh
python manage.py deploy
python manage.py status
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

If you are not using ingress yet, port-forward first:

```bash
kubectl -n kubeai port-forward svc/kubeai 8000:80
```

Or, from repo root:

```bash
make example-single-node
make bootstrap-k3s
python manage.py render
python manage.py deploy
```
