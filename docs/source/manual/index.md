# User manual

Task-oriented guides for running `infer-stack`'s leasing/controller stack.

For installation and the first endpoint, start with the
[project quickstart](https://github.com/AIQ-Kitware/infer_stack/blob/main/README.md#quickstart).
The [repository documentation index](https://github.com/AIQ-Kitware/infer_stack/blob/main/docs/index.md)
links to the current model registry, catalog, operations, benchmarks and backend
guides. The worked tutorials below are also included in this built manual.

```{toctree}
:maxdepth: 1

ollama-openwebui-tutorial
leasing-demo
```

- **[Ollama + Open WebUI tutorial](ollama-openwebui-tutorial.md)** — stand up a
  self-managing Ollama server with Open WebUI from the CLI (GPU-pinned; worked
  example on a dual GTX 1080 Ti box).
- **[Leasing demo](leasing-demo.md)** — the happy-path walkthrough: configure
  storage once, deploy a chat model as a standing service, and talk to it
  through Open WebUI.
