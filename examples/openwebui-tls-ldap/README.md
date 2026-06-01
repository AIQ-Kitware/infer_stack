# Open WebUI with TLS reverse proxy and LDAP

This example runs the built-in `openwebui-tls-ldap` profile: an opt-in
Compose stack with

- Ollama as the provider,
- Open WebUI connected directly to Ollama,
- Open WebUI LDAP login wiring,
- nginx as a TLS reverse proxy, and
- no public Ollama or Open WebUI host ports (only the proxy is exposed).

The reverse proxy and LDAP settings are ordinary config fields. If the
typed nginx renderer is not enough, set
`frontends.reverse_proxy.config_path` to an existing nginx config and
infer-stack mounts that file instead of rendering `state.runtime/nginx.conf`.

## Setup

```bash
infer-stack setup --backend compose --profile openwebui-tls-ldap
```

Then edit `config.yaml` and/or the generated `.env` after the first render:

```bash
infer-stack render --yes --simulate-hardware 2x24
```

The generated `.env` contains LDAP placeholders such as `LDAP_HOST`,
`LDAP_PASSWD`, and `LDAP_SEARCH_BASE`. Fill those in before starting the
stack.

## TLS certificates

Cert/key/dhparam host paths live under `frontends.reverse_proxy.ssl`:

```yaml
frontends:
  reverse_proxy:
    ssl:
      certificate: ./certs/site.crt
      certificate_key: ./certs/site.key
      dhparam: ./dhparam.pem
```

> **Relative paths resolve against the generated directory.** These
> values are written verbatim into the generated `docker-compose.yml`, so
> Docker resolves them relative to that file's location — not your
> current working directory. Either place `certs/` and `dhparam.pem`
> alongside the generated compose file, or use absolute paths.
> `infer-stack render` prints a warning when a referenced cert or config
> file cannot be found.

## Fully manual nginx config

```yaml
frontends:
  reverse_proxy:
    enabled: true
    config_path: /absolute/path/to/nginx.conf
```
