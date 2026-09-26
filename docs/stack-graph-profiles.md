# Stack graph profiles (removed)

Named stack profiles (`providers` / `gateways` / `frontends` / `routes`,
selected with `--profile` or `active_profile`, applied with `setup`, `switch`
and `up`) no longer exist. The leasing catalog replaced them: `catalog.yaml`
declares `models`, `endpoints`, `runtime_hosts` and `bundles`, and
`infer-stack acquire <endpoint>` leases an endpoint and brings up its engine
behind the LiteLLM gateway and Open WebUI. Gateway and UI are settings
(`infer-stack config set litellm|ui|reverse_proxy …`), not graph nodes.

Start with the README's "Primary leasing workflow" and "Catalog model"
sections, then [the user manual](source/manual/index.md) and
[litellm-gateway-routing.md](litellm-gateway-routing.md).

"Profile" survives in one place: the ledger's **recovery snapshot**, the
frozen copy of every non-ledger render input (settings, image pins, the
published catalog union) that recovery re-renders from. `acquire` advances it
automatically; `infer-stack config publish` pre-seeds or previews it
explicitly. See [ADR 0001](adr/0001-user-config-is-authoritative.md) and the
docstring of `infer_stack/leasing/profile.py`.
