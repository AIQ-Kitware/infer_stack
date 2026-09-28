# Cluster setup for infer-stack

`infer-stack` targets Kubernetes capabilities, not a particular Kubernetes
distribution. K3s is the first provisioning integration because it gives a
small workstation cluster a short path to a conformant Kubernetes API without
making cluster administration the main project.

The architecture boundary is:

```text
any Kubernetes distribution
        |
        | kubectl / Kubernetes API
        v
infer-stack kube nodes / setup
        |
        v
GPU scheduling + discovery, KubeAI, resource profiles
```

Distribution-specific lifecycle commands live below that boundary:

```text
infer-stack kube k3s bootstrap
infer-stack kube k3s join

# A future integration could add, for example:
# infer-stack kube k0s ...
```

Nothing in `infer-stack kube setup`, model placement, the lease controller, or
the KubeAI backend requires K3s. An existing kubeadm, RKE2, k0s, EKS, or other
Kubernetes cluster should start at [Use an existing cluster](#use-an-existing-cluster).

## Mental model

A small cluster has one or more Kubernetes **control-plane** nodes and one or
more **worker** nodes. The control plane owns Kubernetes API/state/scheduling;
workers run pods.

For the first K3s topology, use one workstation as a K3s **server** and join
other workstations as K3s **agents**:

```text
workstation A
  K3s server
  Kubernetes control plane
  worker too (default K3s behavior)
  may run GPU model pods

       | Kubernetes cluster
       +---------------- workstation B
       |                   K3s agent / worker / GPU
       |
       +---------------- workstation C
                           K3s agent / worker / GPU
```

The server is not required to be a dedicated appliance. In the initial
workstation cluster it can keep contributing its GPU.

This is separate from infer-stack's own authority. Use one infer-stack control
plane for a Kubernetes namespace; do not have independent infer-stack state
directories concurrently acquire/release against the same namespace. The
machine running that authority can initially be the K3s server for simplicity,
but Kubernetes control-plane ownership and infer-stack lease/config ownership
are different concepts.

## What generic setup expects

Before `infer-stack kube setup --apply`, the generic integration cares about
observable capabilities rather than how the cluster was installed:

- `kubectl` selects the intended cluster context and can reach its API.
- Nodes are Ready.
- GPU nodes expose allocatable `nvidia.com/gpu` when NVIDIA scheduling is
  desired.
- GPU product/memory discovery labels are available for profile generation.
- The node/container runtime can actually launch NVIDIA workloads. A working
  externally managed GPU integration is accepted; infer-stack does not require
  a particular installer.
- Helm is available when infer-stack needs to reconcile a Helm-managed
  component.
- KubeAI's CRD/service are present, or infer-stack can install/reconcile them.

The read-only check is always the first command:

```bash
infer-stack kube nodes
infer-stack kube setup
```

Only after reviewing the target context and plan:

```bash
infer-stack kube setup --apply
```

Host NVIDIA driver/container-runtime installation stays outside infer-stack.
The exact host procedure depends on the distribution and operating system.

## First provisioning target: K3s

### 1. Prepare the first workstation

Choose the first workstation's stable LAN/VPN address. Install and verify its
NVIDIA driver/container runtime before starting K3s when this node will serve
GPU workloads.

Record an exact K3s version for a real cluster so every node joins with the
same version.

### 2. Create the K3s server

On workstation A:

```bash
infer-stack kube k3s bootstrap --version=<exact-k3s-version>
```

This command is intentionally scoped under `kube k3s`: it installs/starts a
K3s server, waits for its node to become Ready, and makes the K3s kubeconfig
usable without overwriting an existing `~/.kube/config`.

Then inspect the generic Kubernetes integration:

```bash
infer-stack kube nodes
infer-stack kube setup
infer-stack kube setup --apply
```

The setup commands above are not K3s-specific.

### 3. Obtain the join information

On the K3s server, the agent token is stored at:

```text
/var/lib/rancher/k3s/server/node-token
```

Copy that token to a mode-0600 file on the workstation being joined using your
normal secure transport. Also record workstation A's reachable server URL:

```text
https://<workstation-A-address>:6443
```

Do not put the token directly on a command line or in shell history.

### 4. Join another workstation

After preparing the NVIDIA host runtime on workstation B when applicable:

```bash
infer-stack kube k3s join \
    --server=https://<workstation-A-address>:6443 \
    --token-file=~/.private/k3s-token \
    --node-name=<workstation-B-name> \
    --version=<same-k3s-version>
```

Repeat that command on each additional workstation with its own node name.

K3s networking/firewall requirements are distribution-specific operational
requirements, not infer-stack requirements. In particular, agents must be able
to reach the K3s API on the server, and inter-node traffic required by the
configured K3s networking backend must be allowed on the trusted cluster
network.

### 5. Reconcile newly visible hardware

Back on the machine operating infer-stack:

```bash
infer-stack kube nodes
infer-stack kube setup
infer-stack kube setup --apply
```

A newly joined GPU product may produce a new KubeAI resource profile. Existing
operator-owned profiles with the same names remain authoritative.

### 6. Verify the serving path

After the cluster inventory is correct, a fresh Kubernetes-only control host
can make KubeAI its durable default:

```bash
infer-stack config set backend kubeai
infer-stack config set kubeai_gateway cluster
infer-stack doctor
```

If this machine already has a Compose deployment history that you want to keep
and return to later, do **not** reuse that Compose ledger as the KubeAI ledger.
Use the temporary-testing workflow below instead.

Then exercise a real acquire/generation/release and the multi-host handoff
checks described in [the KubeAI backend guide](kubeai-backend.md).

## Temporarily swap Compose-configured workstations into a cluster

It is common to start with workstations that already use infer-stack's Compose
backend and temporarily pool them for KubeAI testing. There are two independent
pieces of state to preserve:

1. the workstation's existing Compose configuration/ledger; and
2. the Kubernetes node's cluster identity.

Neither needs to be destroyed. Use a separate KubeAI data root for the cluster
authority, and temporarily **cordon + drain** a node when its physical GPU is
being handed back to local Compose. The Kubernetes distribution agent keeps
running throughout; `detach` does not uninstall, stop, or remove the node.

### Keep the existing Compose authority intact

Before using a Compose-configured machine as a Kubernetes GPU worker, quiesce
its local Compose backend:

```bash
infer-stack release --all --evict
infer-stack ps
```

On the one machine that will operate the KubeAI infer-stack authority, use an
independent data root instead of changing the existing Compose default:

```bash
export INFER_STACK_BACKEND=kubeai
export INFER_STACK_DATA_DIR="$HOME/.local/share/infer_stack-kubeai"

# KubeAI-only settings can coexist with a persisted Compose default.
infer-stack config set kubeai_gateway cluster
infer-stack doctor
```

`INFER_STACK_BACKEND` overrides the persisted `backend: compose` only for the
current shell. `INFER_STACK_DATA_DIR` gives KubeAI its own ledger and rendered
state, which is important because recovery snapshots intentionally do not
change backend kind inside one ledger. The normal catalog/config root remains
shared, so the same endpoint definitions can be exercised on both backends.

Workstation B does not need a local KubeAI infer-stack authority merely because
it is a Kubernetes worker. Its existing Compose configuration can remain
untouched while the authority on workstation A schedules KubeAI pods onto it.

### Hand one node back to Compose

If you are ending the KubeAI test entirely rather than removing only one
worker, first quiesce the KubeAI authority while its temporary environment is
still active:

```bash
infer-stack release --all --evict
infer-stack ps
```

For a single-node handoff, detach can instead evict that node's workload pods;
their controllers may reschedule them onto other attached workers.

Run the node lifecycle command from any shell whose `kubectl` context has
permission to drain the target node. The first invocation is a read-only
preview:

```bash
infer-stack kube node detach <node-name>
infer-stack kube node detach <node-name> --yes
```

Detach:

- records that infer-stack owns this temporary cordon;
- cordons the node so Kubernetes cannot schedule new ordinary workloads there;
- drains controller-managed workload pods;
- keeps DaemonSet/static pods and the Kubernetes agent/control plane running;
- refuses unmanaged/bare pods instead of using `kubectl drain --force`.

`infer-stack kube nodes` reports such a node with `SCHED=compose`. A K3s server
that is also the sole control-plane node may be detached this way: the control
plane remains running because detach does not stop the K3s service. Cluster
workloads can of course become Pending if every worker is cordoned.

After detach succeeds, use the target workstation's original Compose setup.
If the current shell was the KubeAI authority shell, return to the persisted
Compose default by dropping the temporary overrides:

```bash
unset INFER_STACK_BACKEND INFER_STACK_DATA_DIR
infer-stack doctor
```

### Hand the node back to Kubernetes

First stop/release every local Compose GPU workload on the target workstation:

```bash
infer-stack release --all --evict
infer-stack ps
```

Then, from a cluster-admin shell, preview and explicitly re-enable scheduling:

```bash
infer-stack kube node attach <node-name>
infer-stack kube node attach <node-name> --yes
```

`attach --yes` is the operator's assertion that local Compose no longer owns
the GPU. It waits for the existing node to be Ready, uncordons it, and removes
the infer-stack detach marker. It refuses to uncordon a node that was cordoned
by someone else, so temporary backend switching cannot silently undo unrelated
cluster maintenance.

To operate KubeAI again on the authority machine:

```bash
export INFER_STACK_BACKEND=kubeai
export INFER_STACK_DATA_DIR="$HOME/.local/share/infer_stack-kubeai"
infer-stack doctor
```

There is no second `kube k3s join`: attach reuses the node identity and agent
configuration that were preserved during detach.

For two Compose-configured machines A and B, a complete test cycle is therefore:

```text
Compose A + Compose B
  -> quiesce both Compose stacks
  -> create/join Kubernetes once
  -> use a separate KubeAI data root on A
  -> test KubeAI on A+B
  -> release KubeAI workloads
  -> kube node detach A/B
  -> use Compose A/B again
  -> quiesce Compose A/B
  -> kube node attach A/B
  -> resume the same KubeAI authority/data root
```

## Use an existing cluster

If Kubernetes already exists, do not run a provisioning command. Select its
kubeconfig/context and begin at the generic layer:

```bash
kubectl config current-context
infer-stack kube nodes
infer-stack kube setup
infer-stack kube setup --apply
infer-stack doctor
```

The generic setup path must not depend on K3s files, services, tokens, Flannel,
or installer behavior. If a future Kubernetes distribution needs first-class
provisioning convenience, add it under its own `infer-stack kube <distro>`
namespace while preserving this generic contract.

## Ownership boundary

`infer-stack kube` deliberately stops short of being a general Kubernetes
administrator:

- `kube setup` owns only the integration components/capabilities infer-stack
  needs.
- `kube node detach/attach` owns only a temporary scheduling cordon used to
  hand a host's GPU between Kubernetes and local Compose; it does not remove
  cluster membership.
- Distribution-specific provisioning stays in scoped subcommands such as
  `kube k3s`.
- Arbitrary cluster administration remains `kubectl`, Helm, and the chosen
  distribution's tooling.
- Existing externally managed GPU/KubeAI capabilities are accepted instead of
  replaced merely because they were installed differently.

This keeps K3s easy to use today without making it part of infer-stack's
backend or scheduling architecture.
