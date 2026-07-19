# Known limitations

`infer-stack` is still at planning-stage maturity. The limitations below are
accepted for now and define the environment in which the current implementation
is intended to be operated.

## Generated shell environment files require trusted inputs

Lease helper commands can emit shell files containing `export NAME=value`
assignments. Values are not currently shell-escaped. Sourcing one of these files
can therefore execute shell syntax embedded in a configured value.

Treat the catalog, config, model names, endpoint URLs, API keys, and generated
shell files as fully trusted local input. Do not source generated environment
files derived from untrusted or multi-tenant configuration. This is a known
security limitation, not a supported sanitization boundary.

A future hardening pass should quote values for the target shell, validate
variable names, and reject values containing unsupported control characters.

## Generated Compose environment files may be readable by other local users

Generated Compose `.env` files can contain deployment credentials such as the
LiteLLM master key or database password. They are currently created using the
process umask rather than an enforced private file mode. On a multi-user host,
a permissive umask can make those files readable by other local accounts.

Operate infer-stack under a dedicated trusted account and keep its config/data
roots private. Until file modes are enforced by the application, use a
restrictive umask such as `umask 077` before setup, render, and controller
operations. Do not treat the generated directory as safe for mutually
untrusted local users.

## One control plane per host or backend namespace

`--data-dir` and `INFER_STACK_DATA_DIR` relocate infer-stack state; they do not
namespace independent controllers. Compose resources currently share a fixed
project identity, and KubeAI reconciliation identifies resources with a shared
managed label. Two controllers using different roots against the same Docker
host or Kubernetes namespace can disagree about desired state and remove or
replace one another's resources.

The supported operating model is one infer-stack control plane per Docker host,
or one per Kubernetes namespace:

1. Pick one config root and one data root.
2. Ensure every controller and administrative command uses those same roots.
3. Stop the existing controller before relocating either root.
4. Move or re-create the state, update the configured paths, and then start the
   single controller again.

This limitation is deliberate for now. Adding an instance identifier to every
Compose and Kubernetes resource could isolate cleanup operations, but it would
not make two independent GPU schedulers safe or cooperative on one machine.
True multi-instance support therefore needs both resource namespacing and a
shared hardware-allocation model; partial namespacing would give a misleading
sense of safety.

## Linux-only execution

Linux is the only supported execution platform. The implementation relies on
POSIX process locking and Linux-oriented container, GPU, and service-management
workflows. Windows is not supported or tested. Other POSIX systems are not a
supported deployment target even when individual pure-Python modules happen to
work there.
