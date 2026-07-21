## 2026-07-18 20:05:24 -0400

Model: GPT-5.6 Thinking.

User intent: address the major review findings that are actionable now without
changing lease lifetime behavior. Record the shell-environment and generated
credential-file risks as accepted planning-stage limitations, define the
single-owner meaning of configurable data roots, stop advertising unsupported
Windows execution, and repair the declared test environment and lock artifacts.

The central design decision was not to add a partial instance namespace. The
current scheduler treats a host or backend namespace as one allocation domain;
adding only resource-name isolation would prevent some cleanup collisions while
leaving independent controllers free to oversubscribe the same GPUs. The docs
therefore define --data-dir as relocation of one control plane and explicitly
forbid concurrent roots against the same Docker host or Kubernetes namespace.
True multi-instance support remains a larger design requiring both resource
identity and cooperative hardware allocation.

Platform support is now explicit: Linux only. The checked-in xcookie config and
generated test matrix both use Linux, so regenerating CI should not silently
restore Windows or macOS jobs. This matches the implementation's POSIX locking
and Linux container/GPU assumptions rather than pretending import portability is
runtime support.

The packaging fix includes every pytest plugin required by the configured test
commands. xdoctest is required by pyproject addopts, and pytest-cov is required
by the sdist/wheel CI commands. Both were added to the tests extra; uv.lock and
the exported tests constraint file were regenerated with the existing
exclude-newer instant preserved exactly.

Risks and tradeoffs: the documented shell quoting and file-mode problems remain
real security limitations by explicit user choice. Linux-only CI intentionally
reduces portability coverage. The leasing run TTL behavior is deliberately
untouched and should be discussed separately before implementation.
