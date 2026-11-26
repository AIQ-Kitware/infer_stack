## 2026-11-25 20:07:09 -0500

Model: ChatGPT (GPT-6.2; configuration not exposed).

The user wanted the existing slides to speak to distinct audiences rather
than merely enumerate features: why adopt, how to use, evidence of benefits,
and interesting demos. The previous source overlay provided a good baseline
and pinned dependencies. I chose four standalone audience decks (7/9/8/8
slides) with shared presentation-local layout and data helpers, leaving the
18-slide architecture overview intact. The original build command still
works through `npm run build:technical`; the default builds all five.

The benchmark is useful because the measured ranking depends on the metric
and request pattern. I preserved its dates, matched-card context, missing
points, one-repeat caveat, and distinction between extrapolation and
measurements. No new performance claims or hardware tests were invented.
The adoption brief uses an illustrative day of work rather than claiming
measured operator time savings. The demo deck is a presenter guide to actual
interactive steps, not fabricated screenshots. A small stdlib chat client
and runbook support the demos, including a no-GPU path and careful cleanup.
Real GPU startup time, live CLI behavior and exact TUI rendering could not
be verified here; presenters should rehearse against their own configuration.

The reusable lesson is to share layout and evidence, not force distinct
narratives to share slide sequences. Build and live execution must stay
separate so presentations can be generated without GPU resources or secrets.
The runbook recommends a private working directory and the ignore rules
protect ordinary demo env files.

Validation: all 50 slides built from the existing pinned dependencies;
original 18 slide and 18 note XML files were byte-identical; all slides
have source notes. All four new decks exported to PDF and rendered; layout
and overflow checks passed after moving a strip clear of a flow diagram.
Native chart lines/dots intentionally intersect. The stdlib client was tested
against a loopback mock for success, HTTP 429 (no secret disclosure), and
rejection of multi-model selection. No real Docker/GPU run was attempted.
