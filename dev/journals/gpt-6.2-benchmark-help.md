## 2026-11-25 16:12:35 -0500

Model: ChatGPT (GPT-6.2); finer configuration not exposed.

The user wants `infer-stack benchmark run --help` to direct users to `infer-stack catalog endpoint list` so they can discover names before running a sweep. I updated the command description, positional and named selector help, and missing-selector/unmatched-glob errors. The latest live-benchmark CLI is the base, preserving its reporting and cancellation behavior. This deliberately avoids duplicating catalog discovery or changing benchmark defaults. The catalog command registration was verified in the latest source snapshot; the modified module compiles. Actual kwconf help rendering was not tested because kwconf is not installed here.
