# Benchmark comparison dashboard

Replaced the flat HTML view with cohort-scoped comparisons by hardware identity and workload settings. Preserve the existing CSV, Markdown, ledger schema, configuration identities, and atomic live-report writes. Added coverage/ready time, long-context scorecards, metric-specific observed leaders, six inline SVG charts, and collapsible detailed rows. Failed/absent observations are not plotted as zero; reported cached-token zero is not treated as proof of an engine cache miss.

The HTML uses no JavaScript, CDN, or plotting dependency. Tested the report rendering and grouping against synthetic records and generated a preview from the supplied completed six-endpoint sweep. No new GPU benchmarks were run.
