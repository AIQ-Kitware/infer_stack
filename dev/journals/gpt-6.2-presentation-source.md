## 2026-11-25 19:52:48 -0500

Model: ChatGPT (GPT-6.2; configuration not exposed).

The user wants the technical overview slides to live under `presentations/`
with only editable sources tracked. The original PptxGenJS generator and
numerical benchmark rows were still available, so I preserved the deck rather
than reverse-engineering it from the PowerPoint. I kept the original layout,
notes and dated benchmark interpretation intact; the build now uses paths
relative to the generator, a presentation-local npm lockfile, and an ignored
`build/` directory. PDF export is optional and uses LibreOffice. The existing
layout diagnostic helpers are opt-in through an environment variable; they
are not required or vendored. This avoids tying the Python library or a normal
presentation build to an agent-specific machine or tooling installation.

The benchmark JSON is a sanitized historical snapshot, not a live feed. Its
narrative callouts and chart scales must be revisited if the dataset changes.
Font availability and renderer versions may affect visual fidelity even
though the slide source is pinned. Future decks should use separate folders
and commit source and non-sensitive data, not rendered files or dependencies.

Validation: `npm ci` using the available offline npm cache; `npm run build`;
`npm run pdf` (18 pages); all 18 slide XML files and 18 notes XML files matched
the original delivered deck byte-for-byte; slide overflow check passed.
