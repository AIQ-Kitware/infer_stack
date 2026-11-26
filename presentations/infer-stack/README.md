# Infer-stack presentation suite

Five independent decks for different audiences, sharing a pinned PptxGenJS
build and the anonymized historical benchmark snapshot. The audience decks
are deliberately short and tell different stories; use the technical overview
when a deeper architecture explanation is needed.

| Deck | Audience | Slides | Suggested delivery | Build |
|---|---|---:|---|---|
| Why infer-stack? | Prospective adopters, research leads | 7 | 5–8 min | `npm run build:why` |
| Using infer-stack | Developers and researchers | 9 | 15–20 min | `npm run build:how` |
| Measuring inference | ML engineers, evaluation researchers | 8 | 10–15 min | `npm run build:evidence` |
| Infer-stack live | Technical demo audience | 8 | 15–25 min, select demos | `npm run build:demo` |
| Technical overview | Contributors, architecture discussions | 18 | 20–30 min | `npm run build:technical` |

The timings are guidance for the presenter, not promises about live startup or
benchmark duration. All new decks have their own opening and closing, and
speaker notes with source references, caveats and demonstration guidance.
The original technical overview is preserved unchanged.

## Build

Prerequisites: Node.js 20+ and npm. LibreOffice Impress is optional for PDFs.
From the repository root:

```sh
cd presentations/infer-stack
npm ci
npm run build
```

`npm run build` now generates all five PowerPoints in `build/`. To generate
only the four new audience decks, run `node build-suite.cjs`; to generate one
or several, pass the corresponding names (`why`, `how`, `evidence`, `demo`).
The original `node build.cjs` still builds the technical overview.

```sh
npm run pdf
# Or export just one already-built deck:
libreoffice --headless --convert-to pdf --outdir build build/infer-stack-why.pptx
```

| Output stem under `build/` | Contents |
|---|---|
| `infer-stack-why` | Adoption brief |
| `infer-stack-how` | Hands-on tutorial |
| `infer-stack-evidence` | Performance case study |
| `infer-stack-demo` | Live demonstrations |
| `infer-stack-technical-overview` | Existing detailed deck |

Generating slides does not run infer-stack, start containers, require a GPU,
or contact model servers. The first `npm ci` may download packages; the build
itself is offline. Font availability (Aptos and Liberation Mono) and renderer
versions can affect presentation appearance. All elements are editable native
PowerPoint shapes and text, including the plotted curves. No embedded raster
assets are required. The QA helper path is optional for development only:
`INFER_STACK_SLIDE_QA_HELPERS=/path/to/pptxgenjs_helpers node build-suite.cjs`.

## Deliver a talk

Use [DEMO.md](DEMO.md) for setup, exact commands, expected observations,
cleanup and fallbacks. `demo_client.py` is a standard-library-only chat client
which reads the connection information supplied by `acquire` or `run`.
It does not print the API key. Run it from this folder, or use its full path.
The build and demo execution are separate: never start GPU work merely to
regenerate slides.

- Adoption talk: explain the problem and the lease contract, then show the
  measured tradeoff and invite the audience to try the simulator.
- Workshop: walk through the how deck, then select one or two demo modules.
- Performance talk: use the evidence deck, open a saved dashboard and discuss
  what further experiments would discriminate the serving configurations.
- Architecture review: use the unchanged technical overview, then select the
  evidence slides if useful.

## Maintain the source

- `build.cjs`: original 18-slide technical overview.
- `build-suite.cjs`: sequential audience-deck builder.
- `decks/{why,how,evidence,demo}.cjs`: slide content and notes, one block per slide.
- `layout.cjs`: common design and native drawing helpers.
- `metrics.cjs`: shared lookup into the dated numerical snapshot; the chart
  values and selected numerical headlines in audience decks come from it.
- `data/benchmark-2026-11-25.json`: sanitized benchmark export, with 96
  successful requests and four context-headroom skips. It retains endpoint
  names, target, phase, success, API-reported token counts, TTFT, estimated
  streaming and end-to-end rates, and reported cached tokens. No hostnames,
  GPU UUIDs, credentials, or raw logs. All tests ran on the same RTX 3090
  (observed PCIe Gen3 x2), one repeat, 512-token generation budget.
- `package.json` and `package-lock.json`: pinned presentation-only dependencies.

The benchmark is a fixed historical snapshot. When adding new results, keep
historical datasets and update any associated chart bounds, narrative,
hardware claims, assumptions, and source notes. Do not present a simulator
result as measured model performance or serving throughput as model quality.
Review commands against the current README and CLI when the project evolves.

Only text sources go into Git. Generated slides, PDFs, `node_modules/`, and
`build/` are ignored. No dependencies are added to the Python package.
