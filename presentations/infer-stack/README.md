# Infer-stack technical overview

The source of the 18-slide deck created on November 25, 2026. The generator
uses PptxGenJS and constructs all diagrams, charts, and text as editable
PowerPoint elements. Speaker notes include the source references and caveats.

## Build

Prerequisites: Node.js 20 or later and npm. For PDF export, install LibreOffice
Impress with its `libreoffice` command on `PATH`.

From the repository root:

```sh
cd presentations/infer-stack
npm ci
npm run build
```

The generated deck is `build/infer-stack-technical-overview.pptx`. To rebuild and
export the PDF as well:

```sh
npm run pdf
```

The PDF is `build/infer-stack-technical-overview.pdf`. Open the generated
PowerPoint or PDF with your preferred viewer. Both commands operate in this
folder, without running infer-stack, starting Docker, or accessing a GPU.
The normal build needs no additional assets or external slide helpers. The
lockfile pins the JavaScript dependencies; the initial `npm ci` may download
packages. The generator requests Aptos and Liberation Mono fonts. Rendering
can vary if these fonts are unavailable or different versions of PowerPoint
or LibreOffice are used.

## Edit and maintain

- `build.cjs`: layout helpers followed by one numbered block per slide, including
  speaker notes and references. The final slides contain the conclusions and
  interpretation of the fixed benchmark snapshot.
- `data/benchmark-2026-11-25.json`: sanitized numerical data for the historical
  case study, from the supplied `summary.csv`. It retains endpoint names,
  target context, phase, success, API token counts, TTFT, estimated streaming
  throughput, end-to-end throughput, and reported cache tokens. It contains
  96 successful measurements and four context-headroom skips. Hostnames,
  physical GPU identifiers, configuration hashes, raw logs, and credentials
  are not included. All measurements came from one RTX 3090 (observed PCIe
  Gen3 x2), with one repeat and a 512-token generation budget.
- `package.json`, `package-lock.json`: presentation-local build dependencies.

This dataset is a dated snapshot, not a live benchmark feed. If replacing it,
update the accompanying narrative, axis limits, highlighted numerical callouts,
notes, and citations together. A new snapshot may deserve its own deck or data
file rather than replacing the historical result. As infer-stack evolves,
review the CLI examples and architecture slides against the current docs.

All source files are text. Generated PowerPoints, PDFs, and `node_modules/`
are ignored; do not commit them. No extra dependency is added to the Python
package. The normal build does not rely on a machine-specific path.
