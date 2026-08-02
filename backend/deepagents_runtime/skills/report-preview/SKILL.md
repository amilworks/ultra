---
name: report-preview
description: Verify an HTML report renders correctly BEFORE finishing the run — headless render of outputs/report.html to a screenshot plus a console/broken-image log, inspected with inspect_images. Use after writing or revising any .html deliverable; not for markdown reports (they render natively) and not a substitute for reading your own numbers.
---

# Report preview (HTML self-check)

## When to use
After writing or revising an HTML deliverable (`outputs/report.html` or any
`.html` artifact), before declaring the task done. A report that renders
broken for the reader — a figure path that never resolved, a script error, a
layout that collapsed — is a defect you can catch yourself in one render.
Markdown reports do not need this; the platform renders them natively.

## How
1. Render:
   ```bash
   python3 /skills/report-preview/scripts/render_report.py outputs/report.html
   ```
2. Read the exit code and stdout. Exit 0 = clean. Exit 2 = findings — open
   `/workspace/diagnostics/report_preview/report.console.json`, fix the
   report, re-render. Exit 3 = rendering is unavailable in this environment:
   continue without it — do NOT retry, install anything, treat it as a
   report defect, or hand-probe for browser binaries first (the renderer's
   exit code IS the availability check; skip the `which chromium` /
   `import playwright` reconnaissance).
   In the final answer, this is worth at most ONE neutral clause, without
   naming internal tools: "Visual rendering was not verified in this
   environment; structural checks passed." Never enumerate missing
   packages/binaries, never repeat the limitation in multiple sections, and
   never frame it as a complaint — the reader is a scientist, not the
   platform operator. Capability details belong in run diagnostics, not in
   deliverables.
3. Inspect the screenshot with `inspect_images` on
   `/workspace/diagnostics/report_preview/report.png` and look at it the way
   a reader would: figures present, numbers legible, sections in order,
   nothing overlapping or truncated.
4. Fix and re-render until clean, then finish. Previews live under
   `/workspace/diagnostics/` on purpose — they are scratch, never durable
   outputs, and must not be referenced from the report.

## What the render checks (and why network is off)
The page renders from `file://` with all external requests blocked — the same
deal the reader gets: the platform renders HTML reports in a sandbox whose
CSP blocks outbound requests. If your report "needs" a CDN font, script, or
remote image, it is already broken for the reader; inline the asset or drop
it. Figures referenced by `outputs/`-relative paths are embedded by the
platform at read time, and render here when the file exists next to the
report — a broken image in the preview means the path is wrong or the file
was never written.

The console log records: console errors, page (script) errors, broken
images, blocked external requests, and failed requests. Console errors from
your own inline scripts are report defects; fix them.

## Boundaries
- Verifies RENDERING, not science: numbers, uncertainties, and citations are
  governed by the scientific-reporting skill and are your job to check.
- One render after each revision is enough; do not loop screenshots hunting
  pixel perfection.
- If chromium is unavailable (exit 3), the report still ships — visual
  self-check is an additional safeguard, not a gate.
