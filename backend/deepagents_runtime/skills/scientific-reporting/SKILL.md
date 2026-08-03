---
name: scientific-reporting
description: Domain-neutral report and artifact-hygiene guidance for scientific write-ups — task-appropriate methods, uncertainty, validation, limitations, references, figure standards, subagent attribution, and durable-output discipline. Use when communicating results or producing durable scientific deliverables, not while computing them.
---

# Scientific Reporting

## When to use
Read this skill whenever the deliverable includes a written report, a results
summary for a scientist, or durable artifacts (figures, CSVs, code) that a
researcher will reuse.

## Report contract
A strong scientific report has these sections; omit one only when it is
genuinely empty, and then say so explicitly:

1. **Question / Inputs** — the scientific question, source data or materials,
   relevant selection criteria, parameters, conventions, and units.
2. **Methods** — algorithms and procedures by name, task-appropriate sampling
   or repetitions, preprocessing, controls, and enough configuration to
   reproduce the result without reading the implementation.
3. **Results and uncertainty** — report each decision-relevant result with the
   uncertainty, variability, confidence interval, calibration caveat, or
   resolution limit appropriate to that task. Do not invent repeated trials or
   uncertainty estimates when the available evidence cannot support them.
4. **Validation** — state what was checked and how. Attribute delegated checks
   to the subagent by name and include the decisive measurements or observations
   rather than saying only that they were "consistent."
5. **Limitations** — identify data quality, sampling, measurement, model,
   validation, generalizability, and unexamined-scope limitations that actually
   apply to the task; do not import limitations from an unrelated domain.
6. **References / grounding** — cite external sources when a claim depends on
   them, at the level the available evidence supports. Never invent citations;
   distinguish measured results from externally grounded interpretation.

## Report format
- Default to markdown (`outputs/report.md`). Write HTML (`outputs/report.html`)
  when the deliverable genuinely earns it: interactive figures, a dashboard-like
  results page, or figure-dense layouts markdown cannot hold. Both formats get
  the same reading surface in the chat — format is a presentation choice, not a
  visibility one.
- An HTML report must be fully self-contained: inline all CSS and JS, no
  external network references (fonts, CDNs, trackers) — the reader renders it
  in a sandbox that blocks outbound requests, so anything external is simply
  missing. Reference run figures by their `outputs/`-relative path
  (`<img src="outputs/fig1.png">`); the platform resolves and embeds them.
- **Start from the house template**:
  `/skills/scientific-reporting/assets/report_template.html`. Copy it, keep its
  `<style>` block (tokens, type scale, KPI tiles, table and figure rules — the
  product's calm voice), replace the sample content. Do not re-derive a design
  from scratch; do not add color to chrome — the `--data-*` hues are for data.
- **Math**: emit native MathML — `latex2mathml` is installed
  (`python3 -c "import latex2mathml.converter as c; print(c.convert(r'\frac{2PR}{P+R}'))"`)
  and the reader's browser renders `<math>` with zero JS or fonts. Never
  hand-roll equation layout with spans, and never link MathJax/KaTeX CDNs.
- **Interactive charts**: `plotly` is installed; embed with
  `fig.to_html(full_html=False, include_plotlyjs="inline")` (inline JS is
  allowed; the network is not). Use it only when interaction earns its ~3.5MB;
  static figures stay matplotlib PNGs at 300 dpi.
- **Table of contents**: plain fragment links (`<a href="#results">`) with
  matching section `id`s are fully supported — the reading canvas turns them
  into in-document scrolling. Give sections `scroll-margin-top` (the template
  does). Never use `target="_top"` or `target="_blank"` links as navigation;
  the reader's sandbox blocks them.
- To revise a report in a follow-up run, write the SAME filename again
  (`outputs/report.html`). The platform chains registrations of one path into
  versions behind a single document; a new filename mints a new document, so
  rename only when the deliverable is genuinely a different report.

## Figure standards
- Every figure: axis labels with units, title or caption stating parameters,
  legend when more than one series, 300 DPI for static exports.
- Reference each figure from the report text near the claim it supports —
  no orphan figures, no figures dumped at the end.

## Interactive HTML pages — render proof, offline assets, bounded verifiers
- **Render proof is required, not optional.** The completion guard withholds
  completion while an `.html` deliverable lacks passing headless-render
  evidence. Verify the FINAL built file (not a draft): load it in headless
  Chromium (`PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright`) with
  network disabled, require zero console errors and zero page errors, and
  exercise at least one interaction, asserting its effect (a readout changes,
  canvas pixels change). Only on a pass, write the evidence JSON — at minimum
  `{"console_errors": [], "page_errors": []}` — to
  `/workspace/diagnostics/report_preview/<page>.console.json`. Never write
  the evidence file for a failing page; fix the page instead.
- **Pages must work offline.** No CDN or external URLs of any kind — the
  sandbox has no network and neither may the deliverable. For 3D, inline the
  vendored build at `/opt/report-assets/three.iife.min.js`. Static + numeric
  checks cannot see load-time failures; only the render proves the page.
- **Verifiers must be bounded.** Pass the execute tool's `timeout` parameter
  on every verification or subprocess call. Do not hand-roll character-scanning
  parsers or state machines (a non-advancing state hangs the run for hours);
  use `html.parser`, `json`, `ast`, or the preinstalled libraries.

## Artifact hygiene
- `/outputs/` (durable) holds only artifacts a researcher should keep: final
  code, final figures, data tables, the report. Each durable artifact must be
  referenced from the report with one line on what it contains.
- Debug, diagnose, and scratch scripts go under `/workspace/diagnostics/`
  (never `/outputs/`); the backend skips that directory when collecting
  durable artifacts, and unreferenced top-level workspace scripts are not
  collected either — a top-level script becomes durable only when the report
  or final answer names it.
- Name artifacts descriptively (`bifurcation_diagram.png`, not `plot2.png`),
  and keep names stable across follow-up runs that revise them.

## Communication
- Lead the final answer with the conclusion and its confidence, then the
  evidence; keep the full detail in the report artifact.
- **Platform internals stay out of deliverables.** The reader is a scientist,
  not the platform operator. An environment limitation (no visual preview, a
  capability unavailable) gets at most ONE neutral clause in the answer —
  "visual rendering was not verified in this environment" — with no internal
  tool, package, or infrastructure names (browser engines, sandbox images,
  execution machinery), no repetition across sections, and no complaint
  tone. State what WAS verified rather than itemizing what could not be.
  Never put such notes inside the report artifact itself.
- Report wall-clock session time and inner compute time as separate labeled
  numbers when runtime is mentioned at all.
- State explicitly which subtask was delegated, to which subagent, and what
  came back — the reader should be able to audit the division of labor.
