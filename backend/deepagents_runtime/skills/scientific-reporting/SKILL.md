---
name: scientific-reporting
description: Report contract and artifact hygiene for the write-up — required sections (methods, uncertainties, verification, limitations, references), figure standards, subagent attribution, and durable-output discipline. Use when writing a scientific report, a results summary for a researcher, or producing durable deliverables (figures, CSVs, code, markdown) — i.e. while communicating results, not while computing them.
---

# Scientific Reporting

## When to use
Read this skill whenever the deliverable includes a written report, a results
summary for a scientist, or durable artifacts (figures, CSVs, code) that a
researcher will reuse.

## Report contract
A frontier-quality report has these sections; omit one only when it is
genuinely empty, and then say so explicitly:

1. **System / Problem** — equations, parameters, and conventions, with units.
2. **Methods** — algorithms by name, step sizes, transients, durations,
   seeds, initial conditions. Enough to reproduce without reading the code.
3. **Results with uncertainties** — every decision-relevant number carries
   ± spread or a stated resolution limit. Tables and CSVs include the
   uncertainty column, not bare point estimates.
4. **Verification** — what was independently checked, by whom: attribute
   delegated checks to the subagent by name and quote its key numbers next
   to yours. "Consistent" without numbers is not verification.
5. **Limitations** — finite observation time, integrator artifacts,
   multistability / initial-condition dependence, parameter ranges not
   explored, and any unresolved discrepancies with known results.
6. **Known results / references** — when the system is canonical, a short
   comparison against commonly known behavior, named at the textbook/author
   level only when confident. Never invent precise citations; absence of a
   literature comparison must be stated, not silent.

## Figure standards
- Every figure: axis labels with units, title or caption stating parameters,
  legend when more than one series, 300 DPI for static exports.
- Reference each figure from the report text near the claim it supports —
  no orphan figures, no figures dumped at the end.

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
- Report wall-clock session time and inner compute time as separate labeled
  numbers when runtime is mentioned at all.
- State explicitly which subtask was delegated, to which subagent, and what
  came back — the reader should be able to audit the division of labor.
