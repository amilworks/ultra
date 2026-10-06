---
name: bisque-module-advisor
description: Pick and run the right BisQue analysis module for a task — list the modules registered on the linked BisQue server, match them to the user's goal and data, explain which inputs a run needs (with real example values), and run the module on BisQue so the data is analysed in place instead of copied into Ultra. Use for "which BisQue module should I use for X", "what can BisQue do with my images", "run <module> on this image/dataset", or any BisQue analysis request.
---

# BisQue module advisor

## When to use
Use this whenever the user asks what BisQue can do, which BisQue module or analysis fits a
task, or asks to run a BisQue module. It requires a linked BisQue account (the bisque_* tools
are only registered when one is linked).

## Recommend a module
1. **Find the user's data first** when they mention it: `bisque_search_resources` (scope="owner")
   or `bisque_dataset_members`, so you know its type (image / dataset / file / video) and format.
2. **List the registered modules**: `bisque_list_modules(task="<the user's goal in plain words>")`.
   Only these modules can run; never recommend a module that is not in this list.
3. **Describe the top candidates**: `bisque_describe_module(name=...)` for the 1–3 best matches.
   Read the help text and `example_run` — the inputs of the most recent finished run show the
   concrete values a run needs (for example a model `weights` file already stored on BisQue).
4. **Check compatibility** against the user's data: input types and `template.accepted_type`
   (image vs dataset vs video), and required inputs you cannot fill.
5. **Answer with**: the recommended module(s) and why, the inputs a run needs (and which ones you
   will take from `example_run`), what it produces, and caveats — module descriptions are often a
   single line, so say so when the evidence is thin rather than overstating fit.

## Run a module (only when the user explicitly asks to run it)
- Use `bisque_run_module(module_name=..., inputs={...})`. It is only available when the run is
  authorized for `bisque.run_module`; if it is missing, tell the user to ask explicitly
  (e.g. "run PrairieDogDetection on BisQue on <image>").
- Pass every resource input as a **BisQue resource URI or resource_uniq** — never download the
  data into Ultra to run a BisQue module. Use the weights / model files from `example_run`.
- For annotation outputs, give a distinctive `Annotation Name` (e.g. `ultra_<task>_<date>`) so the
  results are easy to find and remove.
- Modules run on BisQue's engine and often take tens of minutes. Report the run's
  `client_view_url` and status; check later with `bisque_module_runs(mex_uri=...)`.
- When it finishes, report outputs via their `client_view_url`. Annotation results land on the
  input image — count them with `bisque_image_annotations`.

## Do not
- Do not download BisQue data into Ultra when a registered module already covers the task.
- Do not invent module names, inputs, or parameter values.
- Do not run a module on another user's image with an option that writes annotations unless the
  user owns or explicitly chose that image.
