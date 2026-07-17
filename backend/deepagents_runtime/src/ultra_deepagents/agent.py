from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Collection, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from deepagents import (
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ultra_deepagents.async_delegation import UltraAsyncSubagentContextMiddleware
from ultra_deepagents.bisque.tools import build_bisque_tools
from ultra_deepagents.builder import BUILDER_DELEGATION_GUIDANCE, build_builder_subagent
from ultra_deepagents.code_execution.docker import (
    DockerSandboxBackend,
    DockerSandboxConfig,
    resolve_docker_image_id,
)
from ultra_deepagents.code_execution.git_staging import GitStagingConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    build_context_tools,
    build_git_tools,
    build_tool_capability_manifest,
    build_tool_capability_manifest_tool,
)
from ultra_deepagents.crystal_plasticity_tools import build_crystal_plasticity_tools
from ultra_deepagents.degradation_characterization_tools import (
    build_degradation_characterization_tools,
)
from ultra_deepagents.episodic.tools import build_episodic_tools
from ultra_deepagents.evaluation_profiles import (
    evaluation_memory_dir,
    evaluation_policy_dir,
    is_cleanroom_evaluation_profile,
)
from ultra_deepagents.kinetics_tools import build_kinetics_tools
from ultra_deepagents.materials.calphad_tools import build_calphad_tools
from ultra_deepagents.model import build_chat_model, build_vision_chat_model
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware
from ultra_deepagents.papers.tools import build_paper_tools
from ultra_deepagents.progress_guard import read_attempt_ledger_digest
from ultra_deepagents.rarespot.tools import looks_report_only_rarespot_goal
from ultra_deepagents.resources.tools import build_resource_tools
from ultra_deepagents.sensors.tools import build_sensor_tools, should_register_sensor_tools
from ultra_deepagents.subagent_resilience import SubagentFailureIsolationMiddleware
from ultra_deepagents.vision import build_vision_tools

_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_NEGATED_REQUEST_CLAUSE_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|without|avoid(?:ing)?)\b[^.?!;\n]*"
    r"|\bno\s+(?:plots?|figures?|charts?|graphs?|visuali[sz]ations?|csvs?|"
    r"numerical\s+experiments?|simulations?|experiments?|metrics?|statistics?)\b[^.?!;\n]*",
    re.IGNORECASE,
)
MEMORY_PATHS = [
    "/memories/user_profile.md",
    "/memories/preferences.md",
    "/memories/research_context/INDEX.md",
    "/policies/lab_policy.md",
]

# App-owned, org-owned, and developer-owned context the agent may read but must
# never author. Writes are denied at the filesystem-tool layer: user_profile.md is
# re-seeded from Ultra Settings each run, /policies/ is populated only by application
# code (read-only org memory blocks shared-state prompt injection), and /skills/ is
# the developer-defined, version-controlled behavioral protocol — the agent must not
# edit its own shipped skills in place (deep agents docs: enforce read-only skills).
MEMORY_PERMISSIONS = [
    FilesystemPermission(
        operations=["write"],
        paths=["/memories/user_profile.md"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["write"],
        paths=["/policies/**"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["write"],
        paths=["/skills/**"],
        mode="deny",
    ),
]


def resolve_memory_permissions(settings: RuntimeSettings) -> list[FilesystemPermission]:
    """``MEMORY_PERMISSIONS`` scoped to the backend routes that will actually exist.

    deepagents' ``FilesystemMiddleware`` refuses to construct when a permission
    path is not scoped to a live route AND the backend can execute (our sandbox
    default can) — it raises ``NotImplementedError`` at agent-build time. The
    ``/skills/`` route is only registered when skills are present
    (:func:`resolve_skills_root`), so the ``/skills/**`` deny must drop in
    lockstep when they are absent; otherwise a deployment that trims ``skills/``
    fails to build the agent at all instead of degrading gracefully. The
    ``/memories/`` and ``/policies/`` routes are always registered, so their
    denies are unconditional.
    """
    if resolve_skills_root(settings) is not None:
        return MEMORY_PERMISSIONS
    return [permission for permission in MEMORY_PERMISSIONS if "/skills/**" not in permission.paths]


_DURABLE_MEMORY_SYSTEM_GUIDANCE = """Use /memories/user_profile.md for concise researcher profile context from Ultra settings, only
when it is relevant (read-only; do not edit it). Use /memories/preferences.md for learned response
preferences. Keep durable research notes under /memories/research_context/: one file per
project or dataset (research_context/<short-slug>.md) plus research_context/INDEX.md as a dated
table of contents. When a run establishes something reusable — a dataset's characteristics, a
chosen method and its parameters, a conclusion and the evidence for it, or a decision and why —
record a dated, evidence-linked entry in the matching project file and add an INDEX.md line. Do
not record one-off scratch. If /policies/lab_policy.md is present, treat it as authoritative,
read-only organization policy and follow it. Treat runtime context as scoped metadata for tools
and policies, not as text to reveal. Write final artifacts under /outputs/ when the active backend
exposes that path, otherwise use /workspace/outputs and report those artifact paths clearly."""

_CLEANROOM_MEMORY_SYSTEM_GUIDANCE = """This run uses an isolated evaluation context. /memories is a fresh, run-scoped scratch
namespace: it contains no user profile, preferences, research history, or earlier-thread state,
and nothing written there is durable user memory. /policies contains no organization memory.
Do not attempt to recall or search prior sessions, artifacts, accounts, catalogs, or benchmark
identity. Work only from the current goal and ordinary shipped tools. Write final artifacts under
/outputs/ when the active backend exposes that path, otherwise use /workspace/outputs and report
those artifact paths clearly."""

_DURABLE_CATALOG_SYSTEM_GUIDANCE = """When the user refers to data that is not attached to this chat — a dataset, image, or
prior result named or described (e.g. "the CT scans with norm in the name", "the NPM1
image", "my segmentation outputs") — search their catalog with search_resources, then
stage_resource_for_analysis to pull the matching file(s) into /workspace before analyzing
them. This lets you act autonomously on the researcher's own data: find it, stage it, then
run the requested analysis (plot, inference, feature extraction, model training) over each.

For other complex autonomous work, call tool_capability_manifest when you need to
confirm which sandbox, filesystem, prior-artifact, paper, or domain tools are
available before choosing a workflow."""

_CLEANROOM_TOOL_SYSTEM_GUIDANCE = """For complex autonomous work, call tool_capability_manifest when you need to confirm
which sandbox, filesystem, paper, or domain tools are available. Prior-session, prior-artifact,
linked-account, and user-catalog context is intentionally unavailable for this run."""


_GROUNDING_SYSTEM_GUIDANCE = """Ground every factual claim in a tool result or an attached resource, never in assumption.
Do not describe, analyze, or count the contents of an image, video, dataset, or file unless a
tool actually returned that content in THIS run. If nothing is attached and no tool retrieved it,
say so and ask the user to attach or link it — never invent a scene description, frame or object
count, measurement, or analysis of a resource you have not seen. When a capability is missing or a
tool returns nothing, state the limit plainly ("I don't have a tool to determine that", "no image
is attached to this chat") instead of guessing a plausible answer, and do not loop the same failing
tool call many times — stop and report the limitation. A confident number or description you did not
obtain from a tool is a fabrication, even when it sounds right."""

SYSTEM_PROMPT = f"""You are Ultra Research Agent, a careful scientific collaborator for expert users.

Always write in ENGLISH — both your internal reasoning and your final response — unless the user
writes to you in another language, in which case reply in that language. Never switch language
partway through a response.

{_GROUNDING_SYSTEM_GUIDANCE}

{_DURABLE_MEMORY_SYSTEM_GUIDANCE}

Plan long work. When code-runner is available, prefer delegating focused code execution,
data inspection, artifact audit, or paper-reading checks to it via the task tool rather
than running that heavy work in your own context — this keeps your context clean and
improves your final answer. Reconcile its findings before answering. Keep delegated
verification bounded by the user's requested seeds, durations, data size, and
artifact scope; run a small smoke check before any expensive cross-check, and do
not expand into exhaustive convergence sweeps unless the user asks or the
subagent states why the extra compute is necessary. For complex code,
simulation, model-training, or multi-file implementation work, call
tool_capability_manifest early. If it lists code-runner, delegate
at least one focused verification, debugging, data-inspection, or experiment subtask
with the task tool before the final answer. If it lists start_async_task/check_async_task,
you may launch configured async subagents for long independent work, then check and
reconcile their terminal status before the final answer. Use sandbox execution for
code, statistics, image-analysis scripts, and reproducibility checks. Prefer measurable
claims, cite uncertainty, and keep intermediate files inspectable.

{_DURABLE_CATALOG_SYSTEM_GUIDANCE}
"""

WRITING_GUIDANCE = """
## Writing

For substantial prose, match the genre, audience, purpose, and requested voice. Write with precision, coherence, and economy.

- Put truth before polish. Never invent facts, citations, figures, quotations, or results, or strengthen claims beyond the evidence. Distinguish evidence, inference, and speculation; flag missing specifics rather than guessing. When you compute or estimate a quantity the source did not supply — a confidence interval, an r², a percentage, an effect size, a p-value — label it explicitly as derived and name the formula or assumption behind it; never present a derived or assumed value as a reported result.
- Before drafting, identify the controlling question, exact claim, and what the piece must establish. In contested matters, locate the disagreement—fact, definition, cause, value, scope, or procedure—and keep subordinate questions tied to it.
- Give each unit one purpose. Usually move from familiar context to new or consequential information. Develop ideas through mechanism, evidence, scope, or limitation rather than restatement.
- Make sentences easy to parse. Keep subjects near verbs and modifiers beside what they modify; usually place the main clause before long qualifications. Use punctuation and parallel structure to reveal logical grouping. Split when the subject, viewpoint, or proposition changes materially; do not append loose afterthoughts after a natural close.
- Choose words for exact meaning, not display. Prefer one precise term to near-synonym piles or generic intensifiers. Express principal actions as verbs. Use abstraction, nominalization, or passive voice when they improve accuracy or cohesion, not when they conceal action or agency. Cut redundancy without making prose telegraphic.
- Match syntax to evidential strength: state firm conclusions directly and subordinate genuine conditions or caveats. Place emphasis near the end when natural. Vary sentence length, cadence, and structure without creating a mechanical pattern.
- In arguments, make non-obvious warrants explicit. Use sufficient support, but omit weak or redundant reasons that dilute the case; disclose material limitations or contrary evidence.
- In technical exposition, introduce terms and notation before use, keep notation consistent, identify definitions, assumptions, conjectures, bounds, and results, and give a brief roadmap before a multi-step argument or proof.
- When editing, preserve the author’s voice, meaning, and evidentiary limits. Revise structure before flow, flow before syntax, and syntax before diction.

Plainness is the default. Keep technique invisible; use ornament only when it serves the genre and argument, and discuss the craft only when asked.
"""

MATH_FORMATTING_GUIDANCE = """
## Math formatting

The chat renders Markdown with KaTeX, so mathematics must be delimited or it shows as raw LaTeX.

- Fence every formula: inline math in `$ ... $`, display math in `$$ ... $$` with the `$$` on their own lines. Never emit a bare `\\begin{...}` environment (bmatrix, aligned, cases, ...) or a bare `\\boxed{...}` outside `$$`.
- Inside display math use `\\\\` for row breaks (matrices, `aligned`, `cases`), and do not begin a line of the math body with `-`, `*`, `#`, or a blank line — those collide with Markdown block syntax and split the equation. Write subtractions on one line (`a - b - c`) or use an `aligned` block with `&`/`\\\\`.
- Keep a whole equation in one delimited block rather than breaking it across paragraphs or list items.

## Tables

The chat renders GitHub-Flavored-Markdown pipe tables, which only display when the header, the `---` delimiter row, and every data row have the SAME number of columns.

- Give the delimiter row exactly one `---` cell per column, matching the header.
- If a cell's text contains a literal `|` (absolute value `|x|`, bitwise/logical or, alternatives `a|b`), escape it as `\\|`, or wrap the cell in `$...$` / backticks — an unescaped `|` silently adds a column and the whole table renders as raw text.
- Keep tables out of blockquotes and list items when possible; a top-level table is the most reliable.

Table typography, so tables read as calmly as the surrounding prose:

- Keep body cells plain text — no bold or italic inside cells — and style parallel tables in one answer identically. Bold is reserved for at most one genuine summary row (for example a totals row).
- Wrap literal identifiers taken from data — class labels, file names, dataset/field/column names — in backticks so they render as code and survive line-wrapping intact.
- Keep cells terse (a value or a short label) and put explanation in prose near the table. Numeric columns right-align automatically; never pad cells with spaces to align them.
"""

PLOT_WORKFLOW_GUIDANCE = """
For code, simulation, model-training, or algorithm-demonstration prompts that ask for plots,
run the code and create inspectable artifacts. In the final answer, mention each plot near
the explanation it supports, with a caption immediately after each figure reference. Keep
captions inline, not all at the end. Save static Matplotlib figures as publication-quality
outputs with at least 300 PPI: use the workspace matplotlibrc defaults
(`savefig.dpi: 300`) or call `fig.savefig(..., dpi=300, bbox_inches="tight")` /
`plt.savefig(..., dpi=300, bbox_inches="tight")` explicitly. For animated GIFs, also save a
static 300 PPI summary or key-frame figure when the user needs a durable publication-style
visual. For follow-up plot edits
such as "Change the plot to include error bars", edit
or rerun the existing plotting code and identify the updated artifact instead of starting an
unrelated analysis.
"""

SANDBOX_RUNTIME_GUIDANCE = """
For sandbox execution, do not wrap sandbox commands with shell timeout, gtimeout, or
execute timeout arguments. Long-running analysis is allowed. Prefer scientifically
meaningful convergence checks, smaller smoke-test data, checkpoints, and resumable
artifacts over arbitrary wall-clock caps.
"""

LONG_COMPUTE_RUNTIME_GUIDANCE = """
For long scientific/computational runs, do not launch the full grid, sweep, training job, or long integration as the first expensive command. First run a pilot timing pass on a tiny representative subset: one seed, one duration/window, one grid point, one batch, or about 1-5% of the data. Record elapsed seconds and extrapolate the estimated runtime for the full requested scope. Compare that estimate to the user/control-plane runtime budget when present, otherwise to the sandbox wall-clock cap from tool_capability_manifest. If the estimated runtime exceeds budget, shrink or chunk the grid: reduce seeds, durations, resolution, sample count, or batch size; run resumable chunks; and state the reduced scope explicitly. During execution, checkpoint durable progress under /outputs after each condition or batch, and at least every few minutes for long loops: metrics.jsonl/csv, partial tables, completed-batch manifests, model checkpoints, and enough parameters/seeds to resume. Keep scratch/temp files under /workspace, but progress evidence and final deliverables belong in /outputs.
"""

# Curated, decision-relevant subset of the sandbox image's preinstalled scientific
# stack (deploy/docker/deepagents-sandbox.Dockerfile). Surfaced to the coordinator so
# it reasons over what is actually importable in an offline container instead of
# guessing or trying to pip install. Keep in rough sync with the Dockerfile.
SANDBOX_KEY_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "matplotlib",
    "seaborn",
    "networkx",
    "torch",
    "torchvision",
    "opencv-python-headless",
    "Pillow",
    "imageio",
    "imagecodecs",
    "tifffile",
    "zarr",
    "ome-zarr",
    "dask",
    "xarray",
    "h5py",
    "pyarrow",
    "SimpleITK",
    "itk",
    "nibabel",
    "nilearn",
    "pydicom",
    "dicom2nifti",
    "highdicom",
    "monai",
    "dipy",
    "torchio",
    "bioio",
    "bioio-ome-tiff",
    "bioio-czi",
    "bioio-nd2",
    "openslide-python",
    "pyvips",
    "connected-components-3d",
    "mrcfile",
    "pynrrd",
    "ome-types",
    "roifile",
    "pymatgen",
    "pymatgen-analysis-defects",
    "ase",
    "spglib",
    "pycalphad",
    "scheil",
    "damask",
    "matminer",
    "orix",
    "kikuchipy",
    "diffsims",
    "defdap",
    "porespy",
]


def build_sandbox_resources_guidance(settings: RuntimeSettings) -> str:
    """One concise, run-adaptive paragraph telling the coordinator what the sandbox is.

    The exact envelope (cores, memory, shm, GPU, full package list) lives in
    tool_capability_manifest; this is the always-on summary so the model never tries to
    pip install in an offline container or write GPU code the sandbox can't run.
    """
    if settings.sandbox_gpus.strip():
        gpu_clause = "the sandbox has a GPU attached, so torch.cuda is usable for in-container code"
    else:
        gpu_clause = (
            "the sandbox itself has no GPU (torch runs on CPU) — reach GPU inference "
            "through the MegaSeg/RareSpot tools, not sandbox code"
        )
    network_clause = (
        "an OFFLINE container (no internet): you cannot pip/conda install at runtime — use "
        "only its preinstalled scientific Python stack (numpy/scipy/pandas, scikit-image/"
        "scikit-learn, matplotlib, torch/torchvision, SimpleITK/nibabel/pydicom/monai/dipy, "
        "bioio/tifffile/zarr/dask, and more)"
        if settings.sandbox_network == "none"
        else (
            "a NETWORK-ENABLED container (outbound internet is ON): you may fetch URLs, call "
            "HTTP APIs, and git clone at runtime, writing results under /workspace. To add a "
            "Python package use `pip install --user <pkg>` — the rootfs is read-only, so --user "
            "installs into /workspace and stays importable; a preinstalled scientific stack is "
            "also available, prefer it when it already has what you need"
        )
    )
    return (
        "The code sandbox is "
        + network_clause
        + ". It has generous CPU and memory for heavy compute, but "
        + gpu_clause
        + ". Each execute() is a fresh, ephemeral container: only /workspace and /outputs "
        "persist, and background processes (cmd &, nohup) do not survive a call, so "
        "checkpoint long runs to files. Write large temp/intermediate files under "
        "/workspace, not /tmp. Call tool_capability_manifest for the exact compute "
        "envelope (cores, memory, shm, GPU, full package list). "
        + LONG_COMPUTE_RUNTIME_GUIDANCE.strip()
    )


TEXT_ONLY_ARTIFACT_GUIDANCE = """
This deployment's active model is text-only. Do not call read_file on image, audio, video,
or PDF artifacts for visual inspection. Verify generated plots by checking source code,
input data, file existence, file size, and structured outputs such as result.json. Describe
figures from the computations that produced them, and keep artifact paths visible so the UI
can render the files inline.
"""

PRIOR_ARTIFACT_GUIDANCE = """
When this is a follow-up, prior durable artifacts may be listed in the run context. Use
artifact_manifest to inspect them. If code execution needs a prior file, call
stage_artifact_for_analysis first and use the returned /workspace/staged_artifacts path.
Do not say prior files are unavailable until artifact_manifest confirms none are present.
"""

UPLOADED_FILE_GUIDANCE = """
When the user selected or uploaded files for this run, their file IDs are listed in the run
context. Before writing code that reads those files, call stage_uploaded_files_for_analysis
and use the returned /workspace/staged_uploads paths. Do not guess upload paths or claim
uploaded files are unavailable until that tool reports they are missing.
"""

PAPER_REVIEW_GUIDANCE = """
For arXiv links and uploaded/local PDFs, use the paper tools instead of guessing from the
URL or filename. Ingest the paper first, search/read exact pages for claims, equations,
assumptions, methods, limitations, and figure references, and cite paper claims with page
grounding such as paper_id:p7. When a figure or equation page matters to the explanation,
call render_paper_page so the UI can display that page inline. For follow-up questions,
call paper_manifest first, then search_paper/read_paper_pages on the cached paper before
answering. Do not rely only on rendered page images or prior summaries for paper-specific
follow-up claims.
"""

BISQUE_GUIDANCE = """
When BisQue resource or dataset URIs are present, use the BisQue tools through the Go
control plane. Use bisque_search_resources for account-scoped queries, bisque_download_resource
to materialize selected BisQue resources into local file_ids before code execution, and
bisque_upload_files to send local V2 upload file_ids back to the linked BisQue account.
BisQue upload, dataset creation, module execution, and other mutating tools transmit data or
change remote account state. Call them only when the user explicitly asks for that remote
action. A request to save or produce durable outputs means local `/outputs` artifacts, which
Ultra collects automatically; it is not permission to upload them to BisQue or create a
BisQue dataset.
For requests about "my" BisQue resources, pass scope="owner"; for newest/recent resources,
pass sort="recent"; for file-type questions, pass extensions such as ["png"] or
["nii", "nii.gz", "nifti"] instead of estimating from broad search results.
For dataset questions ("do I have any datasets?"), pass resource_type="dataset"; for
account-wide counts ("how many images do I have?"), pass count_all=True and report the count.
For "how many images are IN dataset X", do NOT use a scoped image search or count_all — that
counts every image visible to the user (their own plus everything shared with them), not the
dataset's members. Instead call bisque_dataset_members with the dataset's resource_uniq (found
via resource_type="dataset"); report its member_count. To count graphical annotations: BisQue
annotations are gobjects attached to images (there is NO "annotation" resource type — never
search for one), and they are NESTED — the actual shapes (rectangles, polygons, points) sit under
class groups (e.g. gt2 -> burrow -> many rectangles). Use bisque_image_annotations for one image
and bisque_dataset_annotation_summary to answer "how many images in dataset X have annotations"
(it scans every member; report its images_with_annotations). Both return the annotation shape count
and a per-class label_counts / label_totals breakdown (e.g. burrow vs prairie_dog) — report that
breakdown when the user asks about annotation classes. Never infer annotation presence from tags,
filenames, or module runs, and never state an image has zero annotations without having called one
of these tools.
When the user explicitly asks to upload multiple related outputs and group them remotely,
use bisque_create_dataset with the resource_uri values from the upload responses.
To push generated figures/results to BisQue, use bisque_upload_workspace_files with the
output path you produced (e.g. /workspace/outputs/figure.png or /outputs/figure.png). When the
user asks to push a figure that was generated in an EARLIER turn, that file is a prior durable
artifact, not a current-run file: pass its durable artifact path (e.g.
outputs/ct_scan_visualization.png), its basename, or its artifact_id from artifact_manifest —
bisque_upload_workspace_files resolves prior artifacts and pushes them by artifact_id. Call
artifact_manifest first if you are unsure of the exact artifact path or id.
When you report an uploaded, pushed, downloaded, or searched BisQue resource to the user, give
the client_view_url returned by the tool — that is the canonical link that opens the resource in
the BisQue web viewer (https://<bisque>/client_service/view?resource=<resource_uri>). Do NOT
report the bare data_service resource_uri (the API URL) as the place to view it, and never
hand-construct a BisQue URL yourself: always use the exact client_view_url the tool returned.
Keep resource_uri only for follow-up tool calls such as bisque_create_dataset.
For numeric metadata comparisons (age, slice count, dose, year, study size, etc.) use the
metadata_filters argument, never tag_query relational operators: BisQue compares numeric
tags lexically, so tag_query='age:>50' incorrectly returns age 7 and omits age 100.
metadata_filters takes [{"tag","op","value"}] with op in eq/ne/gt/gte/lt/lte/contains.
Example: "all CT scans above age 50" -> bisque_search_resources(resource_type="image",
tag_query="modality:CT", metadata_filters=[{"tag":"age","op":"gt","value":"50"}],
scope="owner", count_all=True). Keep string-equality tags (modality:CT) in tag_query for
server-side narrowing and put the numeric comparison in metadata_filters.
For BisQue module-execution (MEX) runs, use bisque_module_runs. Answer "what modules have I
run recently on BisQue?" by calling it with no mex_uri (returns each run's module_name, status,
and time); pass status="FINISHED" to filter. To pull a result from a finished run — e.g. a
segmentation mask — call bisque_module_runs(mex_uri=<the run's uri>) to get its outputs, then
take the output's resource_uri and call bisque_download_resource to materialize it into Ultra
for analysis, or report the output's client_view_url so the user can view it on BisQue. Do not
treat a mex as an image; its results live in its outputs.
Never expose or ask for BisQue credentials in the answer; the control plane owns account auth.
"""

# Shown instead of BISQUE_GUIDANCE when BisQue tools are NOT registered for this run,
# so the model never names or pretends to call a BisQue tool it does not have.
BISQUE_UNLINKED_HINT = """
BisQue tools are not connected for this run. If the user asks about BisQue — their images,
datasets, or module-execution (MEX) runs — do not name or attempt to call any bisque_* tool and
do not claim you cannot find a tool. Instead, tell the user to link their BisQue account from the
Settings menu to enable BisQue access, then offer to retry.
"""

EPISODIC_GUIDANCE = """
You can recall this researcher's own past Ultra sessions with search_past_research. Call it
whenever a request depends on earlier work — phrases like "last time", "previously", "earlier",
"what did we conclude", "the parameters from my last … run", "compare my <year> and <year>
results", or a reference to a dataset/run you did not see in this conversation. Pass a focused
query (a dataset name, method, or topic) and read the returned dated summaries before answering;
use since_days to bound recency when the user names a timeframe. Ground any recalled claim in the
returned results and cite it by date or title — never invent a prior conclusion that is not there.
Prefer search_past_research over guessing, and over asking the user to repeat context they already
established in a previous session.
"""


RESULTS_CONTRACT_GUIDANCE = """
Results contract (Pro intelligence — mandatory for the final chat answer, not only the report):
- Report every decision-relevant estimate as mean ± spread from at least 3 seeds or initial
  conditions and at least 2 observation durations, including clearly-resolved cases, and
  carry the spread column into any CSV/table artifact.
- State the decision rule verbatim and apply it per row: assign a definitive class only when
  |estimate| > 3× its spread AND at least one independent non-primary discriminator agrees;
  otherwise label the row marginal.
- Classification tables include the parameters that produced each row: initial conditions,
  duration, step size, estimate ± spread, discriminators agreeing, class, confidence.
- When a structural claim depends on a projection, wrapping, or binning choice (for example
  strobe clusters modulo 2π), report the count under both views or explain the aliasing.
- Probe at least 3 initial conditions at borderline parameter values and state whether the
  classification is basin- or IC-dependent.
- Name the canonical system when one matches and compare against commonly reported behavior;
  never fabricate citations — "commonly reported" phrasing with a confidence note is correct.
- Include a short Limitations paragraph in the chat answer itself.
- Durable /outputs hold only final code, verification scripts, data tables, figures, and the
  report; scratch or diagnostic scripts stay under /workspace (use /workspace/diagnostics/).
"""

MATERIALS_RESULTS_CONTRACT_GUIDANCE = """
Materials results contract (Pro intelligence — mandatory for the final chat answer and artifacts):
- Match validation to the materials modality. Deterministic crystallography, simulated XRD,
  and CALPHAD calculations are verified with tolerance/parameter stability, physical
  invariants, and provenance checks; do not apply the dynamical-systems replication contract.
- Identify every input by staged path or artifact ID and record its format, relevant metadata,
  and a checksum for any user-supplied CIF, diffraction file, spectrum, orientation map, volume,
  or thermodynamic database used to support a conclusion.
- For CIF/structure and symmetry claims, report composition, occupancies/disorder assumptions,
  cell setting, symmetry library, symprec and angle tolerance, and the result across a stated
  tolerance sweep. Preserve ordered-vs-disordered site identity and flag unstable assignments.
- For simulated XRD, state the radiation source and wavelength, structure/occupancy assumptions,
  two-theta range, and broadening model. Emit a peak table with two-theta, relative intensity,
  and indexed hkl values; label calculated patterns as simulated rather than experimental.
- For CALPHAD, use `calphad_inspect_database` followed by `calphad_run_equilibrium` or
  `calphad_run_scheil`; a raw generic `execute` solve is a replay/debug aid and does not satisfy
  the verified backend contract. Record the content-addressed evidence artifact, exact TDB name/version/source/checksum,
  components, candidate phases, composition basis, temperature, pressure, and numerical conditions.
  Retain `VA` whenever the authenticated inspection inventory declares it (without an X(VA) axis),
  and use the typed runtime's canonical first-sorted physical dependent component.
  Verify finite phase vertices and chemical potentials, phase and per-vertex composition closure,
  vertex-weighted bulk mass balance, and Gibbs-Euler consistency. Never use pycalphad package test
  fixtures as a scientific database or silently replace a missing database with a heuristic.
- For a classic Scheil--Gulliver solidification path, use `calphad_run_scheil` after the same
  authenticated inspection. Supply one scalar bulk mole-fraction point, a defensible phase set
  containing `LIQUID`, an all-liquid start temperature, a bounded temperature step, and a
  residual-liquid criterion. Pressure is fixed at 101325 Pa; VA is retained without X(VA).
  Require pointwise phase/composition closure and reconstructed elemental inventory closure across
  all retained increments. Report the four Scheil assumptions and separate numerical convergence
  from assessment validity. Never label this result as back diffusion, finite-rate solid diffusion,
  precipitation, or phase field.
- For mobility/diffusion/back-diffusion/precipitation kinetics, use the typed
  `materials_transport_coefficients`, `materials_run_diffusion_1d`, and
  `materials_run_binary_precipitation_kwn` tools with an explicitly selected governed TDB.
  Never substitute generic `execute` output for their evidence. Treat back diffusion as
  post-solidification single-phase diffusion only; never imply a moving solid/liquid interface.
  Report the Kawin/NumPy/pycalphad versions, selected database provenance, fixed pressure,
  mass-closure checks, grid/bin convergence status, assumptions, and content-addressed artifact.
- For crystal-plasticity geometry, resolved shear, Schmid factors, and CPFE input readiness, use
  `materials_analyze_crystal_slip` and `materials_validate_cpfe_contract` directly. Never make a
  code-runner discover these Python APIs. Preserve the required phase ID, active
  crystal-to-sample convention, sample-frame stress, explicit units, selected canonical family,
  and HCP `c_over_a`. A valid CPFE contract is structural input validation only: the typed tool
  must report execution as unsupported until a qualified constitutive integrator and FE/spectral
  solver are bound, and must never fabricate stress-strain or convergence output.
  When durable outputs are requested and either typed tool returns
  `analysis_artifact.canonical_json` and
  `materials_validation_artifact.canonical_json`, write those exact strings directly to the
  requested analysis JSON and `/outputs/materials_validation.json`; do not introspect the Python
  validation API, reconstruct verdict fields, or delegate that serialization to code-runner.
  A deterministic typed input rejection is terminal evidence for that request: report the one
  error without repeating it across seeds, durations, families, or substitute inputs.
- Treat DFT, molecular dynamics, phonons, and formation-energy requests as an explicit capability
  boundary: this runtime has no production DFT or molecular-dynamics engine. Mark the requested
  calculation unsupported unless a real solver or interatomic potential and its provenance are
  available; never substitute a heuristic result and present it as a completed calculation.
- For EBSD and microstructure estimates, state symmetry/reference-frame conventions, voxel size,
  analyzed volume and sample count; report distributions and boundary exclusions, and sweep the
  decisive segmentation/indexing parameter over a defensible range. Tie confidence to stability
  against that sweep and to the named null or independent measurement.
- State units for every decision-relevant quantity, package versions for every named method, the
  checks that passed or failed, and a short Limitations paragraph in the chat answer itself.
- When durable materials evidence is requested, write /outputs/materials_validation.json with the canonical
  ultra_deepagents.materials.validation schema: per-check validator_id/outcome/observed/expected,
  tolerance rationale, units, versions, and hashed evidence; report its scientific_status
  separately from orchestration run_status, record capability_supported and
  contradiction_failures, and fail closed on missing required validators. Read the relevant
  materials skill before analysis. Build the verdict only with `assess_scientific_status`,
  serialize only that returned assessment with `canonical_record_json`, and require
  `parse_assessment_record` to accept the exact final bytes. Never invent a substitute schema or
  edit top-level verdict fields; a parse failure makes the claim unverified.
- Durable /outputs hold final code, validation records, tables, figures, and reports; scratch or
  diagnostic scripts stay under /workspace (use /workspace/diagnostics/).
"""

SCOPED_DELEGATION_CONFIDENCE_LEVELS = ["high", "medium", "low", "unresolved"]

SCOPED_DELEGATION_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One concise paragraph with the subtask result.",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific numerical, textual, or methodological findings — short phrases, not raw output dumps; reference large logs/tables by /workspace path.",
        },
        "artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace or durable output path, when one was produced or inspected.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What the artifact contains and why it matters.",
                    },
                },
                "required": ["path", "description"],
                "additionalProperties": False,
            },
            "description": "Relevant files, reports, figures, or staged inputs.",
        },
        "failures": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Errors, missing inputs, or caveats — short phrases, not full tracebacks; reference large logs by /workspace path. Use an empty array when none occurred.",
        },
        "confidence": {
            "type": "string",
            "enum": SCOPED_DELEGATION_CONFIDENCE_LEVELS,
            "description": (
                "Earned confidence level. Use high only when the decision rule passes "
                "(estimate magnitude exceeds 3x its spread, or equivalent); use "
                "unresolved when the evidence cannot distinguish the hypotheses."
            ),
        },
        "confidence_basis": {
            "type": "string",
            "description": "One sentence tying the confidence level to the decision rule or evidence.",
        },
    },
    "required": [
        "summary",
        "key_findings",
        "artifacts",
        "failures",
        "confidence",
        "confidence_basis",
    ],
    "additionalProperties": False,
}

BASE_SUBAGENTS = [
    {
        "name": "literature-reviewer",
        "description": (
            "Reads arXiv/PDF papers with narrow paper tools, searches exact pages, extracts "
            "equations/figures/methods, and returns page-grounded review notes. Use it "
            "whenever a request depends on reading one or more papers in depth — extracting "
            "claims, methods, equations, or figures from specific pages — so the long page "
            "text stays out of the coordinator's context."
        ),
        "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
        "system_prompt": (
            "You are a page-grounded literature reviewer for research users. Use only the "
            "provided paper tools to ingest, search, read, and render papers. Never make a "
            "paper-specific claim unless it is grounded in an ingested paper page or chunk. "
            "Return concise notes with: papers reviewed, key claims, methods, equations, "
            "figures/pages worth rendering, limitations, relevance to the user's task, "
            "missing evidence, and confidence. Cite claims as paper_id:pN. If the user asks "
            "about a figure, equation, proof, or architecture diagram, call render_paper_page "
            "for the relevant page. IMPORTANT: return only the distilled notes — do NOT paste "
            "full page text or long quotes into your fields; cite by paper_id:pN and keep the "
            "summary under ~200 words."
        ),
    },
]

SCOPED_DELEGATION_CONTEXT_TOOLS = {
    "artifact_manifest",
    "calphad_inspect_database",
    "calphad_run_equilibrium",
    "calphad_run_scheil",
    "materials_run_binary_precipitation_kwn",
    "materials_run_diffusion_1d",
    "materials_analyze_crystal_slip",
    "materials_calculate_diffraction_profile_metrics",
    "materials_convert_uniform_corrosion",
    "materials_evaluate_mode_i_lefm",
    "materials_evaluate_norton_arrhenius_creep",
    "materials_evaluate_oxidation_mass_gain",
    "materials_fit_held_out_rigid_registration",
    "materials_fit_paris_law",
    "materials_processing_method_support",
    "materials_transport_coefficients",
    "materials_validate_cpfe_contract",
    "inspect_selected_sensor_series",
    "stage_artifact_for_analysis",
    "stage_uploaded_files_for_analysis",
}

# Subagents that do computational work and benefit from the rigor/reporting
# skills. literature-reviewer is page-grounded paper review and is excluded so it
# does not re-pay the SkillsMiddleware overhead for skills it would never activate.
QWEN_CODE_RUNNER_NAME = "qwen-code-runner"

_SKILL_BEARING_SUBAGENTS = {"code-runner", QWEN_CODE_RUNNER_NAME}

# A single, unambiguous computational delegate. (A separate data-analyst was
# dropped: it had a tool set identical to code-runner — deepagents injects
# execute + filesystem into every subagent — and co-registered inseparably with
# it, so the model never chose it. code-runner absorbs the data-inspection role.)
SCOPED_DELEGATION_SUBAGENTS = [
    {
        "name": "code-runner",
        "description": (
            "Runs focused sandbox execution, debugging, reproducibility checks, plotting, "
            "numerical experiments, and dataset/artifact inspection, then returns concise "
            "findings and artifact paths. Use it whenever a request needs running, debugging, "
            "profiling, or plotting code, a numerical experiment, a reproducibility check, or "
            "staging and inspecting uploaded/prior data files."
        ),
        "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
        "system_prompt": (
            "You are Ultra's scoped code-runner subagent. Use built-in filesystem and "
            "sandbox execution tools for focused code, statistics, plotting, simulation, "
            "model-training, or reproducibility subtasks. Use the provided context tools to "
            "stage selected uploads and stage prior artifacts before code reads them; avoid "
            "guessing paths. For pure data-inspection subtasks you may stage and summarize "
            "files (schema, shape, metadata, quality concerns) without running heavy code. "
            "Keep intermediate files under /workspace and durable outputs under "
            "/outputs when available. Preserve the user's requested compute scope: do not add "
            "longer durations, finer step sizes, more seeds, or broader convergence sweeps unless "
            "the subtask explicitly asks for them or a short smoke check reveals a material "
            "uncertainty. " + LONG_COMPUTE_RUNTIME_GUIDANCE.strip() + " Return a concise "
            "final report with commands/scripts run, key numerical results, generated "
            "artifact paths, failures, and confidence. IMPORTANT: return only the distilled "
            "result — do NOT paste raw stdout, full tracebacks, or large tables into your "
            "fields; write large data to /workspace and reference it by path, and keep the "
            "summary under ~200 words. Set confidence=high only when "
            "your evidence passes a stated decision rule (for example estimate "
            "magnitude above 3x its spread); otherwise use medium, low, or unresolved, "
            "and explain the basis in confidence_basis. Do not perform broad literature "
            "review, BisQue account operations, RareSpot inference, or user-facing final "
            "synthesis; the coordinator reconciles your result."
        ),
    },
]


QWEN_CODE_RUNNER = {
    "name": QWEN_CODE_RUNNER_NAME,
    "description": (
        "Qwen3.6-powered multimodal coding agent for focused sandbox execution, "
        "debugging, plotting, artifact inspection, and image-aware code checks. Use it "
        "when code work benefits from reading visual artifacts, screenshots, generated "
        "plots, figures, or image-like data, or when an independent second coding pass "
        "should run on the Qwen endpoint."
    ),
    "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
    "system_prompt": (
        "You are Ultra's Qwen multimodal code-runner. Use built-in filesystem and "
        "sandbox execution tools for focused coding, debugging, plotting, simulation, "
        "model-training, reproducibility, or artifact-inspection subtasks. You run on "
        "the Qwen3.6 multimodal endpoint, so you may inspect visual artifacts when the "
        "subtask genuinely depends on plots, screenshots, figures, or images; still "
        "prefer numeric/textual verification for measurable claims and do not turn a "
        "visual check into an open-ended diagnosis. Use the provided context tools to "
        "stage selected uploads and prior artifacts before code reads them; avoid "
        "guessing paths. Keep intermediate files under /workspace and durable outputs "
        "under /outputs when available. Preserve the user's requested compute scope: "
        "do not add longer durations, finer step sizes, more seeds, or broader "
        "convergence sweeps unless the subtask explicitly asks for them or a short "
        "smoke check reveals material uncertainty. "
        + LONG_COMPUTE_RUNTIME_GUIDANCE.strip()
        + " Return a concise final report with "
        "commands/scripts run, key numerical results, visual/artifact findings, "
        "generated artifact paths, failures, and confidence. Do not perform broad "
        "literature review, BisQue account operations, RareSpot inference, or "
        "user-facing final synthesis; the coordinator reconciles your result."
    ),
}


# The "second pair of eyes": a vision-language reasoner the text coordinator
# delegates visual-judgment to. It SEES pixels via the inspect_images tool (a
# self-contained call to the on-prem Qwen3.6-27B VLM); its own loop model stays the
# inherited text model, so it needs no per-subagent VLM model and no multimodal
# middleware exemption (the tool returns text). Detection/counting stays with the
# specialist models — this is a reasoner/verifier, not a detector.
VISION_SUBAGENT = {
    "name": "vision-reasoner",
    "description": (
        "A second pair of eyes that actually SEES images with a vision-language model. "
        "Delegate visual-judgment tasks: verify whether a detector's box is a real object "
        "or a false positive (it looks closer at a zoomed crop), describe an image in "
        "detail, read or verify a plot/scientific figure (axes, values, error bars), OCR "
        "figure text, extract a PDF table through the dedicated provenance-sealed table "
        "tool, give an advisory 'what is this structure?' hypothesis, or compare "
        "multiple images. Use it whenever a decision depends on what is actually in an "
        "image and you (the coordinator) cannot see pixels. Do NOT use it to COUNT many "
        "small objects, measure pixels/areas/distances, or produce/correct bounding boxes "
        "— those stay with the specialist detectors (YOLO/RareSpot/MegaSeg)."
    ),
    "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
    "system_prompt": (
        "You are Ultra's vision-reasoner subagent — a careful second pair of eyes backed by "
        "a vision-language model. You can only SEE pixels by calling inspect_images; reason "
        "over what it reports, then return the distilled structured result.\n"
        "GROUNDED by default: inspect_images runs WITHOUT extended thinking unless you opt into "
        "mode='reasoning'/'precise'. For an open-ended 'what does this image show / does it show "
        "condition D' judgment, KEEP it grounded and report ONLY what the pixels support — "
        "extended thinking makes this model reason itself INTO a plausible-but-false finding "
        "(it will narrate an entire condition that is not present). Use reasoning/precise ONLY "
        "for a narrow, specific check: one detection crop, or one exact figure/number read.\n"
        "How to work: (1) From the goal, identify the image path(s) (/workspace/..., "
        "/outputs/...) and the precise visual question. (2) Call inspect_images with a "
        "focused question; to verify a single detection, pass its bbox so the object is "
        "cropped and zoomed. For MORE THAN ~3-4 images/slices you MUST screen first: ONE "
        "screen_images call over the whole set (fast, no extended thinking, batched + "
        "concurrent) returns a per-image line; then deep-read with inspect_images (grounded — "
        "the default) ONLY on the slices/items the screen flags as decisive or genuinely "
        "ambiguous. NEVER "
        "loop deep inspect_images over a whole stack of slices — that is the slow, "
        "wrong path that wastes the fast screen. How MANY deep reads is yours to choose by what "
        "the screen flags: a clear case needs ~2-3, a hard or multi-finding case needs more, "
        "and for false-positive verification deep-read every genuinely ambiguous positive — "
        "TRUST the screen result for the rest. (3) Synthesize a concise verdict in your "
        "fields. Be precise and "
        "conservative: only assert a property (e.g. a 'declining trend') when the image clearly "
        "shows it; ambiguous or trendless images (e.g. random scatter) are NOT positives.\n"
        "For false-positive verification, ask inspect_images for a structured verdict and "
        "require >=2 concrete visual observations; if the evidence is thin or the crop is "
        "ambiguous, report uncertain rather than guessing. The detector over-detects and is "
        "not calibrated — do not anchor on its raw confidence.\n"
        "HARD LIMITS — you are NOT a detector. Never COUNT many small objects, never measure "
        "pixels/areas/distances, never emit or correct bounding-box coordinates, never hunt "
        "for 'missed' detections across a whole image. If asked to, return "
        "confidence=unresolved with a failures entry redirecting to the specialist detector "
        "— never a fabricated count or measurement.\n"
        "For a table in an ingested PDF, use extract_paper_table_evidence with the exact "
        "paper_id and one-based page. Never use free-form inspect_images output as a durable "
        "table transcription. Preserve unreadable cells as null and describe the sealed result "
        "as model-observed until born-digital text or independent human/source validation agrees. "
        "Return the tool-written sealed-evidence and raw-response /outputs paths and SHA-256 "
        "values with the distilled result.\n"
        "Keep summary under ~200 words; return only the distilled verdict (do not paste long "
        "model output). Set confidence=high only when the visual evidence is unambiguous, "
        "else medium/low/unresolved with the basis in confidence_basis. The coordinator "
        "reconciles your verdict; it augments, never replaces, the detector's measurement."
    ),
}

_VISION_GOAL_TOKENS = (
    "image",
    "images",
    "photo",
    "picture",
    "figure",
    "plot",
    "chart",
    "diagram",
    "overlay",
    "screenshot",
    "snapshot",
    "visual",
    "look at",
    "second pair of eyes",
    "false positive",
    "false-positive",
    "verify the detection",
    "detection",
    "detections",
    "detector",
    "yolo",
    "rarespot",
    "prairie dog",
    "prairie dogs",
    "burrow",
    "burrows",
    "microscopy",
    "histology",
    "scan",
    "segmentation",
    "cell",
    "cells",
)

VISION_DELEGATION_GUIDANCE = """
A vision-reasoner subagent is available — a second pair of eyes that can SEE images (you
cannot). Delegate to it via the task tool whenever a decision depends on what is actually in
an image: verifying whether a detector flagged a false positive (it inspects a zoomed crop of
the box), describing an image in detail, reading/verifying a plot or scientific figure, OCRing
figure text, or comparing images. Pass the image path(s) (/workspace/... or /outputs/...) and a
precise visual question; for a detection, include its bounding box. Do NOT ask it to count many
small objects, measure pixels/areas, or produce coordinates — that is the specialist detectors'
job; vision-reasoner verifies and reasons, it does not replace the detector's measurement. Its
verdict augments (never overwrites) detector counts; reconcile it as a second opinion.
When you hand it a SET of slices/images, tell it to screen the whole set in ONE pass first and
then deep-inspect only the few decisive slices — do not delegate "inspect each." A measurable or
quantitative question (an index, ratio, count, or slope with a known reference range) is YOURS to
COMPUTE in the sandbox; the vision pass corroborates the number, it does not produce it.
For a table in an ingested PDF, delegate its paper_id, one-based page, and exact table selector;
the vision-reasoner must use extract_paper_table_evidence, preserve unreadable cells as null, and
return the sealed model-observation evidence plus its tool-written sealed-evidence and raw-response
/outputs paths and SHA-256 values before you compute derived quantities.
""".strip()

QWEN_CODE_DELEGATION_GUIDANCE = """
Two coding delegates may be available: `code-runner` and `qwen-code-runner`. Use
`code-runner` for the advanced reasoning coding pass: implementation, debugging,
numerical experiments, reproducibility checks, and text/numeric artifact inspection. Use
`qwen-code-runner` when the same code work benefits from a multimodal second pass over
plots, screenshots, figures, or image-like outputs, or when an independent coding check
should run on the Qwen endpoint. For independent subtasks, you may fan out to both and
reconcile their findings yourself; keep each delegation focused and bounded.
""".strip()


def _should_register_vision_subagent(
    context: AgentRunContext | None, settings: RuntimeSettings
) -> bool:
    """Register the vision-reasoner when the VLM is enabled AND the run plausibly
    involves images. Unlike ``looks_scoped_delegation_goal`` (which EXCLUDES
    rarespot/prairie/burrow goals), this INCLUDES them — verifying a RareSpot
    detection is the headline vision use case."""
    if not settings.qwen_vlm_enabled or context is None:
        return False
    if not settings.qwen_vlm_base_url:
        logging.getLogger(__name__).warning(
            "qwen_vlm_enabled but no QWEN_VLM_BASE_URL configured; "
            "vision-reasoner will NOT be registered."
        )
        return False
    # Fail safe, not silently broken: if the VLM is enabled but no key resolved, do not
    # register a tool that can only error on first use; warn the operator instead.
    if not settings.qwen_vlm_api_key or settings.qwen_vlm_api_key == "EMPTY":
        logging.getLogger(__name__).warning(
            "qwen_vlm_enabled but no API key resolved (set QWEN_VLM_API_KEY or "
            "QWEN_VLM_API_KEY_FILE); vision-reasoner will NOT be registered."
        )
        return False
    if context.selected_file_ids or context.selected_resource_uris or context.selected_dataset_uris:
        return True
    goal = " ".join(str(context.goal or "").lower().split())
    if _has_ingested_papers(context) and any(
        token in goal for token in ("table", "page", "paper", "pdf")
    ):
        return True
    return any(token in goal for token in _VISION_GOAL_TOKENS)


def _should_register_qwen_code_runner(
    context: AgentRunContext | None, settings: RuntimeSettings
) -> bool:
    if not _should_register_scoped_delegation_subagents(context):
        return False
    if not settings.qwen_vlm_enabled:
        return False
    if not settings.qwen_vlm_base_url:
        logging.getLogger(__name__).warning(
            "qwen_vlm_enabled but no QWEN_VLM_BASE_URL configured; "
            "qwen-code-runner will NOT be registered."
        )
        return False
    if not settings.qwen_vlm_api_key or settings.qwen_vlm_api_key == "EMPTY":
        logging.getLogger(__name__).warning(
            "qwen_vlm_enabled but no API key resolved (set QWEN_VLM_API_KEY or "
            "QWEN_VLM_API_KEY_FILE); qwen-code-runner will NOT be registered."
        )
        return False
    return True


_ULTRA_HARNESS_PROFILE_REGISTERED = False


def ensure_ultra_harness_profile() -> None:
    """Keep long code/tool work in the coordinator unless we opt into a narrow subagent."""
    global _ULTRA_HARNESS_PROFILE_REGISTERED
    if _ULTRA_HARNESS_PROFILE_REGISTERED:
        return
    register_harness_profile(
        "openai",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _ULTRA_HARNESS_PROFILE_REGISTERED = True


def build_subagents(
    paper_tools: Sequence[BaseTool | Any] | None = None,
    *,
    context: AgentRunContext | None = None,
    context_tools: Sequence[BaseTool | Any] | None = None,
    text_only_model: bool = True,
    skills_sources: Sequence[str] | None = None,
    vision_tools: Sequence[BaseTool | Any] | None = None,
    qwen_coding_model: BaseChatModel | Any | None = None,
) -> list[dict[str, Any]]:
    subagents: list[dict[str, Any]] = []

    if paper_tools:
        literature = dict(BASE_SUBAGENTS[0])
        literature["response_format"] = deepcopy(literature["response_format"])
        literature["tools"] = list(paper_tools)
        subagents.append(literature)

    if vision_tools:
        # The vision-reasoner sees pixels only through inspect_images; its loop model
        # is the inherited text model, so the text-only middleware below is harmless
        # (no image blocks ever reach it) and no per-subagent VLM model is needed.
        vision = dict(VISION_SUBAGENT)
        vision["response_format"] = deepcopy(vision["response_format"])
        vision["tools"] = list(vision_tools)
        subagents.append(vision)

    if _should_register_scoped_delegation_subagents(context):
        delegation_context_tools = _filter_tools_by_name(
            context_tools or (),
            SCOPED_DELEGATION_CONTEXT_TOOLS,
        )
        for template in SCOPED_DELEGATION_SUBAGENTS:
            subagent = dict(template)
            if "response_format" in subagent:
                subagent["response_format"] = deepcopy(subagent["response_format"])
            if subagent["name"] in _SKILL_BEARING_SUBAGENTS:
                subagent["tools"] = delegation_context_tools
            subagents.append(subagent)
        if qwen_coding_model is not None:
            qwen = dict(QWEN_CODE_RUNNER)
            qwen["response_format"] = deepcopy(qwen["response_format"])
            qwen["model"] = qwen_coding_model
            qwen["tools"] = delegation_context_tools
            subagents.append(qwen)

    for subagent in subagents:
        # Only the computational subagents benefit from the rigor/reporting
        # protocols; literature-reviewer is page-grounded paper review and would
        # just re-pay the per-subagent SkillsMiddleware overhead without ever
        # activating them. Subagents share the parent backend's /skills/ route.
        if skills_sources and subagent["name"] in _SKILL_BEARING_SUBAGENTS:
            subagent["skills"] = list(skills_sources)
        if text_only_model and subagent["name"] != QWEN_CODE_RUNNER_NAME:
            subagent["middleware"] = [TextOnlyMultimodalMiddleware()]
    return subagents


def build_async_subagents(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext | None = None,
) -> list[dict[str, Any]]:
    """Configured remote/background Deep Agents subagents.

    These are intentionally operator-configured rather than inferred from a
    prompt. Deep Agents routes specs with ``graph_id`` through
    AsyncSubAgentMiddleware, which exposes start/check/update/cancel/list tools.
    """
    if not _should_register_async_delegation_subagents(context):
        return []
    return [
        _normalize_configured_async_subagent(spec, index=index)
        for index, spec in enumerate(settings.async_subagents)
    ]


def _normalize_configured_async_subagent(spec: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"RuntimeSettings.async_subagents[{index}] must be an object")
    prefix = f"RuntimeSettings.async_subagents[{index}]"
    normalized = dict(spec)
    normalized["name"] = _required_configured_async_subagent_string(
        spec,
        "name",
        prefix=prefix,
    )
    normalized["description"] = _required_configured_async_subagent_string(
        spec,
        "description",
        prefix=prefix,
    )
    normalized["graph_id"] = _required_configured_async_subagent_string(
        spec,
        "graph_id",
        prefix=prefix,
    )
    return normalized


def _required_configured_async_subagent_string(
    spec: dict[str, Any],
    field: str,
    *,
    prefix: str,
) -> str:
    if field not in spec or spec[field] is None:
        raise ValueError(f"{prefix}.{field} is required")
    value = spec[field]
    if not isinstance(value, str):
        raise ValueError(f"{prefix}.{field} must be a non-empty string")
    value = value.strip()
    if not value:
        raise ValueError(f"{prefix}.{field} is required")
    return value


def _filter_tools_by_name(
    tools: Sequence[BaseTool | Any],
    names: set[str],
) -> list[BaseTool | Any]:
    return [tool for tool in tools if getattr(tool, "name", "") in names]


_MATERIALS_METHOD_TOKENS = (
    "4d stem",
    "4d-stem",
    "acoustic emission",
    "antisite",
    "atom probe",
    "back diffusion",
    "calphad",
    "cif",
    "cp2k",
    "cpfe",
    "corrosion",
    "creep",
    "crystal plasticity",
    "crystal structure",
    "crystallographic",
    "crystallography",
    "defect",
    "diffraction",
    "diffusion mobility",
    "dft",
    "density functional",
    "density-functional theory",
    "dream.3d",
    "dream3d",
    "eds",
    "ebsd",
    "energy above hull",
    "energy-above-hull",
    "formation energy",
    "fracture",
    "fatigue",
    "frenkel",
    "grain boundary",
    "grain size",
    "inverse pole figure",
    "interstitial",
    "ipf map",
    "kikuchi",
    "lammps",
    "materials science",
    "microstructure",
    "misorientation",
    "molecular dynamics",
    "molecular-dynamics",
    "oxidation",
    "phase fraction",
    "phase field",
    "phase diagram",
    "phonon",
    "photodiode",
    "pole figure",
    "porosity",
    "poscar",
    "precipitation",
    "process telemetry",
    "raman",
    "resolved shear",
    "rietveld",
    "saed",
    "scheil",
    "schmid factor",
    "sensor data",
    "sensor telemetry",
    "slip system",
    "slip-system",
    "solidification",
    "space group",
    "spacegroup",
    "substitutional",
    "substitution",
    "substitute",
    "spectroscopic",
    "spectroscopy",
    "stress-strain",
    "tdb",
    "thermodynamic database",
    "thermodynamic",
    "thermodynamics",
    "thermal imaging",
    "thermocouple",
    "transmission electron microscopy",
    "vacancy",
    "quantum espresso",
    "vasp",
    "x-ray diffraction",
    "xps",
    "xrd",
    "waveform",
)

_COMPUTATIONAL_ACTION_TOKENS = (
    "analy",
    "build",
    "calculat",
    "characteriz",
    "classif",
    "comput",
    "construct",
    "convert",
    "creat",
    "determin",
    "estimat",
    "evaluat",
    "extract",
    "featur",
    "fit",
    "find",
    "generat",
    "identif",
    "index",
    "inspect",
    "measur",
    "model",
    "enumerat",
    "plot",
    "predict",
    "process",
    "quantif",
    "refine",
    "render",
    "run",
    "segment",
    "simulat",
    "solve",
    "sweep",
    "substitut",
    "validat",
)


def _plural_tolerant_method_pattern(token: str) -> re.Pattern[str]:
    """Word-bounded matcher that also accepts the token's regular plural.

    The flagship phrasing "equilibrium phase fractions" previously missed the
    singular token "phase fraction" because the trailing word boundary rejected
    the plural. Regular ``-s``/``-es`` and ``-y``/``-ies`` inflections match;
    the leading boundary still keeps acronyms like ``cif`` exact.
    """
    escaped = re.escape(token)
    if token.endswith("y"):
        escaped = escaped[:-1] + "(?:y|ies)"
    else:
        escaped += "(?:e?s)?"
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


_MATERIALS_METHOD_PATTERNS = tuple(
    _plural_tolerant_method_pattern(token) for token in _MATERIALS_METHOD_TOKENS
)


def looks_materials_computational_goal(goal: str) -> bool:
    """Return whether the request calls for a named materials computation.

    The general delegation classifier historically keyed on words such as
    ``simulation`` and ``analyze``. Canonical materials requests instead say
    "identify this CIF", "calculate an XRD pattern", or "build a CALPHAD
    phase diagram", so they silently ran without a computational delegate.
    Requiring both a domain method and an action avoids registering one for a
    purely definitional question such as "what is XRD?". Method tokens are
    word-bounded so the acronym ``CIF`` cannot match an unrelated word such as
    ``specific``.
    """
    request_text = request_classification_text(goal)
    lowered = " ".join(request_text.lower().split())
    has_materials_method = any(pattern.search(lowered) for pattern in _MATERIALS_METHOD_PATTERNS)
    has_action = any(
        re.search(rf"\b{re.escape(token)}", lowered) for token in _COMPUTATIONAL_ACTION_TOKENS
    )
    strong_materials_pair = any(
        left.search(lowered) and right.search(lowered)
        for left, right in (
            (re.compile(r"\bspace[- ]?group\b"), re.compile(r"\bcif\b")),
            (
                re.compile(r"\bphases?\b"),
                re.compile(r"\b(?:xrd|x[- ]?ray diffraction)\b"),
            ),
            (re.compile(r"\bgrains?\b"), re.compile(r"\bdream(?:\.?3d|\.3d)\b")),
        )
    )
    dft_shorthand = bool(
        re.search(r"\bdft\b", lowered)
        and (
            has_action
            or re.search(
                r"\b(?:silicon|gallium|gan|sic|alloy|crystal|material|"
                r"band structure|electronic structure|poscar|cif)\b",
                lowered,
            )
        )
    )
    md_shorthand = bool(
        re.search(r"\bmd\b", lowered)
        and re.search(r"\b(?:simulat|run|model|trajectory|atomistic|lammps)", lowered)
    )
    return (
        (has_materials_method and has_action)
        or strong_materials_pair
        or dft_shorthand
        or md_shorthand
    )


def _selection_suggested_domain(context: AgentRunContext | None) -> str:
    if context is None or not isinstance(context.selection_context, dict):
        return ""
    raw = str(context.selection_context.get("suggested_domain") or "")
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if normalized in {"material", "materials", "material_science", "materials_science"}:
        return "materials"
    return normalized


def is_materials_context(context: AgentRunContext | None) -> bool:
    """Treat the UI hint as routing evidence, with prompt detection as fallback.

    A CALPHAD-shaped run (goal tokens or a selected thermodynamic database)
    also counts: a run that carries the typed CALPHAD tools must carry the
    materials skill routing and, on Pro, the materials results contract —
    previously "equilibrium phase fractions" registered the tools without
    either.
    """
    if context is None:
        return False
    return (
        _selection_suggested_domain(context) == "materials"
        or looks_materials_computational_goal(context.goal)
        or _should_register_calphad_tools(context)
        or _should_register_kinetics_tools(context)
        or _should_register_crystal_plasticity_tools(context)
        or _should_register_degradation_tools(context)
        or _should_register_characterization_validation_tools(context)
        or _should_register_processing_support_tool(context)
        or should_register_sensor_tools(context)
    )


def _materials_platform_enabled() -> bool:
    """Whether the materials-science platform is switched on for this deployment.

    Mirrors ``RuntimeSettings.materials_enabled`` (same env var). The prompt
    builders below do not receive ``settings``, so they use this to suppress the
    materials skill-routing brief and the materials results contract on a
    materials-disabled deployment — where a prompt might still trip the shared
    (dual-use) materials tokens but must not be steered into materials framing.
    """
    return os.getenv("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_CALPHAD_TOOL_GOAL_TOKENS = (
    "alloy phase stability",
    "calphad",
    "equilibrium phase fraction",
    "equilibrium phase fractions",
    "isothermal section",
    "phase diagram",
    "phase equilibria",
    "phase equilibrium",
    "scheil",
    "scheil-gulliver",
    "solidification path",
    "solidification segregation",
    "tdb",
    "thermodynamic database",
    "thermodynamic phase stability",
)


def _should_register_calphad_tools(context: AgentRunContext | None) -> bool:
    """Register the narrow typed primitive only for a CALPHAD-shaped run.

    A generic materials run (for example EBSD or XRD) should not carry the tool
    schema. A selected TDB descriptor is sufficient even when the user's wording
    is terse; otherwise the request must explicitly name CALPHAD/TDB work.
    """

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    if any(token in goal for token in _CALPHAD_TOOL_GOAL_TOKENS):
        return True
    for descriptor in context.resource_descriptors:
        if str(descriptor.get("type") or "").strip() != "selected_resource":
            continue
        name = str(descriptor.get("original_name") or "").strip().lower()
        content_type = str(descriptor.get("content_type") or "").strip().lower()
        metadata = descriptor.get("metadata")
        if (
            # .db is deliberately absent: the binding layer rejects it as
            # unsupported_calphad_resource_format, and since the CALPHAD gate
            # now implies the materials context, a selected SQLite file must
            # not flip a run into materials routing.
            name.endswith((".tdb", ".dat"))
            or content_type == "application/x-thermocalc-tdb"
            or (isinstance(metadata, dict) and isinstance(metadata.get("calphad"), dict))
        ):
            return True
    return False


_KINETICS_TOOL_GOAL_TOKENS = (
    "1-d diffusion",
    "1d diffusion",
    "back diffusion",
    "binary precipitation",
    "diffusion coefficient",
    "diffusion coefficients",
    "diffusion couple",
    "diffusion profile",
    "diffusivity",
    "finite-volume diffusion",
    "interdiffusion",
    "kawin",
    "kampmann-wagner",
    "kampmann wagner",
    "kwn precipitation",
    "mobility coefficient",
    "mobility coefficients",
    "one-dimensional diffusion",
    "precipitation kinetics",
    "single-phase diffusion",
    "tracer diffusivity",
)


def _should_register_kinetics_tools(context: AgentRunContext | None) -> bool:
    """Register Kawin only for an explicitly kinetics-shaped materials run."""

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    return any(token in goal for token in _KINETICS_TOOL_GOAL_TOKENS)


_CRYSTAL_PLASTICITY_TOOL_GOAL_TOKENS = (
    "basal slip",
    "cpfe",
    "crss",
    "crystal plasticity",
    "crystal-plasticity",
    "octahedral slip",
    "prismatic slip",
    "pyramidal c+a",
    "pyramidal slip",
    "resolved shear",
    "schmid factor",
    "slip geometry",
    "slip families",
    "slip family",
    "slip-system",
    "slip system",
)


def _should_register_crystal_plasticity_tools(context: AgentRunContext | None) -> bool:
    """Register only the bounded analytical/CPFE-contract surface."""

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    return any(token in goal for token in _CRYSTAL_PLASTICITY_TOOL_GOAL_TOKENS)


_DEGRADATION_TOOL_GOAL_TOKENS = (
    "corrosion penetration",
    "corrosion current density",
    "creep rate",
    "fatigue crack growth",
    "faraday corrosion",
    "faraday law",
    "lefm",
    "linear elastic fracture mechanics",
    "mode i fracture",
    "mode-i fracture",
    "mass gain law",
    "mass-gain law",
    "norton creep",
    "norton-arrhenius",
    "norton law",
    "oxidation kinetics",
    "oxidation mass gain",
    "parabolic oxidation",
    "paris law",
    "paris equation",
    "paris relation",
    "secondary creep",
    "steady-state creep",
    "stress intensity factor",
    "uniform corrosion",
)

_DEGRADATION_TYPED_TOOL_NAMES = frozenset(
    {
        "materials_convert_uniform_corrosion",
        "materials_evaluate_mode_i_lefm",
        "materials_evaluate_norton_arrhenius_creep",
        "materials_evaluate_oxidation_mass_gain",
        "materials_fit_paris_law",
    }
)


def _requests_allowlisted_typed_tool_execution(goal: str, tool_names: Collection[str]) -> bool:
    """Match a direct imperative for an exact first-party typed tool name.

    ``goal`` has already passed through :func:`request_classification_text`, so
    fenced examples and explicitly negated clauses are absent.  Requiring an
    imperative at the start of a request clause keeps documentation questions
    and incidental identifier mentions from expanding the tool surface.
    """

    normalized = goal.replace("`", "")
    request_prefix = (
        r"(?:^|[.!?;:\n])\s*"
        r"(?:(?:please|then)\s+|(?:can|could|would)\s+you\s+|"
        r"i\s+(?:want|need)\s+you\s+to\s+)?"
        r"(?:call|invoke|run|execute|use)\s+"
        r"(?:the\s+)?(?:(?:first-party|bounded|typed)\s+)*(?:tool\s+)?"
    )
    return any(
        re.search(
            rf"{request_prefix}(?<![a-z0-9_]){re.escape(tool_name)}(?![a-z0-9_])",
            normalized,
        )
        is not None
        for tool_name in tool_names
    )


def _has_explicit_computational_action(goal: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(token)}", goal) for token in _COMPUTATIONAL_ACTION_TOKENS
    ) or bool(re.search(r"\b(?:register|screen|score)\w*\b", goal))


def _should_register_degradation_tools(context: AgentRunContext | None) -> bool:
    """Register bounded degradation reducers only for an explicit calculation.

    A purely explanatory fatigue/fracture/corrosion question should not carry five
    numerical schemas. The specialized method and an action must both be present.
    """

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    # An exact first-party tool request is already a closed computational intent.
    # Underscores are word characters, so phrases such as
    # ``materials_evaluate_oxidation_mass_gain`` do not match the natural-language
    # tokens or action regexes above.  Without this exact-name lane, a user who
    # explicitly requests the typed tool is paradoxically routed to generic tool
    # discovery instead of receiving the bounded degradation surface.
    if _requests_allowlisted_typed_tool_execution(goal, _DEGRADATION_TYPED_TOOL_NAMES):
        return True
    method_named = any(token in goal for token in _DEGRADATION_TOOL_GOAL_TOKENS)
    return method_named and _has_explicit_computational_action(goal)


_CHARACTERIZATION_VALIDATION_TOOL_GOAL_TOKENS = (
    "diffraction profile comparison",
    "diffraction profile metric",
    "diffraction profile residual",
    "held-out registration",
    "held out registration",
    "kabsch",
    "profile goodness of fit",
    "rigid registration",
    "rietveld profile metric",
)


def _should_register_characterization_validation_tools(
    context: AgentRunContext | None,
) -> bool:
    """Register only profile-metric or known-correspondence registration tools."""

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    method_named = (
        any(token in goal for token in _CHARACTERIZATION_VALIDATION_TOOL_GOAL_TOKENS)
        or bool(re.search(r"\b(?:rp|rexp|rwp)\b", goal))
        or bool(
            re.search(r"\b(?:ebsd|4d[- ]?stem|tem)\b", goal)
            and re.search(r"\b(?:apt|atom probe|tem)\b", goal)
            and re.search(r"\b(?:align|correspondence|landmark|register)\w*\b", goal)
        )
    )
    return method_named and _has_explicit_computational_action(goal)


_PROCESSING_SUPPORT_METHOD_TOKENS = (
    "back diffusion",
    "coupled solidification",
    "diffusion mobility",
    "kawin",
    "kampmann-wagner",
    "kwn",
    "mobility diffusion",
    "phase field",
    "phase-field",
    "precipitation kinetics",
    "processing kinetics",
    "scheil",
    "solidification",
)
_PROCESSING_SUPPORT_INTENT_RE = re.compile(
    r"\b(?:availab|boundar|capabilit|implemented|implementation|qualified|readiness|"
    r"ready|support)\w*\b|\bcan\s+(?:you|the\s+platform|ultra)\b|\bdo\s+we\s+have\b"
)


def _should_register_processing_support_tool(context: AgentRunContext | None) -> bool:
    """Register the zero-argument support matrix without shadowing real solvers.

    Ordinary Scheil, diffusion, or KWN execution stays on its dedicated typed
    runtime. Phase-field and coupled moving-interface requests register this
    boundary tool because no in-process numerical solver is qualified.
    """

    if context is None:
        return False
    goal = " ".join(request_classification_text(context.goal).lower().split())
    if "materials_processing_method_support" in goal:
        return True
    support_intent = _PROCESSING_SUPPORT_INTENT_RE.search(goal) is not None
    method_named = any(token in goal for token in _PROCESSING_SUPPORT_METHOD_TOKENS)
    if support_intent and (method_named or "processing method" in goal):
        return True
    external_boundary_method = any(
        token in goal
        for token in (
            "phase field",
            "phase-field",
            "coupled solidification",
            "moving-interface solidification",
        )
    )
    return external_boundary_method and _has_explicit_computational_action(goal)


def looks_scoped_delegation_goal(goal: str) -> bool:
    """True when a goal is computational-study shaped (and not a RareSpot run).

    This is the prompt-only half of scoped subagent registration; selection
    metadata is added by ``_should_register_scoped_delegation_subagents``.
    Pro result-contract classification is deliberately separate because named
    materials methods use a domain contract rather than the dynamics contract.
    """
    goal = str(goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    if looks_materials_computational_goal(goal):
        return True
    return any(
        token in lowered
        for token in (
            "analy",
            "code",
            "debug",
            "experiment",
            "metric",
            "model",
            "plot",
            "reproduc",
            "script",
            "simulation",
            "statistics",
            "train",
            "workflow",
        )
    )


def looks_quantitative_rigor_goal(goal: str) -> bool:
    """True when a goal needs the Pro quantitative/scientific results contract.

    This is intentionally narrower than scoped delegation. Code review, debug,
    and workflow-analysis prompts can benefit from subagents, but should not be
    forced into irrelevant ``±`` / 3×-spread language. Named materials analyses
    use ``MATERIALS_RESULTS_CONTRACT_GUIDANCE`` instead; returning false here also
    prevents the runner's completion validator from re-imposing dynamics-specific
    seeds, observation durations, time steps, and initial-condition rules.
    """
    goal = str(goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    if looks_materials_computational_goal(goal):
        return False
    request_text = request_classification_text(goal)
    lowered = " ".join(request_text.lower().split())
    return any(
        token in lowered
        for token in (
            "benchmark",
            "bifurcation",
            "classif",
            "convergence",
            "estimate",
            "experiment",
            "exponent",
            "fit",
            "lyapunov",
            "metric",
            "monte carlo",
            "numerical",
            "parameter sweep",
            "quantitative",
            "regime",
            "simulation",
            "statistic",
            "spectrum",
        )
    )


def request_classification_text(text: str) -> str:
    """Normalize request text before keyword-based contract classification.

    Guards should classify the user's request, not example/source code or
    explicitly negated work such as "do not create plots".
    """
    stripped = _FENCED_CODE_BLOCK_RE.sub(" ", str(text or ""))
    return _NEGATED_REQUEST_CLAUSE_RE.sub(" ", stripped)


def is_rigor_intelligence(context: AgentRunContext | None) -> bool:
    """True when the run's Intelligence selection maps to the rigor tier.

    UI vocabulary: Intelligence "High" (default) keeps standard behavior;
    Intelligence "Pro" arrives as ``workflow_hint.id == "pro_mode"`` on the
    run envelope and activates the enforced results contract.
    """
    if context is None:
        return False
    hint = context.workflow_hint if isinstance(context.workflow_hint, dict) else {}
    return str(hint.get("id") or "").strip().lower() == "pro_mode"


def _should_register_scoped_delegation_subagents(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    return is_materials_context(context) or looks_scoped_delegation_goal(context.goal)


def _should_register_async_delegation_subagents(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    if is_cleanroom_evaluation_profile(context.evaluation_profile):
        return False
    goal = str(context.goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    return any(
        token in lowered
        for token in (
            "analy",
            "async",
            "background",
            "code",
            "debug",
            "delegate",
            "experiment",
            "long-running",
            "metric",
            "model",
            "plot",
            "reproduc",
            "script",
            "simulation",
            "statistics",
            "train",
            "workflow",
        )
    )


def build_system_prompt(settings: RuntimeSettings, context: AgentRunContext | None = None) -> str:
    cleanroom = bool(
        context is not None and is_cleanroom_evaluation_profile(context.evaluation_profile)
    )
    system_prompt = SYSTEM_PROMPT
    if cleanroom:
        system_prompt = system_prompt.replace(
            _DURABLE_MEMORY_SYSTEM_GUIDANCE,
            _CLEANROOM_MEMORY_SYSTEM_GUIDANCE,
            1,
        )
        system_prompt = system_prompt.replace(
            _DURABLE_CATALOG_SYSTEM_GUIDANCE,
            _CLEANROOM_TOOL_SYSTEM_GUIDANCE,
            1,
        )
    sections = [
        system_prompt.strip(),
        WRITING_GUIDANCE.strip(),
        MATH_FORMATTING_GUIDANCE.strip(),
        PLOT_WORKFLOW_GUIDANCE.strip(),
        SANDBOX_RUNTIME_GUIDANCE.strip(),
        build_sandbox_resources_guidance(settings),
    ]
    if not settings.model_supports_multimodal:
        sections.append(TEXT_ONLY_ARTIFACT_GUIDANCE.strip())
    if not cleanroom:
        sections.append(PRIOR_ARTIFACT_GUIDANCE.strip())
        sections.append(UPLOADED_FILE_GUIDANCE.strip())
    # Only advertise domain workflows when they apply to this run; otherwise the
    # model carries hundreds of tokens of instructions for tools it does not
    # have. Paper guidance keys on _should_register_paper_tools; BisQue guidance
    # keys on _should_register_bisque_tools (with BISQUE_UNLINKED_HINT otherwise).
    if context is None or _should_register_paper_tools(context):
        sections.append(PAPER_REVIEW_GUIDANCE.strip())
    # Only advertise the BisQue tools when they are actually registered for this
    # run; otherwise the model is told to call tools it does not have (the
    # reported "bisque_module_runs is not among the registered tools" failure).
    if not cleanroom:
        if context is None or _should_register_bisque_tools(context):
            sections.append(BISQUE_GUIDANCE.strip())
        else:
            sections.append(BISQUE_UNLINKED_HINT.strip())
    if context is None or _should_register_episodic_tools(context):
        sections.append(EPISODIC_GUIDANCE.strip())
    # Advertise the vision-reasoner only when it is actually registered for the run,
    # so generic text runs do not pay for delegation guidance to an absent subagent.
    if context is not None and _should_register_vision_subagent(context, settings):
        sections.append(VISION_DELEGATION_GUIDANCE)
    if context is not None and _should_register_qwen_code_runner(context, settings):
        sections.append(QWEN_CODE_DELEGATION_GUIDANCE)
    # Advertise the Builder's delegation discipline only when it is enabled (and thus
    # registered): hand heavy/iterative coding to the Builder EARLY with the goal + data
    # paths instead of debugging inline in the coordinator's own context. Two live traces
    # motivated this: a 527K-token over-prep run, then a 6.7M-token livelock in which the
    # coordinator drove 155 inline code-runner calls instead of delegating the loop.
    if settings.builder_enabled:
        sections.append(BUILDER_DELEGATION_GUIDANCE)
    if context is not None:
        brief = build_run_context_brief(context)
        if brief:
            sections.append(brief)
    return "\n\n".join(sections) + "\n"


def build_run_context_brief(context: AgentRunContext, *, max_artifacts: int = 8) -> str:
    lines = [
        "Active run context:",
        f"- run_id: {context.run_id}",
        f"- thread_id: {context.thread_id}",
    ]
    if context.goal.strip():
        lines.append(f"- goal: {context.goal.strip()}")
    if _materials_platform_enabled() and is_materials_context(context):
        lines.extend(
            [
                "- suggested_domain: materials (selection-context hint or prompt-classifier "
                "routing evidence; confirm it against the selected data)",
                "- materials skill routing: read /skills/computational-materials/SKILL.md "
                "for EBSD/microstructure, /skills/materials-structure-thermo/SKILL.md for "
                "CIF/symmetry/point defects/CALPHAD, "
                "/skills/materials-processing-kinetics/SKILL.md for solidification/kinetics, "
                "/skills/materials-characterization/SKILL.md for "
                "XRD/spectroscopy, and /skills/materials-characterization-advanced/SKILL.md "
                "for measured profiles/registration/held-out validation, and "
                "/skills/materials-sensor-data/SKILL.md for calibrated waveforms/telemetry "
                "and links to OME-NGFF imagery before analysis",
            ]
        )
        if not is_cleanroom_evaluation_profile(
            context.evaluation_profile
        ) and should_register_sensor_tools(context):
            lines.append(
                "- selected sensor workflow: call inspect_selected_sensor_series before "
                "execute; validate_values=false is metadata-only, request at most one "
                "bounded channel envelope, and treat lineage as unbound unless the tool "
                "explicitly reports tree_verified. Never infer clock synchronization, "
                "calibration, or units."
            )
        if not is_cleanroom_evaluation_profile(
            context.evaluation_profile
        ) and _should_register_crystal_plasticity_tools(context):
            lines.append(
                "- selected crystal-plasticity skill: read "
                "/skills/materials-crystal-plasticity/SKILL.md, then call "
                "materials_analyze_crystal_slip for "
                "canonical geometry/resolved shear/Schmid factors and "
                "materials_validate_cpfe_contract for schema-v1 input readiness. Call these "
                "typed tools directly before considering execute or code-runner delegation. "
                "Treat geometry as non-constitutive and a valid CPFE contract as "
                "execution-unsupported until the tool reports a qualified solver binding. "
                "Copy analysis_artifact.canonical_json and "
                "materials_validation_artifact.canonical_json directly to requested outputs; "
                "do not discover or reconstruct the validation API. A deterministic typed input "
                "rejection is terminal: report the single error without repeated calls or "
                "substitute inputs. Do not create unrequested output files."
            )
        if not is_cleanroom_evaluation_profile(
            context.evaluation_profile
        ) and _should_register_degradation_tools(context):
            lines.append(
                "- selected degradation skill: read "
                "/skills/materials-mechanics-degradation/SKILL.md, then use the matching "
                "bounded typed tool directly before "
                "execute or code-runner: materials_evaluate_mode_i_lefm, "
                "materials_fit_paris_law, materials_evaluate_norton_arrhenius_creep, "
                "materials_evaluate_oxidation_mass_gain, or "
                "materials_convert_uniform_corrosion. Respect its calibration domain and "
                "declared provenance. Never invent placeholder hashes, demo citations, material "
                "states, environments, or validity intervals to satisfy a required schema; when "
                "the caller has not explicitly supplied required evidence, identify the missing "
                "fields instead of calling the tool. A deterministic typed input rejection is "
                "terminal and must not trigger substitute inputs or repeated calls. These reducers "
                "do not predict fracture/fatigue/creep/"
                "oxidation/corrosion life or establish ASTM compliance. Copy returned canonical "
                "analysis and validation artifacts exactly rather than reconstructing them, and "
                "do not create unrequested output files."
            )
        if not is_cleanroom_evaluation_profile(
            context.evaluation_profile
        ) and _should_register_characterization_validation_tools(context):
            lines.append(
                "- advanced-characterization validation: call "
                "materials_calculate_diffraction_profile_metrics for supplied observed/calculated "
                "profiles or materials_fit_held_out_rigid_registration for supplied known "
                "correspondences. Use fixed disjoint calibration/held-out partitions and copy "
                "the canonical artifacts exactly. These tools do not refine/index diffraction, "
                "discover feature correspondences, segment data, or validate physical identity."
            )
        if not is_cleanroom_evaluation_profile(
            context.evaluation_profile
        ) and _should_register_processing_support_tool(context):
            lines.append(
                "- processing-method boundary: call materials_processing_method_support directly "
                "before execute or code-runner. It is zero-argument static support discovery, "
                "not evidence that run inputs or a solver are present. Never replace unsupported "
                "phase-field or coupled moving-interface execution with a toy solver."
            )
    if context.runtime_facts:
        lines.append("Runtime facts:")
        for key in (
            "current_datetime_utc",
            "current_date_utc",
            "user_timezone",
            "local_datetime",
            "run_started_at",
            "product_name",
            "app_name",
            "app_version",
            "deployment_environment",
            "public_url",
        ):
            value = str(context.runtime_facts.get(key) or "").strip()
            if value:
                lines.append(f"- {key}: {value}")
        lines.append(
            "- Use these runtime facts for today, tomorrow, yesterday, current time, "
            "timezone, product/deployment identity, and public URL. Do not infer "
            "current dates from model knowledge."
        )
    runtime_budget_lines = _run_context_runtime_budget_lines(context)
    if runtime_budget_lines:
        lines.append("Runtime budgets:")
        lines.extend(runtime_budget_lines)
    if context.selected_file_ids:
        file_ids = ", ".join(context.selected_file_ids)
        lines.append(
            f"- selected uploaded file ids: {file_ids} | use stage_uploaded_files_for_analysis"
        )
    if _has_bisque_account_binding(context):
        mutation_tools: list[str] = []
        if "bisque.upload" in context.remote_mutation_intents:
            mutation_tools.extend(["bisque_upload_files", "bisque_upload_workspace_files"])
        if "bisque.create_dataset" in context.remote_mutation_intents:
            mutation_tools.append("bisque_create_dataset")
        mutation_note = (
            f"; authorized remote mutation tools for this run: {', '.join(mutation_tools)}"
            if mutation_tools
            else "; no remote mutation capability is authorized for this run"
        )
        lines.append(
            "- linked BisQue account available: use bisque_search_resources, "
            "bisque_download_resource, and bisque_module_runs through the control plane"
            f"{mutation_note}; "
            "use scope='owner' for the user's own resources, sort='recent' for newest-first "
            "queries, extensions=['png'] or extensions=['nii','nii.gz','nifti'] for file-type "
            "searches, resource_type='dataset' for dataset questions, and count_all=True for totals"
        )
    if context.selected_resource_uris:
        resource_uris = ", ".join(context.selected_resource_uris[:8])
        lines.append(
            f"- selected BisQue resource URIs: {resource_uris} | use bisque_download_resource"
        )
    if context.selected_dataset_uris:
        dataset_uris = ", ".join(context.selected_dataset_uris[:8])
        lines.append(
            f"- selected BisQue dataset URIs: {dataset_uris} | use BisQue tools before analysis"
        )
    ingested_papers = context.knowledge_context.get("ingested_papers")
    if isinstance(ingested_papers, list) and ingested_papers:
        lines.append(
            "- ingested papers available through paper_manifest/search_paper/read_paper_pages:"
        )
        for paper in ingested_papers[:max_artifacts]:
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("paper_id") or "").strip()
            source = str(paper.get("source") or "").strip()
            page_count = paper.get("page_count")
            status = str(paper.get("extraction_status") or "unknown").strip()
            lines.append(
                f"  - {paper_id or '(no paper_id)'} | pages={page_count} | "
                f"status={status} | source={source or '(cached)'}"
            )
        if len(ingested_papers) > max_artifacts:
            lines.append(
                f"  - ... {len(ingested_papers) - max_artifacts} more; call paper_manifest"
            )
    artifacts = [
        descriptor
        for descriptor in context.resource_descriptors
        if str(descriptor.get("type") or "artifact") == "artifact"
    ]
    if artifacts:
        lines.append("- prior durable artifacts available:")
        for descriptor in artifacts[:max_artifacts]:
            artifact_id = str(descriptor.get("artifact_id") or "").strip()
            path = str(descriptor.get("path") or descriptor.get("relative_path") or "").strip()
            title = str(descriptor.get("title") or path or artifact_id).strip()
            kind = str(descriptor.get("kind") or "artifact").strip()
            lines.append(
                f"  - {artifact_id or '(no artifact_id)'} | {kind} | {title} | "
                f"{path or '(no path)'} | use stage_artifact_for_analysis"
            )
        if len(artifacts) > max_artifacts:
            lines.append(f"  - ... {len(artifacts) - max_artifacts} more; call artifact_manifest")
    return "\n".join(lines)


def _run_context_runtime_budget_lines(context: AgentRunContext) -> list[str]:
    lines: list[str] = []
    for prefix, data in (("budget", context.budget), ("benchmark", context.benchmark)):
        if not isinstance(data, dict):
            continue
        value = data.get("max_runtime_seconds")
        if isinstance(value, int | float) and value > 0:
            lines.append(f"- {prefix}.max_runtime_seconds: {value:g}")
    return lines


def build_runtime_prompt_suffix(
    context: AgentRunContext | None,
    *,
    elapsed_seconds: float | None = None,
) -> str:
    """Request-dependent system-prompt suffix: run brief, wall-clock, contract.

    Pure helper so the middleware stays trivially testable. The wall-clock line
    exists because the model cannot see session time and otherwise mislabels
    inner sandbox compute as the study's wall-clock time.
    """
    if context is None:
        return ""
    sections: list[str] = []
    brief = build_run_context_brief(context)
    if brief:
        sections.append(brief)
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        sections.append(
            f"Elapsed wall-clock for this run so far: {minutes}m{seconds:02d}s. "
            "When reporting runtimes, report this wall-clock time and any inner "
            "compute time as separate labeled numbers."
        )
    if is_rigor_intelligence(context):
        if (
            _materials_platform_enabled()
            and is_materials_context(context)
            and _should_register_scoped_delegation_subagents(context)
        ):
            sections.append(MATERIALS_RESULTS_CONTRACT_GUIDANCE.strip())
        elif looks_quantitative_rigor_goal(context.goal):
            sections.append(RESULTS_CONTRACT_GUIDANCE.strip())
    return "\n\n".join(sections)


class UltraRunContextPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append the per-run context brief to the assembled system message.

    This must append rather than re-render the whole prompt: user middleware
    runs inside the deepagents base stack, so a wholesale replacement (the
    previous ``@dynamic_prompt`` approach) silently erased sections appended
    by earlier middleware — the skills listing and the task-tool guidance.
    The static portion of the prompt is supplied once at ``create_deep_agent``
    time; only the run-context brief is request-dependent.

    Also appends elapsed wall-clock (the agent is built per run, so instance
    creation approximates run start; the value is set once in ``__init__`` and
    only read afterwards, keeping the instance concurrency-safe) and, for
    Intelligence Pro runs on computational-study goals, the results contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self._started_monotonic = time.monotonic()

    def _brief(self, request: ModelRequest) -> str:
        runtime_context = getattr(request.runtime, "context", None)
        context = runtime_context if isinstance(runtime_context, AgentRunContext) else None
        if context is None:
            return ""
        return build_runtime_prompt_suffix(
            context,
            elapsed_seconds=time.monotonic() - self._started_monotonic,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        brief = self._brief(request)
        if not brief:
            return handler(request)
        return handler(
            request.override(system_message=append_to_system_message(request.system_message, brief))
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        brief = self._brief(request)
        if not brief:
            return await handler(request)
        return await handler(
            request.override(system_message=append_to_system_message(request.system_message, brief))
        )


def build_runtime_prompt_middleware() -> Any:
    return UltraRunContextPromptMiddleware()


class UltraAttemptLedgerMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append the durable attempt-ledger digest to the system message per model call.

    The runner appends a JSONL entry whenever a sandbox execute FAILS or repeats
    with unchanged output (progress_guard). This middleware re-derives a compact
    digest from that file on every model call, so the memory of what already
    failed survives SummarizationMiddleware compaction BY CONSTRUCTION: compaction
    rewrites ``messages`` but never the per-request system prompt. Injecting into
    ``messages`` instead would be self-defeating — the next compaction would erase
    it, restarting the re-discovery → re-run loop this exists to break.

    Advisory only, never a gate: a command that failed earlier may legitimately
    succeed after the workspace changes, so nothing is blocked — the model is
    reminded, not restricted. Healthy runs write no entries and pay nothing.
    Shared across the coordinator and the coding subagents (one instance): the
    ledger is per-run state, and the mtime-cached read is concurrency-safe.
    """

    def __init__(self, ledger_path: Path) -> None:
        super().__init__()
        self._ledger_path = ledger_path
        self._cached_digest = ""
        self._cached_mtime_ns = -1

    def _digest(self) -> str:
        try:
            mtime_ns = self._ledger_path.stat().st_mtime_ns
        except OSError:
            return ""
        if mtime_ns != self._cached_mtime_ns:
            self._cached_digest = read_attempt_ledger_digest(self._ledger_path)
            self._cached_mtime_ns = mtime_ns
        return self._cached_digest

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        digest = self._digest()
        if not digest:
            return handler(request)
        return handler(
            request.override(
                system_message=append_to_system_message(request.system_message, digest)
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        digest = self._digest()
        if not digest:
            return await handler(request)
        return await handler(
            request.override(
                system_message=append_to_system_message(request.system_message, digest)
            )
        )


def attempt_ledger_path(workspace_dir: Path) -> Path:
    """Single source of truth for the per-run ledger location (runner writes it,
    the middleware reads it). Lives under a dot-directory so the non-recursive
    workspace artifact sweep can never promote it to a run deliverable."""
    return workspace_dir / ".ultra" / "attempt_ledger.jsonl"


def sandbox_compute_resources(settings: RuntimeSettings) -> dict[str, Any]:
    """Structured compute envelope of the code sandbox, for tool_capability_manifest.

    Lets the coordinator size work to the host instead of guessing: how many cores it
    may parallelize across, how much memory it has, that the container is offline with a
    fixed package set and ephemeral per call, and where GPU compute actually lives.
    Sourced from the same RuntimeSettings the live sandbox is built from, so the manifest
    can never drift from the actual ``docker run`` flags.
    """
    offline = settings.sandbox_network == "none"
    gpus = settings.sandbox_gpus.strip()
    return {
        "execution_model": (
            "Each execute() runs a FRESH, ephemeral container; only files under "
            "/workspace and /outputs persist between calls. In-memory state and "
            "background processes (cmd &, nohup) do NOT survive a call — there is no "
            "long-lived daemon to poll. Checkpoint long work to durable /outputs files "
            "after each condition or batch; keep scratch/temp files under /workspace."
        ),
        "cpus": (
            "all available host cores" if settings.sandbox_cpus <= 0 else settings.sandbox_cpus
        ),
        "memory_limit": (settings.sandbox_memory.strip() or "host-limited (no per-container cap)"),
        "pids_limit": settings.sandbox_pids_limit or "unbounded",
        "shm_size": (
            settings.sandbox_shm_size.strip()
            or "Docker default (~64MB) — keep /dev/shm and DataLoader-worker use small"
        ),
        "gpu": (
            f"available in-sandbox ({gpus}); torch.cuda is usable"
            if gpus
            else (
                "NONE in-sandbox (torch runs on CPU). For GPU inference use the "
                "MegaSeg/RareSpot tools/services, not the code sandbox."
            )
        ),
        "network": (
            "offline (no internet)"
            if offline
            else f"ENABLED ({settings.sandbox_network}) — outbound internet available"
        ),
        "package_installs": (
            "DISABLED — the sandbox is offline; pip/conda install will fail. Use only "
            "the preinstalled stack in preinstalled_packages."
            if offline
            else (
                "ENABLED — network is on. Install with `pip install --user <pkg>` because the "
                "rootfs is read-only (--user writes to /workspace and stays importable). Prefer "
                "preinstalled_packages when they already cover the need."
            )
        ),
        "wall_clock_cap_seconds": settings.sandbox_timeout_seconds or "none",
        "filesystem": (
            "rootfs is read-only; write everything under /workspace (durable deliverables "
            "to /outputs). HOME=/workspace and TMPDIR=/workspace/.tmp; /tmp is a small "
            "RAM-backed tmpfs, so write large temp/intermediate files under /workspace."
        ),
        "preinstalled_packages": list(SANDBOX_KEY_PACKAGES),
    }


def build_sandbox_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
    outputs_dir: str | Path | None = None,
    run_id: str = "",
) -> DockerSandboxBackend:
    immutable_image_id = resolve_docker_image_id(settings.sandbox_image)
    return DockerSandboxBackend(
        workspace_dir=workspace_dir,
        outputs_dir=outputs_dir,
        config=DockerSandboxConfig(
            # When the image is already local, pin execution to the resolved
            # immutable configuration ID. This is also the ID emitted in run
            # traces, so scientific benchmark provenance describes the image
            # Docker actually launches rather than a mutable tag.
            image=immutable_image_id or settings.sandbox_image,
            network=settings.sandbox_network,
            cpus=settings.sandbox_cpus,
            memory=settings.sandbox_memory,
            pids_limit=settings.sandbox_pids_limit,
            shm_size=settings.sandbox_shm_size,
            gpus=settings.sandbox_gpus,
            timeout_seconds=settings.sandbox_timeout_seconds,
            output_limit_bytes=settings.sandbox_output_limit_bytes,
            worker_id=settings.worker_id,
            run_id=run_id,
        ),
    )


def _memory_user_slug(user_id: str | None) -> str:
    """Filesystem-safe slug for a user id, so per-user memory directories never
    collide or escape the memory root."""
    raw = str(user_id or "").strip()
    if not raw:
        return ""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.")
    return slug[:128]


def resolve_user_memory_root(
    settings: RuntimeSettings,
    user_id: str | None,
    *,
    thread_id: str | None = None,
) -> Path:
    """Per-user durable memory directory under the shared memory root.

    Scoping ``/memories/`` per user keeps one researcher's preferences and
    research context from leaking into another's runs. Falls back to the shared
    memory when no user is attached. The shared ``papers/`` cache lives directly
    under the memory root, so user and anonymous directories are nested to avoid
    collisions.
    """
    base = Path(settings.memory_root)
    slug = _memory_user_slug(user_id)
    if slug:
        return base / "users" / slug
    thread_slug = _memory_user_slug(thread_id)
    if thread_slug:
        return base / "anonymous_threads" / thread_slug
    return base / "anonymous" / "unscoped"


SKILLS_SOURCES = ["/skills/"]
"""Backend-relative skill sources for SkillsMiddleware (progressive disclosure)."""


# The materials skills live alongside the general skills under the skills root.
# When the materials platform is disabled we hide them from the skill listing so
# every non-materials run does not pay the ~1k prompt tokens their descriptions
# would otherwise add on every model call.
_MATERIALS_SKILL_DIR_NAMES = frozenset(
    {
        "computational-materials",
        "materials-characterization",
        "materials-characterization-advanced",
        "materials-crystal-plasticity",
        "materials-mechanics-degradation",
        "materials-processing-kinetics",
        "materials-sensor-data",
        "materials-structure-thermo",
    }
)


def _skill_directory_name(entry: dict[str, Any]) -> str:
    path = str(entry.get("path", "")).rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


class _MaterialsFilteredSkillsBackend(FilesystemBackend):
    """FilesystemBackend that hides the materials skills from directory listings.

    SkillsMiddleware discovers skills by listing the skills root and reading each
    subdirectory's ``SKILL.md``. Filtering ``ls`` therefore keeps the materials
    skills out of the always-on skill index (and its prompt-token cost) while the
    materials platform is disabled, without moving any files. Reads are untouched.
    """

    def ls(self, path: str) -> Any:
        result = super().ls(path)
        entries = getattr(result, "entries", None)
        if not entries:
            return result
        filtered = [
            entry
            for entry in entries
            if _skill_directory_name(entry) not in _MATERIALS_SKILL_DIR_NAMES
        ]
        if len(filtered) == len(entries):
            return result
        return replace(result, entries=filtered)


def resolve_skills_root(settings: RuntimeSettings) -> Path | None:
    """Directory holding repo-shipped agent skills, or None when unavailable.

    An explicit ``settings.skills_root`` wins; otherwise fall back to the
    ``skills/`` directory shipped next to this package so the worker finds the
    protocols regardless of its working directory. Only a directory containing
    at least one ``*/SKILL.md`` counts, so deployments without skills register
    no middleware instead of advertising an empty skill list.
    """
    if settings.skills_root.strip():
        root = Path(settings.skills_root).expanduser()
        if root.is_dir() and any(root.glob("*/SKILL.md")):
            return root
        return None
    candidates = [
        Path(__file__).resolve().parent / "skills",  # shipped inside the wheel (site-packages)
        Path(__file__).resolve().parents[2] / "skills",  # editable / source checkout (src/ layout)
        Path(
            "/app/deepagents_runtime/skills"
        ),  # container build-context copy (belt-and-suspenders)
    ]
    for root in candidates:
        if root.is_dir() and any(root.glob("*/SKILL.md")):
            return root
    return None


def resolve_org_policies_root(settings: RuntimeSettings, org_id: str | None) -> Path:
    """Per-org directory for read-only policy memory.

    Org-scoped so a lab's shared policy is one source of truth across its members,
    and isolated from other orgs. Defaults under the shared memory root so it rides
    the same barrel in production. Writes here are blocked by ``MEMORY_PERMISSIONS``;
    only application code (the seed script / Store) populates it.
    """
    base = (
        Path(settings.policies_root)
        if settings.policies_root.strip()
        else Path(settings.memory_root) / "policies"
    )
    slug = _memory_user_slug(org_id)
    return base / slug if slug else base / "shared"


def build_agent_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
    artifact_dir: str | Path | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    org_id: str | None = None,
    run_id: str | None = None,
    evaluation_profile: str = "",
) -> CompositeBackend:
    """Route sandbox execution separately from durable agent files."""
    cleanroom = is_cleanroom_evaluation_profile(evaluation_profile)
    if cleanroom:
        memory_root = evaluation_memory_dir(
            settings.memory_root,
            evaluation_profile,
            run_id or "",
        )
    else:
        memory_root = resolve_user_memory_root(settings, user_id, thread_id=thread_id)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else Path(settings.artifact_root)
    memory_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # /memories/ and /outputs/ can hold large memory notes and offloaded tool
    # results; raise the grep size cap (default 10 MB only bounds the Python grep
    # fallback skip, not read/write) so search recall covers big files when
    # ripgrep is unavailable.
    routes: dict[str, Any] = {
        "/memories/": FilesystemBackend(memory_root, virtual_mode=True, max_file_size_mb=128),
        "/outputs/": FilesystemBackend(artifact_root, virtual_mode=True, max_file_size_mb=128),
    }
    skills_root = resolve_skills_root(settings)
    if skills_root is not None:
        routes["/skills/"] = (
            FilesystemBackend(skills_root, virtual_mode=True)
            if _materials_platform_enabled()
            else _MaterialsFilteredSkillsBackend(skills_root, virtual_mode=True)
        )
    policies_root = (
        evaluation_policy_dir(settings.memory_root, evaluation_profile, run_id or "")
        if cleanroom
        else resolve_org_policies_root(settings, org_id)
    )
    policies_root.mkdir(parents=True, exist_ok=True)
    routes["/policies/"] = FilesystemBackend(policies_root, virtual_mode=True)

    return CompositeBackend(
        default=build_sandbox_backend(
            settings,
            workspace_dir=workspace_dir,
            outputs_dir=artifact_root,
            run_id=run_id or "",
        ),
        routes=routes,
        artifacts_root="/outputs/.deepagents",
    )


def _docker_sandbox_from_backend(backend: Any) -> DockerSandboxBackend | None:
    """Return the exact immutable sandbox used by the agent's execute surface."""

    if isinstance(backend, DockerSandboxBackend):
        return backend
    if isinstance(backend, CompositeBackend) and isinstance(backend.default, DockerSandboxBackend):
        return backend.default
    return None


_IMMUTABLE_SANDBOX_IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TYPED_CALPHAD_OUTER_TIMEOUT_SECONDS = 60
_TYPED_CALPHAD_MAX_CPUS = 8.0
_TYPED_CALPHAD_MAX_MEMORY_BYTES = 32 * 1024**3
_TYPED_CALPHAD_MAX_PIDS = 4096
_TYPED_KINETICS_OUTER_TIMEOUT_SECONDS = 30
_TYPED_KINETICS_MAX_CPUS = 2.0
_TYPED_KINETICS_MEMORY = "8g"
_TYPED_KINETICS_MAX_PIDS = 256
_TYPED_KINETICS_OUTPUT_LIMIT_BYTES = 8 * 1024**2 + 128 * 1024


def _docker_memory_bytes(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kmgt]?)(?:i?b)?", value.casefold())
    if match is None:
        return None
    amount = float(match.group(1))
    multipliers = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    size = amount * multipliers[match.group(2)]
    if not math.isfinite(size) or size <= 0 or size > _TYPED_CALPHAD_MAX_MEMORY_BYTES:
        return None
    return int(size)


def _bounded_calphad_sandbox_backend(
    backend: DockerSandboxBackend | None,
) -> DockerSandboxBackend | None:
    """Clone the run sandbox with a hard outer cap for the typed primitive.

    The ordinary execute surface may intentionally allow long or networked
    research jobs. The typed CALPHAD path is stricter: immutable image, offline
    networking, no-new-privileges, and positive CPU/memory/PID bounds are all
    required in addition to its operator-nonextensible 60-second ceiling.
    """

    if backend is None or backend.outputs_dir is None:
        return None
    image = str(backend.config.image or "").strip().lower()
    if not _IMMUTABLE_SANDBOX_IMAGE_RE.fullmatch(image):
        return None
    cpus = float(backend.config.cpus)
    memory = str(backend.config.memory or "").strip()
    pids_limit = int(backend.config.pids_limit)
    if (
        not math.isfinite(cpus)
        or cpus <= 0
        or _docker_memory_bytes(memory) is None
        or pids_limit <= 0
    ):
        return None
    operator_cap = int(backend.config.timeout_seconds or 0)
    timeout_seconds = min(operator_cap, _TYPED_CALPHAD_OUTER_TIMEOUT_SECONDS)
    if timeout_seconds <= 0:
        timeout_seconds = _TYPED_CALPHAD_OUTER_TIMEOUT_SECONDS
    return DockerSandboxBackend(
        workspace_dir=backend.workspace_dir,
        outputs_dir=backend.outputs_dir,
        config=replace(
            backend.config,
            image=image,
            network="none",
            cpus=min(cpus, _TYPED_CALPHAD_MAX_CPUS),
            pids_limit=min(pids_limit, _TYPED_CALPHAD_MAX_PIDS),
            no_new_privileges=True,
            gpus="",
            timeout_seconds=timeout_seconds,
        ),
        progress_callback=getattr(backend, "_progress_callback", None),
    )


def _bounded_kinetics_sandbox_backend(
    backend: DockerSandboxBackend | None,
) -> DockerSandboxBackend | None:
    """Build the separately authorized, networkless Kawin NumPy-2 backend.

    The operator supplies both a local image reference and the immutable image
    configuration ID that reference is authorized to resolve to.  Resolving
    and comparing them here keeps Compose and direct worker launches fail
    closed even if a launcher-side preflight was accidentally skipped.
    """

    if backend is None or backend.outputs_dir is None:
        return None
    image_reference = os.getenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE", "").strip()
    authorized_image = os.getenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID", "").strip().lower()
    if not image_reference or not _IMMUTABLE_SANDBOX_IMAGE_RE.fullmatch(authorized_image):
        return None
    resolved_image = resolve_docker_image_id(image_reference).strip().lower()
    if resolved_image != authorized_image:
        return None
    return DockerSandboxBackend(
        workspace_dir=backend.workspace_dir,
        outputs_dir=backend.outputs_dir,
        config=DockerSandboxConfig(
            image=authorized_image,
            network="none",
            cpus=_TYPED_KINETICS_MAX_CPUS,
            memory=_TYPED_KINETICS_MEMORY,
            pids_limit=_TYPED_KINETICS_MAX_PIDS,
            no_new_privileges=True,
            gpus="",
            timeout_seconds=_TYPED_KINETICS_OUTER_TIMEOUT_SECONDS,
            output_limit_bytes=_TYPED_KINETICS_OUTPUT_LIMIT_BYTES,
            worker_id=backend.config.worker_id,
            run_id=backend.config.run_id,
        ),
        progress_callback=getattr(backend, "_progress_callback", None),
    )


def build_research_agent(
    settings: RuntimeSettings,
    *,
    model: BaseChatModel | None = None,
    backend: Any | None = None,
    workspace_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    tools: Sequence[BaseTool | Any] | None = None,
    context: AgentRunContext | None = None,
    checkpointer: Any | None = None,
    surface_attestation_sink: Callable[[dict[str, str]], None] | None = None,
) -> Any:
    ensure_ultra_harness_profile()
    cleanroom = bool(
        context is not None and is_cleanroom_evaluation_profile(context.evaluation_profile)
    )
    resolved_backend = backend
    skills_sources: list[str] | None = None
    if resolved_backend is None and workspace_dir is not None:
        resolved_backend = build_agent_backend(
            settings,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            user_id=context.user_id if context is not None else None,
            thread_id=context.thread_id if context is not None else None,
            org_id=context.org_id if context is not None else None,
            run_id=context.run_id if context is not None else None,
            evaluation_profile=context.evaluation_profile if context is not None else "",
        )
        # Skills ride the /skills/ route of the backend we just built; a
        # caller-supplied backend has no such route, so sources stay unset.
        if resolve_skills_root(settings) is not None:
            skills_sources = list(SKILLS_SOURCES)
    if resolved_backend is None:
        resolved_backend = StateBackend()

    middleware: list[Any] = []
    # OUTERMOST tool-call wrapper: contain a failing/slow `task` subagent to a degraded
    # ToolMessage so one bad subagent in a parallel fan-out cannot cancel its siblings and abort
    # the whole run (langgraph's gather has no return_exceptions; the default handler re-raises).
    middleware.append(
        SubagentFailureIsolationMiddleware(timeout_seconds=settings.subagent_task_timeout_seconds)
    )
    middleware.append(build_runtime_prompt_middleware())
    if not settings.model_supports_multimodal:
        middleware.append(TextOnlyMultimodalMiddleware())

    resolved_tools = list(tools or [])
    context_tools = (
        [] if cleanroom else build_context_tools(upload_roots=settings.rarespot_upload_roots)
    )
    calphad_tools: list[Any] = []
    kinetics_tools: list[Any] = []
    crystal_plasticity_tools: list[Any] = []
    degradation_characterization_tools: list[Any] = []
    sensor_tools: list[Any] = []
    calphad_backend = _bounded_calphad_sandbox_backend(
        _docker_sandbox_from_backend(resolved_backend)
    )
    if (
        _should_register_calphad_tools(context)
        and calphad_backend is not None
        and calphad_backend.outputs_dir is not None
    ):
        calphad_tools = build_calphad_tools(
            settings,
            backend=calphad_backend,
            upload_roots=settings.rarespot_upload_roots,
        )
    kinetics_backend = _bounded_kinetics_sandbox_backend(
        _docker_sandbox_from_backend(resolved_backend)
    )
    if (
        not cleanroom
        and _should_register_kinetics_tools(context)
        and kinetics_backend is not None
        and kinetics_backend.outputs_dir is not None
    ):
        kinetics_tools = build_kinetics_tools(
            settings,
            backend=kinetics_backend,
            upload_roots=settings.rarespot_upload_roots,
        )
    if not cleanroom and _should_register_crystal_plasticity_tools(context):
        crystal_plasticity_tools = build_crystal_plasticity_tools()
    register_degradation_tools = _should_register_degradation_tools(context)
    register_characterization_tools = _should_register_characterization_validation_tools(context)
    register_processing_support = _should_register_processing_support_tool(context)
    if not cleanroom and (
        register_degradation_tools or register_characterization_tools or register_processing_support
    ):
        degradation_characterization_tools = build_degradation_characterization_tools(
            include_degradation=register_degradation_tools,
            include_characterization=register_characterization_tools,
            include_processing_support=register_processing_support,
        )
    if not cleanroom and should_register_sensor_tools(context):
        sensor_tools = build_sensor_tools(tuple(settings.rarespot_upload_roots))
    paper_tools = (
        build_paper_tools(
            upload_roots=settings.rarespot_upload_roots,
            cache_root=Path(settings.memory_root) / "papers",
        )
        if _should_register_paper_tools(context)
        else []
    )
    resolved_tools.extend(context_tools)
    # Materials tool families are registered only on a materials-enabled
    # deployment. The per-context _should_register_* gates already keep them off
    # non-materials runs; this flag keeps them off entirely until the platform is
    # switched on (and its runtime images/roles are provisioned).
    if _materials_platform_enabled():
        resolved_tools.extend(calphad_tools)
        resolved_tools.extend(kinetics_tools)
        resolved_tools.extend(crystal_plasticity_tools)
        resolved_tools.extend(degradation_characterization_tools)
        resolved_tools.extend(sensor_tools)
    resolved_tools.extend(paper_tools)
    if _should_register_bisque_tools(context):
        resolved_tools.extend(build_bisque_tools(settings, context=context))
    if _should_register_episodic_tools(context):
        resolved_tools.extend(build_episodic_tools(settings))
    if _should_register_resource_tools(context):
        resolved_tools.extend(
            build_resource_tools(settings, upload_roots=settings.rarespot_upload_roots)
        )
    if _should_register_git_tools(context, settings):
        resolved_tools.extend(build_git_tools(git_staging_config(settings)))
    vision_tools = (
        build_vision_tools(
            settings,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            upload_roots=settings.rarespot_upload_roots,
            paper_context=context,
        )
        if _should_register_vision_subagent(context, settings)
        else []
    )
    qwen_coding_model = (
        build_vision_chat_model(settings)
        if _should_register_qwen_code_runner(context, settings)
        else None
    )
    subagents = build_subagents(
        paper_tools,
        context=context,
        context_tools=[
            *context_tools,
            *calphad_tools,
            *kinetics_tools,
            *crystal_plasticity_tools,
            *degradation_characterization_tools,
            *sensor_tools,
        ],
        text_only_model=not settings.model_supports_multimodal,
        skills_sources=skills_sources,
        vision_tools=vision_tools,
        qwen_coding_model=qwen_coding_model,
    )
    # Durable attempt-ledger digest -> system prompt, on the coordinator AND every
    # coding delegate (in the 6.7M-token livelock, code-runner owned 69% of the
    # spend — the amnesia has to be fixed where the loop actually runs). One shared
    # instance: the ledger is per-run state and the read path is mtime-cached.
    ledger_middleware: UltraAttemptLedgerMiddleware | None = None
    if workspace_dir is not None:
        ledger_middleware = UltraAttemptLedgerMiddleware(attempt_ledger_path(Path(workspace_dir)))
        middleware.append(ledger_middleware)
        coding_delegates = {*_SKILL_BEARING_SUBAGENTS, QWEN_CODE_RUNNER_NAME}
        for subagent in subagents:
            if subagent.get("name") in coding_delegates:
                subagent.setdefault("middleware", []).append(ledger_middleware)
    # The Builder: a model-agnostic autonomous-coding sub-coordinator (a full deep agent
    # with its own GoalLoop + recursion cap) the coordinator delegates a verify-driven
    # GOAL to, so iterative build/debug loops run in ITS isolated context instead of
    # accumulating inline in the coordinator.
    #
    # CLEANUP CANDIDATE (2026-07-01): DISABLED by default (settings.builder_enabled=False)
    # after a live A/B on the comp-bio task — it worked but concentrated ~92% of the run's
    # tokens in its isolated context for little net gain over the plain coordinator +
    # code-runner path (which the progress-stall guard already protects). This whole
    # block, build_builder_subagent, BUILDER_DELEGATION_GUIDANCE, and the builder_*
    # settings are removable if the Builder stays off. Returns None when disabled, so
    # this is a no-op wire today; kept behind the flag for a future A/B.
    builder_subagent = build_builder_subagent(
        settings,
        tools=resolved_tools,
        backend=resolved_backend,
        vision_tools=vision_tools,
        # The Builder is a pre-compiled CompiledSubAgent, so deepagents does NOT inherit
        # the coordinator's permissions onto it — pass the same memory denies explicitly
        # so the Builder lead + its worker cannot write /policies, /skills, /memories.
        permissions=resolve_memory_permissions(settings),
        extra_middleware=[ledger_middleware] if ledger_middleware is not None else None,
    )
    if builder_subagent is not None:
        subagents = [*subagents, builder_subagent]
    async_subagents = build_async_subagents(settings, context=context)
    if async_subagents:
        middleware.append(UltraAsyncSubagentContextMiddleware(async_subagents))
    compute_resources = sandbox_compute_resources(settings)
    capability_manifest_tool = build_tool_capability_manifest_tool(
        resolved_tools,
        available_subagents=subagents,
        available_async_subagents=async_subagents,
        compute_resources=compute_resources,
    )
    registered_tools = [*resolved_tools, capability_manifest_tool]
    domain_manifest = build_tool_capability_manifest(
        registered_tools,
        available_subagents=subagents,
        available_async_subagents=async_subagents,
        compute_resources=compute_resources,
    )
    full_manifest = build_tool_capability_manifest(
        registered_tools,
        available_subagents=subagents,
        available_async_subagents=async_subagents,
        compute_resources=compute_resources,
    )
    system_prompt = build_system_prompt(settings, context)
    if surface_attestation_sink is not None:
        surface_attestation_sink(
            {
                "surface_source": "build_research_agent",
                "domain_tool_manifest_sha256": _agent_surface_sha256(domain_manifest),
                "full_tool_manifest_sha256": _agent_surface_sha256(full_manifest),
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            }
        )
    resolved_tools = registered_tools
    all_subagents = [*subagents, *async_subagents]

    resolved_model = model or build_chat_model(settings)

    return create_deep_agent(
        name="ultra-research-agent",
        model=resolved_model,
        tools=resolved_tools,
        system_prompt=system_prompt,
        context_schema=AgentRunContext,
        subagents=all_subagents,
        skills=skills_sources,
        backend=resolved_backend,
        memory=[] if cleanroom else MEMORY_PATHS,
        permissions=resolve_memory_permissions(settings),
        middleware=middleware,
        checkpointer=checkpointer,
    )


def _agent_surface_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _should_register_paper_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    if _has_ingested_papers(context):
        return True
    goal = str(context.goal or "").lower()
    return "arxiv.org" in goal or ".pdf" in goal or "arxiv:" in goal


def _has_ingested_papers(context: AgentRunContext) -> bool:
    ingested = context.knowledge_context.get("ingested_papers")
    return isinstance(ingested, list) and any(isinstance(item, dict) for item in ingested)


def _should_register_bisque_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    if is_cleanroom_evaluation_profile(context.evaluation_profile):
        return False
    if _has_bisque_account_binding(context):
        return True
    if context.selected_resource_uris or context.selected_dataset_uris:
        return True
    if any(str(pack).lower() == "bisque" for pack in context.allowed_tool_packs):
        return True
    return any(token in str(context.goal or "").lower() for token in ("bisque", "bqapi"))


def _has_bisque_account_binding(context: AgentRunContext) -> bool:
    if str(context.run_metadata.get("bisque_session_id") or "").strip():
        return True
    binding = context.run_metadata.get("bisque_account_binding")
    return isinstance(binding, dict) and str(binding.get("session_sha256") or "").strip() != ""


_GIT_GOAL_TOKENS = (
    "git clone",
    "clone the repo",
    "clone my repo",
    "clone a repo",
    "clone this repo",
    "git repository",
    ".git",
)


def _should_register_git_tools(context: AgentRunContext | None, settings: RuntimeSettings) -> bool:
    """Register the git staging tool only when the goal references a repo.

    Keeps the tool surface lean (like paper/bisque/rarespot gating): triggers on
    explicit git phrases or any allowlisted clone host appearing in the goal.
    """
    if context is None or not settings.git_staging_enabled:
        return False
    goal = str(context.goal or "").lower()
    if any(token in goal for token in _GIT_GOAL_TOKENS):
        return True
    return any(
        host.strip().lower() in goal
        for host in settings.git_staging_allowed_hosts
        if str(host or "").strip()
    )


def git_staging_config(settings: RuntimeSettings) -> GitStagingConfig:
    return GitStagingConfig(
        enabled=settings.git_staging_enabled,
        allowed_hosts=tuple(settings.git_staging_allowed_hosts),
        max_bytes=settings.git_staging_max_bytes,
        timeout_seconds=settings.git_staging_timeout_seconds,
        depth=settings.git_staging_depth,
    )


def _should_register_episodic_tools(context: AgentRunContext | None) -> bool:
    """Episodic memory is broadly useful, so register it for any authenticated
    researcher (a real, non-anonymous ``user_id``). The agent decides when to
    call it; anonymous/dev runs without an identity have no durable history to
    search and skip the tool to keep their tool surface lean."""
    if context is None:
        return False
    if is_cleanroom_evaluation_profile(context.evaluation_profile):
        return False
    return bool(str(context.user_id or "").strip())


def _should_register_resource_tools(context: AgentRunContext | None) -> bool:
    """Catalog search + staging is core to autonomous analysis (pull my own prior
    data into the sandbox), so register it for any authenticated researcher. The
    control plane scopes every query to the run owner, so anonymous runs without
    an identity have no catalog to search and skip the tools."""
    if context is None:
        return False
    if is_cleanroom_evaluation_profile(context.evaluation_profile):
        return False
    return bool(str(context.user_id or "").strip())
