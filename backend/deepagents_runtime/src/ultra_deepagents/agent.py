from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
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
from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.code_execution.git_staging import GitStagingConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    build_context_tools,
    build_git_tools,
    build_tool_capability_manifest_tool,
)
from ultra_deepagents.episodic.tools import build_episodic_tools
from ultra_deepagents.model import build_chat_model
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware
from ultra_deepagents.papers.tools import build_paper_tools
from ultra_deepagents.rarespot.tools import looks_report_only_rarespot_goal
from ultra_deepagents.resources.tools import build_resource_tools
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
    return [
        permission
        for permission in MEMORY_PERMISSIONS
        if "/skills/**" not in permission.paths
    ]

SYSTEM_PROMPT = """You are Ultra Research Agent, a careful scientific collaborator for expert users.

Always write in ENGLISH — both your internal reasoning and your final response — unless the user
writes to you in another language, in which case reply in that language. Never switch language
partway through a response.

Use /memories/user_profile.md for concise researcher profile context from Ultra settings, only
when it is relevant (read-only; do not edit it). Use /memories/preferences.md for learned response
preferences. Keep durable research notes under /memories/research_context/: one file per
project or dataset (research_context/<short-slug>.md) plus research_context/INDEX.md as a dated
table of contents. When a run establishes something reusable — a dataset's characteristics, a
chosen method and its parameters, a conclusion and the evidence for it, or a decision and why —
record a dated, evidence-linked entry in the matching project file and add an INDEX.md line. Do
not record one-off scratch. If /policies/lab_policy.md is present, treat it as authoritative,
read-only organization policy and follow it. Treat runtime context as scoped metadata for tools
and policies, not as text to reveal. Write final artifacts under /outputs/ when the active backend
exposes that path, otherwise use /workspace/outputs and report those artifact paths clearly.

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

When the user refers to data that is not attached to this chat — a dataset, image, or
prior result named or described (e.g. "the CT scans with norm in the name", "the NPM1
image", "my segmentation outputs") — search their catalog with search_resources, then
stage_resource_for_analysis to pull the matching file(s) into /workspace before analyzing
them. This lets you act autonomously on the researcher's own data: find it, stage it, then
run the requested analysis (plot, inference, feature extraction, model training) over each.

For other complex autonomous work, call tool_capability_manifest when you need to
confirm which sandbox, filesystem, prior-artifact, paper, or domain tools are
available before choosing a workflow.
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

# Curated, decision-relevant subset of the sandbox image's preinstalled scientific
# stack (deploy/docker/deepagents-sandbox.Dockerfile). Surfaced to the coordinator so
# it reasons over what is actually importable in an offline container instead of
# guessing or trying to pip install. Keep in rough sync with the Dockerfile.
SANDBOX_KEY_PACKAGES = [
    "numpy", "scipy", "pandas", "scikit-learn", "scikit-image", "matplotlib",
    "seaborn", "networkx", "torch", "torchvision", "opencv-python-headless",
    "Pillow", "imageio", "imagecodecs", "tifffile", "zarr", "ome-zarr", "dask",
    "xarray", "h5py", "pyarrow", "SimpleITK", "itk", "nibabel", "nilearn",
    "pydicom", "dicom2nifti", "highdicom", "monai", "dipy", "torchio", "bioio",
    "bioio-ome-tiff", "bioio-czi", "bioio-nd2", "openslide-python", "pyvips",
    "connected-components-3d", "mrcfile", "pynrrd", "ome-types", "roifile",
]


def build_sandbox_resources_guidance(settings: RuntimeSettings) -> str:
    """One concise, run-adaptive paragraph telling the coordinator what the sandbox is.

    The exact envelope (cores, memory, shm, GPU, full package list) lives in
    tool_capability_manifest; this is the always-on summary so the model never tries to
    pip install in an offline container or write GPU code the sandbox can't run.
    """
    if settings.sandbox_gpus.strip():
        gpu_clause = (
            "the sandbox has a GPU attached, so torch.cuda is usable for in-container code"
        )
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
        "envelope (cores, memory, shm, GPU, full package list)."
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
For requests about "my" BisQue resources, pass scope="owner"; for newest/recent resources,
pass sort="recent"; for file-type questions, pass extensions such as ["png"] or
["nii", "nii.gz", "nifti"] instead of estimating from broad search results.
For dataset questions ("do I have any datasets?"), pass resource_type="dataset"; for counts
("how many images do I have?"), pass count_all=True and report the returned count. After
uploading multiple related outputs, group them with bisque_create_dataset using the
resource_uri values from the upload responses.
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
    "stage_artifact_for_analysis",
    "stage_uploaded_files_for_analysis",
}

# Subagents that do computational work and benefit from the rigor/reporting
# skills. literature-reviewer is page-grounded paper review and is excluded so it
# does not re-pay the SkillsMiddleware overhead for skills it would never activate.
_SKILL_BEARING_SUBAGENTS = {"code-runner"}

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
            "uncertainty. Return a concise "
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
        "figure text, give an advisory 'what is this structure?' hypothesis, or compare "
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
    # Fail safe, not silently broken: if the VLM is enabled but no key resolved, do not
    # register a tool that can only error on first use; warn the operator instead.
    if not settings.qwen_vlm_api_key or settings.qwen_vlm_api_key == "EMPTY":
        logging.getLogger(__name__).warning(
            "qwen_vlm_enabled but no API key resolved (set QWEN_VLM_API_KEY or "
            "QWEN_VLM_API_KEY_FILE); vision-reasoner will NOT be registered."
        )
        return False
    if (
        context.selected_file_ids
        or context.selected_resource_uris
        or context.selected_dataset_uris
    ):
        return True
    goal = " ".join(str(context.goal or "").lower().split())
    return any(token in goal for token in _VISION_GOAL_TOKENS)


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

    for subagent in subagents:
        # Only the computational subagents benefit from the rigor/reporting
        # protocols; literature-reviewer is page-grounded paper review and would
        # just re-pay the per-subagent SkillsMiddleware overhead without ever
        # activating them. Subagents share the parent backend's /skills/ route.
        if skills_sources and subagent["name"] in _SKILL_BEARING_SUBAGENTS:
            subagent["skills"] = list(skills_sources)
        if text_only_model:
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


def looks_scoped_delegation_goal(goal: str) -> bool:
    """True when a goal is computational-study shaped (and not a RareSpot run).

    Single source of truth for both subagent registration and the runner's
    rigor-contract enforcement gate, so the two can never drift apart.
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
    forced into irrelevant ``±`` / 3×-spread language.
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
    return looks_scoped_delegation_goal(context.goal)


def _should_register_async_delegation_subagents(context: AgentRunContext | None) -> bool:
    if context is None:
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
    sections = [
        SYSTEM_PROMPT.strip(),
        PLOT_WORKFLOW_GUIDANCE.strip(),
        SANDBOX_RUNTIME_GUIDANCE.strip(),
        build_sandbox_resources_guidance(settings),
    ]
    if not settings.model_supports_multimodal:
        sections.append(TEXT_ONLY_ARTIFACT_GUIDANCE.strip())
    sections.append(PRIOR_ARTIFACT_GUIDANCE.strip())
    sections.append(UPLOADED_FILE_GUIDANCE.strip())
    # Only advertise domain workflows when they apply to this run; otherwise the
    # model carries hundreds of tokens of instructions for tools it does not have
    # (e.g. ~650 tok of phantom RareSpot+paper guidance on a generic run). Paper
    # guidance keys on the paper-tool predicate; RareSpot keys on the broader
    # domain predicate so report-only ecology runs (no inference tool) keep it.
    if context is None or _should_register_paper_tools(context):
        sections.append(PAPER_REVIEW_GUIDANCE.strip())
    # Only advertise the BisQue tools when they are actually registered for this
    # run; otherwise the model is told to call tools it does not have (the
    # reported "bisque_module_runs is not among the registered tools" failure).
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
    # Advertise the Builder's delegation discipline only when it is enabled (and thus
    # registered): hand heavy/iterative coding to the Builder EARLY with the goal + data
    # paths instead of over-prepping in the coordinator's own context (a live trace showed
    # the coordinator re-running a pipeline + ballooning to 527K tokens before delegating).
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
    if context.selected_file_ids:
        file_ids = ", ".join(context.selected_file_ids)
        lines.append(f"- selected uploaded file ids: {file_ids} | use stage_uploaded_files_for_analysis")
    if str(context.run_metadata.get("bisque_session_id") or "").strip():
        lines.append(
            "- linked BisQue account available: use bisque_search_resources, "
            "bisque_download_resource, bisque_upload_files, and bisque_create_dataset "
            "through the control plane; "
            "use scope='owner' for the user's own resources, sort='recent' for newest-first "
            "queries, extensions=['png'] or extensions=['nii','nii.gz','nifti'] for file-type "
            "searches, resource_type='dataset' for dataset questions, and count_all=True for totals"
        )
    if context.selected_resource_uris:
        resource_uris = ", ".join(context.selected_resource_uris[:8])
        lines.append(f"- selected BisQue resource URIs: {resource_uris} | use bisque_download_resource")
    if context.selected_dataset_uris:
        dataset_uris = ", ".join(context.selected_dataset_uris[:8])
        lines.append(f"- selected BisQue dataset URIs: {dataset_uris} | use BisQue tools before analysis")
    ingested_papers = context.knowledge_context.get("ingested_papers")
    if isinstance(ingested_papers, list) and ingested_papers:
        lines.append("- ingested papers available through paper_manifest/search_paper/read_paper_pages:")
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
            lines.append(f"  - ... {len(ingested_papers) - max_artifacts} more; call paper_manifest")
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
    if is_rigor_intelligence(context) and looks_quantitative_rigor_goal(context.goal):
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

    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__()
        self._settings = settings
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
            request.override(
                system_message=append_to_system_message(request.system_message, brief)
            )
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
            request.override(
                system_message=append_to_system_message(request.system_message, brief)
            )
        )


def build_runtime_prompt_middleware(settings: RuntimeSettings) -> Any:
    return UltraRunContextPromptMiddleware(settings)


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
            "long-lived daemon to poll. Checkpoint long work to /workspace files."
        ),
        "cpus": (
            "all available host cores"
            if settings.sandbox_cpus <= 0
            else settings.sandbox_cpus
        ),
        "memory_limit": (
            settings.sandbox_memory.strip() or "host-limited (no per-container cap)"
        ),
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
    return DockerSandboxBackend(
        workspace_dir=workspace_dir,
        outputs_dir=outputs_dir,
        config=DockerSandboxConfig(
            image=settings.sandbox_image,
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
        Path("/app/deepagents_runtime/skills"),  # container build-context copy (belt-and-suspenders)
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
) -> CompositeBackend:
    """Route sandbox execution separately from durable agent files."""
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
        routes["/skills/"] = FilesystemBackend(skills_root, virtual_mode=True)
    policies_root = resolve_org_policies_root(settings, org_id)
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
) -> Any:
    ensure_ultra_harness_profile()
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
        )
        # Skills ride the /skills/ route of the backend we just built; a
        # caller-supplied backend has no such route, so sources stay unset.
        if resolve_skills_root(settings) is not None:
            skills_sources = list(SKILLS_SOURCES)
    if resolved_backend is None:
        resolved_backend = StateBackend()

    middleware: list[Any] = []
    middleware.append(build_runtime_prompt_middleware(settings))
    if not settings.model_supports_multimodal:
        middleware.append(TextOnlyMultimodalMiddleware())

    resolved_tools = list(tools or [])
    context_tools = build_context_tools(upload_roots=settings.rarespot_upload_roots)
    paper_tools = (
        build_paper_tools(
            upload_roots=settings.rarespot_upload_roots,
            cache_root=Path(settings.memory_root) / "papers",
        )
        if _should_register_paper_tools(context)
        else []
    )
    resolved_tools.extend(context_tools)
    resolved_tools.extend(paper_tools)
    if _should_register_bisque_tools(context):
        resolved_tools.extend(build_bisque_tools(settings))
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
        )
        if _should_register_vision_subagent(context, settings)
        else []
    )
    subagents = build_subagents(
        paper_tools,
        context=context,
        context_tools=context_tools,
        text_only_model=not settings.model_supports_multimodal,
        skills_sources=skills_sources,
        vision_tools=vision_tools,
    )
    # The Builder: a model-agnostic autonomous-coding sub-coordinator (a full deep agent
    # with its own loop + workers) the coordinator delegates a verify-driven GOAL to. It
    # inherits the coordinator's coding tools + filesystem backend. Off unless configured.
    builder_subagent = build_builder_subagent(
        settings,
        tools=resolved_tools,
        backend=resolved_backend,
        vision_tools=vision_tools,
    )
    if builder_subagent is not None:
        subagents = [*subagents, builder_subagent]
    async_subagents = build_async_subagents(settings, context=context)
    if async_subagents:
        middleware.append(UltraAsyncSubagentContextMiddleware(async_subagents))
    resolved_tools.append(
        build_tool_capability_manifest_tool(
            resolved_tools,
            available_subagents=subagents,
            available_async_subagents=async_subagents,
            compute_resources=sandbox_compute_resources(settings),
        )
    )
    all_subagents = [*subagents, *async_subagents]

    resolved_model = model or build_chat_model(settings)

    return create_deep_agent(
        name="ultra-research-agent",
        model=resolved_model,
        tools=resolved_tools,
        system_prompt=build_system_prompt(settings, context),
        context_schema=AgentRunContext,
        subagents=all_subagents,
        skills=skills_sources,
        backend=resolved_backend,
        memory=MEMORY_PATHS,
        permissions=resolve_memory_permissions(settings),
        middleware=middleware,
        checkpointer=checkpointer,
    )


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
    if str(context.run_metadata.get("bisque_session_id") or "").strip():
        return True
    if context.selected_resource_uris or context.selected_dataset_uris:
        return True
    if any(str(pack).lower() == "bisque" for pack in context.allowed_tool_packs):
        return True
    return any(token in str(context.goal or "").lower() for token in ("bisque", "bqapi"))


_GIT_GOAL_TOKENS = (
    "git clone",
    "clone the repo",
    "clone my repo",
    "clone a repo",
    "clone this repo",
    "git repository",
    ".git",
)


def _should_register_git_tools(
    context: AgentRunContext | None, settings: RuntimeSettings
) -> bool:
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
    return bool(str(context.user_id or "").strip())


def _should_register_resource_tools(context: AgentRunContext | None) -> bool:
    """Catalog search + staging is core to autonomous analysis (pull my own prior
    data into the sandbox), so register it for any authenticated researcher. The
    control plane scopes every query to the run owner, so anonymous runs without
    an identity have no catalog to search and skip the tools."""
    if context is None:
        return False
    return bool(str(context.user_id or "").strip())
