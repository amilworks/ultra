import type { LucideIcon } from "lucide-react";
import {
  Binoculars,
  Cpu,
  Download,
  FlaskConical,
  FolderPlus,
  ListChecks,
  Radar,
  ScanSearch,
  Scissors,
  Search,
  Target,
  Upload,
} from "lucide-react";

import type { ChatWorkflowHint } from "@/types";

export type ComposerWorkflowId = "find_resource" | ChatWorkflowHint["id"];

export type ComposerWorkflowPresetState = {
  id: ComposerWorkflowId;
  label: string;
  prompt: string;
  selectedToolNames: string[];
  workflowHint: ChatWorkflowHint | null;
  requiresAttachedFiles: boolean;
  opensResourcePickerOnSelect: boolean;
  clearsAfterResourcePick: boolean;
  persistsAcrossTurns?: boolean;
};

export type ComposerWorkflowDefinition = ComposerWorkflowPresetState & {
  description: string;
  category: "Workflows" | "Resources" | "BisQue" | "Vision";
  icon: LucideIcon;
  keywords: string[];
  /** Shown greyed-out and non-selectable with a "Soon" badge (e.g. unreleased pipelines). */
  comingSoon?: boolean;
};

const makeWorkflowHint = (
  id: ChatWorkflowHint["id"]
): ChatWorkflowHint => ({
  id,
  source: "slash_menu",
});

// BisQue tools register as ONE pack on the backend: any selected tool name equal
// to "bisque" turns on the full BisQue toolset (search/download/upload/dataset/
// module-runs). The model then picks the right tool, steered by the prompt below.
// We deliberately do NOT pass per-action tool names — the backend has no such
// gate, so they were decorative. See _should_register_bisque_tools in agent.py.
const BISQUE_TOOL_PACK = ["bisque"];

export const COMPOSER_WORKFLOWS: ComposerWorkflowDefinition[] = [
  {
    id: "goal_driven_build",
    label: "Goal-driven build",
    description: "Scoped iterate-until-verified loop with a clear done-when and a tight budget.",
    category: "Workflows",
    icon: Target,
    prompt:
      "Treat my request as a goal-driven build. First restate it as ONE measurable success criterion (a DONE-WHEN) plus a baseline. Then iterate efficiently — implement, run, measure, adjust — using the SMALLEST experiment that answers it; do NOT launch an exhaustive multi-seed or multi-duration sweep unless I explicitly ask. Stop as soon as the criterion is met (or you can make a clear call) and report concisely: the result, how you verified it, and one caveat. Render math as LaTeX (inline \\( \\); display equations on their own line with $$ … $$, never inside a > quote) and write absolute value as \\lvert x \\rvert so Markdown tables don't break.\n\nGoal: ",
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["goal", "build", "iterate", "until", "objective", "optimize", "loop"],
  },
  {
    id: "quantitative_analysis",
    label: "Quantitative analysis",
    description:
      "Stats + plots with a results contract (uncertainty, decision rule) and clean formatting.",
    category: "Workflows",
    icon: FlaskConical,
    prompt:
      "Run a focused quantitative analysis of my request. State the question, the method, and a clear decision rule up front; report estimates WITH uncertainty (a CI or ±1σ) and the resulting decision; keep the experiment proportionate to the question. Present results in clean Markdown tables, and render every formula as LaTeX — inline with \\( \\) and display equations on their own line with $$ … $$ (never inside a > quote). Inside table cells write absolute value as \\lvert x \\rvert (never bare |…|) so the table renders.\n\nAnalysis request: ",
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: [
      "quantitative",
      "analysis",
      "statistics",
      "stats",
      "uncertainty",
      "plot",
      "regression",
    ],
  },
  {
    id: "image_analysis",
    label: "Image analysis",
    description: "Describe, measure, and flag anomalies in the attached scientific image(s).",
    category: "Workflows",
    icon: ScanSearch,
    prompt:
      "Analyze the attached/selected scientific image(s) for my request: describe what's visible, measure or quantify what's relevant, and flag anything notable or anomalous — grounded in the pixels, and say so when uncertain. Keep any math in LaTeX (inline \\( \\); display $$ on its own line).\n\nRequest: ",
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["image", "analysis", "measure", "describe", "microscopy", "inspect", "vision"],
  },
  {
    id: "pro_mode",
    label: "Pro Mode",
    description:
      "Dynamic routing across fast dialogue, grounded tools, iterative research, and expert reasoning.",
    category: "Workflows",
    icon: Cpu,
    prompt: "",
    selectedToolNames: [],
    workflowHint: makeWorkflowHint("pro_mode"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    persistsAcrossTurns: true,
    keywords: ["pro", "council", "workflow", "reasoning", "scientific", "social"],
  },
  {
    id: "find_resource",
    label: "Find resource",
    description: "Search your uploaded files and stage them into this chat.",
    category: "Resources",
    icon: Search,
    prompt: "",
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: true,
    clearsAfterResourcePick: true,
    keywords: ["resource", "file", "browser", "catalog", "attach", "stage"],
  },
  {
    id: "find_bisque_assets",
    label: "Search BisQue",
    description: "Search your BisQue catalog — images, datasets, files, or tables — by name, tag, or metadata.",
    category: "BisQue",
    icon: Binoculars,
    prompt:
      "Search my BisQue catalog for this request and summarize the best matches. Apply tag or metadata filters when the request implies them (modality, species, age, year, file type).",
    selectedToolNames: BISQUE_TOOL_PACK,
    workflowHint: makeWorkflowHint("find_bisque_assets"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: [
      "bisque",
      "search",
      "find",
      "assets",
      "dataset",
      "resource",
      "catalog",
      "tags",
      "metadata",
      "advanced",
    ],
  },
  {
    id: "bisque_download_resource",
    label: "Download from BisQue",
    description: "Pull a BisQue resource into this chat's workspace for analysis.",
    category: "BisQue",
    icon: Download,
    prompt: "Download the requested BisQue resource into the workspace and summarize what was staged.",
    selectedToolNames: BISQUE_TOOL_PACK,
    workflowHint: makeWorkflowHint("bisque_download_resource"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "download", "fetch", "resource", "stage", "local"],
  },
  {
    id: "upload_to_bisque",
    label: "Upload to BisQue",
    description: "Push the attached files into your BisQue account.",
    category: "BisQue",
    icon: Upload,
    prompt: "Upload the attached files to BisQue and report each new resource's BisQue link.",
    selectedToolNames: BISQUE_TOOL_PACK,
    workflowHint: makeWorkflowHint("upload_to_bisque"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "upload", "ingest", "store", "push"],
  },
  {
    id: "bisque_create_dataset",
    label: "Create BisQue dataset",
    description: "Group BisQue resources into a new dataset.",
    category: "BisQue",
    icon: FolderPlus,
    prompt: "Create a BisQue dataset from the requested resources and summarize the new dataset.",
    selectedToolNames: BISQUE_TOOL_PACK,
    workflowHint: makeWorkflowHint("bisque_create_dataset"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "dataset", "create", "group", "collection"],
  },
  {
    // The backend BisQue tool (bisque_module_runs) only LISTS recent module-execution
    // (MEX) runs and inspects one run's outputs — it cannot launch a module. The label
    // says "runs" (not "Run a module") to match that capability honestly. id is kept as
    // run_bisque_module for type/telemetry continuity.
    id: "run_bisque_module",
    label: "BisQue module runs",
    description: "List your BisQue module runs, or inspect one run's results.",
    category: "BisQue",
    icon: ListChecks,
    prompt:
      "List my recent BisQue module runs and summarize their status. If I name a specific run, inspect its inputs and outputs.",
    selectedToolNames: BISQUE_TOOL_PACK,
    workflowHint: makeWorkflowHint("run_bisque_module"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "module", "mex", "runs", "status", "results", "history"],
  },
  {
    id: "detect_prairie_dog",
    label: "Prairie dog detection",
    description: "Use the prairie-specific detector on the selected resources.",
    category: "Vision",
    icon: Radar,
    prompt: "Run prairie dog detection on the selected image resources and summarize the detections.",
    // RareSpot detection is now the prairie-dog-detection Skill (run in the code
    // sandbox), not a forced dispatch tool — the prompt + keywords trigger it via
    // progressive disclosure, so no selected tool / workflow hint is needed.
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["prairie", "prairie dog", "burrow", "rarespot", "detection"],
  },
  {
    id: "megaseg",
    label: "MegaSeg segmentation",
    description: "Coming soon — instance / semantic segmentation for microscopy.",
    category: "Vision",
    icon: Scissors,
    comingSoon: true,
    prompt: "",
    selectedToolNames: [],
    workflowHint: null,
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["megaseg", "segmentation", "microscopy", "mask", "instance", "semantic"],
  },
];

export const COMPOSER_WORKFLOW_GROUP_ORDER: Array<ComposerWorkflowDefinition["category"]> = [
  "Workflows",
  "Resources",
  "BisQue",
  "Vision",
];

const WORKFLOW_BY_ID = new Map(
  COMPOSER_WORKFLOWS.map((workflow) => [workflow.id, workflow] as const)
);

export const getComposerWorkflowById = (
  workflowId: ComposerWorkflowId
): ComposerWorkflowDefinition | null => WORKFLOW_BY_ID.get(workflowId) ?? null;

export const toComposerWorkflowPresetState = (
  workflow: ComposerWorkflowDefinition
): ComposerWorkflowPresetState => ({
  id: workflow.id,
  label: workflow.label,
  prompt: workflow.prompt,
  selectedToolNames: [...workflow.selectedToolNames],
  workflowHint: workflow.workflowHint ? { ...workflow.workflowHint } : null,
  requiresAttachedFiles: workflow.requiresAttachedFiles,
  opensResourcePickerOnSelect: workflow.opensResourcePickerOnSelect,
  clearsAfterResourcePick: workflow.clearsAfterResourcePick,
  persistsAcrossTurns: workflow.persistsAcrossTurns ?? false,
});

export const coerceComposerWorkflowPresetState = (
  value: unknown
): ComposerWorkflowPresetState | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const workflowId = String(record.id ?? "").trim() as ComposerWorkflowId;
  const workflow = getComposerWorkflowById(workflowId);
  if (!workflow) {
    return null;
  }
  return {
    ...toComposerWorkflowPresetState(workflow),
    prompt:
      typeof record.prompt === "string" && record.prompt.trim()
        ? record.prompt
        : workflow.prompt,
    label:
      typeof record.label === "string" && record.label.trim()
        ? record.label
        : workflow.label,
    selectedToolNames: Array.isArray(record.selectedToolNames)
      ? record.selectedToolNames.map((item) => String(item || "").trim()).filter(Boolean)
      : [...workflow.selectedToolNames],
  };
};

const normalizeWorkflowToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/^\/+/, "")
    .replace(/\s+/g, " ");

export const filterComposerWorkflows = (
  rawQuery: string
): ComposerWorkflowDefinition[] => {
  const query = normalizeWorkflowToken(rawQuery);
  if (!query) {
    return [...COMPOSER_WORKFLOWS];
  }
  return COMPOSER_WORKFLOWS.filter((workflow) => {
    const searchText = [
      workflow.label,
      workflow.description,
      workflow.category,
      workflow.id,
      ...workflow.keywords,
    ]
      .join(" ")
      .toLowerCase();
    return searchText.includes(query);
  });
};
