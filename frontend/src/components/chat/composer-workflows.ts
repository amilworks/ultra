import type { LucideIcon } from "lucide-react";
import {
  Calculator,
  Binoculars,
  Boxes,
  Code2,
  Cpu,
  Download,
  FileSearch,
  FolderPlus,
  FolderUp,
  Radar,
  ScanSearch,
  Scissors,
  Search,
  Tags,
  Trash2,
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
  category: "Resources" | "Vision" | "More tools";
  icon: LucideIcon;
  keywords: string[];
};

const makeWorkflowHint = (
  id: ChatWorkflowHint["id"]
): ChatWorkflowHint => ({
  id,
  source: "slash_menu",
});

export const COMPOSER_WORKFLOWS: ComposerWorkflowDefinition[] = [
  {
    id: "find_resource",
    label: "Find resource",
    description: "Search your resource catalog and stage files into the current chat.",
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
    label: "Find BisQue assets",
    description: "Search BisQue resources directly and return structured matches in chat.",
    category: "Resources",
    icon: Binoculars,
    prompt: "Find the most relevant BisQue assets for this request and summarize the matches.",
    selectedToolNames: ["bisque_find_assets"],
    workflowHint: makeWorkflowHint("find_bisque_assets"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "search", "assets", "dataset", "resource", "find"],
  },
  {
    id: "search_bisque_resources",
    label: "Search BisQue",
    description: "Run a direct BisQue catalog search and return structured results.",
    category: "Resources",
    icon: Search,
    prompt: "Search BisQue resources for this request and summarize the best matches.",
    selectedToolNames: ["search_bisque_resources"],
    workflowHint: makeWorkflowHint("search_bisque_resources"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "search", "catalog", "resources", "query"],
  },
  {
    id: "bisque_advanced_search",
    label: "Advanced BisQue search",
    description: "Run a deeper tag-aware BisQue search for the current request.",
    category: "Resources",
    icon: FileSearch,
    prompt: "Run an advanced BisQue search for this request and summarize the results.",
    selectedToolNames: ["bisque_advanced_search"],
    workflowHint: makeWorkflowHint("bisque_advanced_search"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "advanced", "search", "tags", "filters"],
  },
  {
    id: "load_bisque_resource",
    label: "Load BisQue metadata",
    description: "Inspect the selected BisQue resource metadata and summarize it.",
    category: "Resources",
    icon: FileSearch,
    prompt: "Load the selected BisQue resource metadata and summarize the important fields.",
    selectedToolNames: ["load_bisque_resource"],
    workflowHint: makeWorkflowHint("load_bisque_resource"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "metadata", "inspect", "view", "resource"],
  },
  {
    id: "bisque_download_resource",
    label: "Download resource",
    description: "Download the selected BisQue resource into the current workspace.",
    category: "More tools",
    icon: Download,
    prompt: "Download the selected BisQue resource and summarize what was saved.",
    selectedToolNames: ["bisque_download_resource"],
    workflowHint: makeWorkflowHint("bisque_download_resource"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "download", "resource", "save", "local"],
  },
  {
    id: "bisque_download_dataset",
    label: "Download dataset",
    description: "Download the members of the selected BisQue dataset.",
    category: "More tools",
    icon: Download,
    prompt: "Download the selected BisQue dataset and summarize the saved members.",
    selectedToolNames: ["bisque_download_dataset"],
    workflowHint: makeWorkflowHint("bisque_download_dataset"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "dataset", "download", "members", "save"],
  },
  {
    id: "upload_to_bisque",
    label: "Upload to BisQue",
    description: "Upload the attached local files into BisQue.",
    category: "More tools",
    icon: Upload,
    prompt: "Upload the attached files to BisQue and summarize the ingested resources.",
    selectedToolNames: ["upload_to_bisque"],
    workflowHint: makeWorkflowHint("upload_to_bisque"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "upload", "ingest", "dataset", "store"],
  },
  {
    id: "bisque_create_dataset",
    label: "Create dataset",
    description: "Create a BisQue dataset from the current selected resources.",
    category: "More tools",
    icon: FolderPlus,
    prompt: "Create a BisQue dataset from the selected resources and summarize the new dataset.",
    selectedToolNames: ["bisque_create_dataset"],
    workflowHint: makeWorkflowHint("bisque_create_dataset"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "dataset", "create", "collection", "group"],
  },
  {
    id: "bisque_add_to_dataset",
    label: "Update dataset",
    description: "Add the selected resources into a BisQue dataset.",
    category: "More tools",
    icon: FolderUp,
    prompt: "Add the selected resources to the requested BisQue dataset and summarize the update.",
    selectedToolNames: ["bisque_add_to_dataset"],
    workflowHint: makeWorkflowHint("bisque_add_to_dataset"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "dataset", "add", "update", "append"],
  },
  {
    id: "bisque_add_gobjects",
    label: "Add annotations",
    description: "Write ROI or gobject annotations onto the selected BisQue resource.",
    category: "More tools",
    icon: Scissors,
    prompt: "Add the requested BisQue annotations to the selected resource and summarize what was written.",
    selectedToolNames: ["bisque_add_gobjects"],
    workflowHint: makeWorkflowHint("bisque_add_gobjects"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "annotation", "gobject", "roi", "polygon", "bbox"],
  },
  {
    id: "add_tags_to_resource",
    label: "Add tags",
    description: "Attach metadata tags to the selected BisQue resource.",
    category: "More tools",
    icon: Tags,
    prompt: "Add the requested tags to the selected BisQue resource and summarize the update.",
    selectedToolNames: ["add_tags_to_resource"],
    workflowHint: makeWorkflowHint("add_tags_to_resource"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "tags", "metadata", "label", "annotate"],
  },
  {
    id: "bisque_fetch_xml",
    label: "Fetch XML",
    description: "Fetch the raw XML representation for the selected BisQue resource.",
    category: "More tools",
    icon: Code2,
    prompt: "Fetch the raw XML for the selected BisQue resource and summarize what was returned.",
    selectedToolNames: ["bisque_fetch_xml"],
    workflowHint: makeWorkflowHint("bisque_fetch_xml"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "xml", "raw", "metadata", "resource"],
  },
  {
    id: "delete_bisque_resource",
    label: "Delete resource",
    description: "Delete the selected BisQue resource after approval.",
    category: "More tools",
    icon: Trash2,
    prompt: "Delete the selected BisQue resource and summarize the result.",
    selectedToolNames: ["delete_bisque_resource"],
    workflowHint: makeWorkflowHint("delete_bisque_resource"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "delete", "remove", "trash", "danger"],
  },
  {
    id: "segment_sam3",
    label: "MedSAM2 segmentation",
    description: "Run automatic MedSAM2 segmentation on the selected image resources.",
    category: "Vision",
    icon: Scissors,
    prompt: "Run automatic MedSAM2 segmentation on the selected image resources and summarize the masks.",
    selectedToolNames: ["segment_image_sam2"],
    workflowHint: makeWorkflowHint("segment_sam3"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["medsam", "medsam2", "sam3", "segment", "segmentation", "masks", "mask"],
  },
  {
    id: "detect_prairie_dog",
    label: "Prairie dog detection",
    description: "Use the prairie-specific detector on the selected resources.",
    category: "Vision",
    icon: Radar,
    prompt: "Run prairie dog detection on the selected image resources and summarize the detections.",
    selectedToolNames: ["yolo_detect"],
    workflowHint: makeWorkflowHint("detect_prairie_dog"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["prairie", "prairie dog", "burrow", "rarespot", "detection"],
  },
  {
    id: "detect_yolo",
    label: "YOLO detection",
    description: "Run the generic YOLO detector on the selected resources.",
    category: "Vision",
    icon: ScanSearch,
    prompt: "Run YOLO detection on the selected image resources and summarize the detections.",
    selectedToolNames: ["yolo_detect"],
    workflowHint: makeWorkflowHint("detect_yolo"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["yolo", "detection", "detect", "boxes", "bbox"],
  },
  {
    id: "estimate_depth_pro",
    label: "DepthPro",
    description: "Estimate monocular depth on the selected image resources.",
    category: "Vision",
    icon: Boxes,
    prompt: "Run DepthPro on the selected image resources and summarize the depth results.",
    selectedToolNames: ["estimate_depth_pro"],
    workflowHint: makeWorkflowHint("estimate_depth_pro"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["depth", "depthpro", "depth pro", "depth map", "monocular"],
  },
  {
    id: "pro_mode",
    label: "Pro Mode",
    description: "Run the Pro Mode control plane with dynamic routing across fast dialogue, grounded tools, iterative research, and expert reasoning.",
    category: "More tools",
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
    id: "scientific_calculator",
    label: "Scientific calculator",
    description: "Enable a lightweight NumPy-backed calculator for deterministic numeric work.",
    category: "More tools",
    icon: Calculator,
    prompt:
      "Use the scientific calculator for numeric substitutions, trigonometry, sums, and array math when the formula is already known, then explain the result briefly.",
    selectedToolNames: ["numpy_calculator"],
    workflowHint: makeWorkflowHint("scientific_calculator"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["calculator", "numpy", "math", "numeric", "physics", "compute"],
  },
  {
    id: "chemistry_workbench",
    label: "Chemistry workbench",
    description: "Enable deterministic chemistry tools for structure facts, deltas, and reaction-site cues.",
    category: "More tools",
    icon: Cpu,
    prompt:
      "Use the chemistry workbench tools when helpful to ground structure facts, functional-group changes, ring strain, and reactive-site cues before answering.",
    selectedToolNames: [
      "structure_report",
      "compare_structures",
      "propose_reactive_sites",
      "formula_balance_check",
    ],
    workflowHint: makeWorkflowHint("chemistry_workbench"),
    requiresAttachedFiles: false,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["chemistry", "organic", "reaction", "mechanism", "structure", "smiles"],
  },
  {
    id: "run_bisque_module",
    label: "Run BisQue module",
    description: "Execute a BisQue module against the selected resources.",
    category: "More tools",
    icon: Cpu,
    prompt: "Run the requested BisQue module on the selected resources and summarize the output.",
    selectedToolNames: ["run_bisque_module"],
    workflowHint: makeWorkflowHint("run_bisque_module"),
    requiresAttachedFiles: true,
    opensResourcePickerOnSelect: false,
    clearsAfterResourcePick: false,
    keywords: ["bisque", "module", "pipeline", "workflow", "plugin"],
  },
];

export const COMPOSER_WORKFLOW_GROUP_ORDER: Array<ComposerWorkflowDefinition["category"]> = [
  "Resources",
  "Vision",
  "More tools",
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
