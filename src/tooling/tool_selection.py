"""Tool subset selection heuristics for active chat runtimes."""

from __future__ import annotations

import re
from pathlib import Path


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg.get("content"))
    return ""


def _tool_map(tools: list[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for tool in tools:
        try:
            name = str(tool["function"]["name"])
        except Exception:
            continue
        mapped[name] = tool
    return mapped


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(needle in haystack for needle in needles)


def _explicit_tool_mentions(user_text: str, tool_names: set[str]) -> set[str]:
    text = str(user_text or "").lower()
    return {name for name in tool_names if str(name).lower() in text}


def _is_metadata_first_image_request(user_text: str) -> bool:
    text = str(user_text or "").lower()
    return _contains_any(
        text,
        (
            "what can you tell me about this image",
            "what can you tell me about these images",
            "tell me about this image",
            "tell me about these images",
            "describe this image",
            "describe these images",
            "summarize this image",
            "summarize these images",
            "quick look at this image",
            "quick look at these images",
            "first look at this image",
            "first look at these images",
            "what can you infer from this image",
            "what can you infer from these images",
            "metadata only for this image",
            "metadata only for these images",
            "before running analysis",
            "before running heavy analysis",
            "before running any analysis",
            "before running any tools",
            "just inspect this image",
            "just inspect these images",
        ),
    )


def _uploaded_path_looks_like_ground_truth(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    if not lowered:
        return False
    if lowered.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffixes = Path(lowered).suffixes
        suffix = (
            "".join(suffixes[-2:]).lower() if len(suffixes) >= 2 else Path(lowered).suffix.lower()
        )
    if suffix not in {
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    words = set(normalized.split())
    if words.intersection(
        {"gt", "label", "labels", "annotation", "annotations", "annot", "manual", "target", "truth"}
    ):
        return True
    return "ground truth" in normalized or "groundtruth" in normalized


def _select_tool_subset(
    messages: list[dict[str, str]], uploaded_files: list | None, all_tools: list[dict]
) -> list[dict]:
    text = _latest_user_message(messages).lower()
    has_uploads = bool(uploaded_files)
    image_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".ome.tif",
        ".ome.tiff",
        ".czi",
        ".nd2",
        ".lif",
        ".lsm",
        ".svs",
        ".vsi",
        ".dv",
        ".r3d",
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".mhd",
    }
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    tabular_suffixes = {".csv", ".tsv", ".tab", ".txt", ".csv.gz", ".tsv.gz"}
    normalized_suffixes: set[str] = set()
    for path in uploaded_files or []:
        if not isinstance(path, str):
            continue
        lower = str(path).strip().lower()
        parsed = Path(lower)
        if parsed.suffix:
            normalized_suffixes.add(parsed.suffix)
        if len(parsed.suffixes) >= 2:
            normalized_suffixes.add("".join(parsed.suffixes[-2:]))
    has_pdf_upload = ".pdf" in normalized_suffixes
    has_image_upload = bool(normalized_suffixes.intersection(image_suffixes))
    has_video_upload = bool(normalized_suffixes.intersection(video_suffixes))
    has_table_upload = bool(normalized_suffixes.intersection(tabular_suffixes))
    has_ground_truth_upload = any(
        _uploaded_path_looks_like_ground_truth(str(path))
        for path in (uploaded_files or [])
        if isinstance(path, str)
    )

    bisque_core_tools = {
        "bisque_find_assets",
        "upload_to_bisque",
        "bisque_ping",
        "bisque_download_resource",
        "search_bisque_resources",
        "load_bisque_resource",
        "bisque_download_dataset",
        "bisque_create_dataset",
        "bisque_add_to_dataset",
    }
    bisque_extended_tools = {
        "delete_bisque_resource",
        "add_tags_to_resource",
        "bisque_fetch_xml",
        "bisque_add_gobjects",
        "bisque_advanced_search",
        "run_bisque_module",
    }
    detection_core_tools = {
        "yolo_list_finetuned_models",
        "yolo_detect",
        "yolo_finetune_detect",
        "quantify_objects",
    }
    segmentation_core_tools = {
        "bioio_load_image",
        "segment_image_megaseg",
        "segment_image_sam3",
        "quantify_segmentation_masks",
    }
    segmentation_interactive_tools = {
        "segment_image_sam2",
        "sam2_prompt_image",
        "segment_video_sam2",
    }
    depth_tools = {"estimate_depth_pro"}
    chemistry_tools = {
        "structure_report",
        "compare_structures",
        "propose_reactive_sites",
        "formula_balance_check",
    }
    stats_tools = {
        "compare_conditions",
        "stats_list_curated_tools",
        "stats_run_curated_tool",
    }
    tabular_tools = {"analyze_csv"}
    code_execution_tools = {"codegen_python_plan", "execute_python_job"}

    selected_names: set[str] = {"repro_report"}
    by_name = _tool_map(all_tools)
    explicit_mentions = _explicit_tool_mentions(text, set(by_name.keys()))
    mentions_bisque = _contains_any(
        text,
        ("bisque", "dataset", "resource", "module", "upload", "download"),
    )
    mentions_segmentation = _contains_any(
        text,
        ("segment", "segmentation", "sam2", "medsam2", "sam3", "mask", "track"),
    )
    mentions_megaseg = _contains_any(text, ("megaseg", "dynunet"))
    mentions_depth = _contains_any(
        text,
        ("depth", "depth map", "depth estimation", "monocular depth", "depthpro"),
    )
    mentions_chemistry = _contains_any(
        text,
        (
            "organic chemistry",
            "reaction sequence",
            "reaction pathway",
            "mechanism",
            "retrosynthesis",
            "wittig",
            "pdc",
            "tsoh",
            "smiles",
            "inchi",
            "molecule",
        ),
    )
    mentions_evaluation = _contains_any(
        text,
        (
            "evaluate",
            "evaluation",
            "ground truth",
            "ground_truth",
            "label",
            "dice",
            "iou",
            "benchmark",
        ),
    )
    mentions_detection = _contains_any(
        text,
        ("yolo", "detection", "detect", "bbox", "bounding box", "class table"),
    )
    mentions_yolo_detection = _contains_any(
        text,
        ("yolo", "object detection", "bounding box", "bbox", "class table"),
    )
    mentions_edge_module = _contains_any(
        text,
        (
            "edge detection",
            "canny",
            "canny edge",
            "edge map",
            "run edge module",
            "edgedetection module",
        ),
    )
    mentions_asset_discovery = _contains_any(
        text,
        (
            "search",
            "find",
            "list",
            "show me",
            "what files",
            "what images",
            "assets",
            "resources",
        ),
    )
    mentions_recent_bisque_catalog = _contains_any(
        text,
        (
            "recently uploaded",
            "recent uploads",
            "latest uploads",
            "most recent",
            "what jpg",
            "what jpeg",
            "what png",
            "what tif",
            "what tiff",
            "what files",
            "what images",
            "which files",
            "which images",
        ),
    )
    mentions_simple_bisque_catalog = bool(
        mentions_bisque and (mentions_asset_discovery or mentions_recent_bisque_catalog)
    )
    mentions_stats = _contains_any(
        text,
        (
            "compare",
            "condition",
            "treatment",
            "control",
            "effect size",
            "confidence interval",
            "p-value",
            "statistical",
            "hypothesis",
            "significant",
        ),
    )
    mentions_tabular = _contains_any(
        text,
        (
            "csv",
            "tsv",
            "dataframe",
            "pandas",
            "spreadsheet",
            "tabular",
            "groupby",
            "data cleaning",
            "malformed csv",
            "column ",
            "row ",
        ),
    )
    mentions_code_execution = _contains_any(
        text,
        (
            "write code",
            "run code",
            "python script",
            "python code",
            "sandbox",
            "execute python",
            "debug code",
            "fix code",
            "pca",
            "random forest",
            "scikit-learn",
            "sklearn",
            "opencv",
            "scipy",
            "numpy",
            "pandas",
        ),
    )
    mentions_image_measurements = _contains_any(
        text,
        (
            "measure",
            "measurement",
            "morphology",
            "shape analysis",
            "feature extraction",
            "mask statistics",
            "area fraction",
            "object size",
        ),
    )
    mentions_one_step = _contains_any(
        text,
        ("one step", "single step", "minimum necessary tools", "minimum tools only"),
    )
    mentions_sam2_flow = _contains_any(
        text,
        ("sam2", "medsam2", "prompt point", "interactive prompt", "track video"),
    )
    metadata_first_image_request = _is_metadata_first_image_request(text)

    if mentions_bisque:
        selected_names.update(bisque_core_tools)
    if mentions_detection:
        selected_names.update(detection_core_tools)
    if mentions_segmentation:
        selected_names.update(segmentation_core_tools)
    if mentions_megaseg:
        selected_names.add("segment_image_megaseg")
    if mentions_sam2_flow:
        selected_names.update(segmentation_interactive_tools)
    if mentions_depth:
        selected_names.update(depth_tools)
    if mentions_chemistry:
        selected_names.update(chemistry_tools)
    if any(
        keyword in text
        for keyword in (
            "quantify mask",
            "mask quantification",
            "segmentation quantification",
            "regionprops",
            "mask-based",
        )
    ):
        selected_names.update(
            {"quantify_segmentation_masks", "evaluate_segmentation_masks", "repro_report"}
        )
    if mentions_stats:
        selected_names.update(stats_tools)
    if mentions_tabular:
        selected_names.update(tabular_tools)
        selected_names.update({"stats_list_curated_tools", "stats_run_curated_tool"})
    if mentions_code_execution:
        selected_names.update(code_execution_tools)
    if mentions_image_measurements:
        selected_names.update(
            {
                "bioio_load_image",
                "segment_image_megaseg",
                "segment_image_sam3",
                "quantify_segmentation_masks",
            }
        )

    if mentions_segmentation and mentions_evaluation:
        selected_names.update({"evaluate_segmentation_masks", "quantify_segmentation_masks"})
        if not has_uploads or has_ground_truth_upload:
            selected_names.add("segment_evaluate_batch")
        if (
            "segment_evaluate_batch" in by_name
            and mentions_one_step
            and not (
                {"segment_image_sam3", "evaluate_segmentation_masks", "segment_evaluate_batch"}
                & explicit_mentions
            )
            and (not has_uploads or has_ground_truth_upload)
        ):
            selected_names.discard("segment_image_sam3")
            selected_names.discard("evaluate_segmentation_masks")
            selected_names = {
                "segment_evaluate_batch",
                "quantify_segmentation_masks",
                "repro_report",
                "bioio_load_image",
            }

    if mentions_segmentation and not mentions_detection:
        selected_names.add("quantify_segmentation_masks")
    if mentions_segmentation and "quantify_objects" not in explicit_mentions:
        selected_names.discard("quantify_objects")

    if _contains_any(text, ("delete resource", "remove resource", "delete from bisque")):
        selected_names.add("delete_bisque_resource")
    if _contains_any(text, ("add tag", "tag resource", "metadata tag", "annotate metadata")):
        selected_names.add("add_tags_to_resource")
    if _contains_any(text, ("xml", "mex", "raw metadata")):
        selected_names.add("bisque_fetch_xml")
    if _contains_any(text, ("advanced search", "lucene", "query xml")):
        selected_names.add("bisque_advanced_search")
    if _contains_any(text, ("run module", "bisque module")):
        selected_names.add("run_bisque_module")
    if mentions_edge_module:
        selected_names.add("run_bisque_module")
        if not has_uploads or mentions_asset_discovery:
            selected_names.update(bisque_core_tools)
    if _contains_any(text, ("gobject", "gobjects", "polygon annotation", "add annotations")):
        selected_names.add("bisque_add_gobjects")
    if selected_names.intersection(bisque_extended_tools):
        selected_names.update(bisque_core_tools)

    if mentions_edge_module and has_uploads and not mentions_asset_discovery:
        selected_names.discard("search_bisque_resources")
        selected_names.discard("bisque_find_assets")
        selected_names.discard("bisque_advanced_search")
        selected_names.discard("load_bisque_resource")
        selected_names.discard("bisque_download_resource")
        selected_names.discard("bisque_download_dataset")
        selected_names.discard("bisque_create_dataset")

    if mentions_edge_module and not mentions_yolo_detection:
        selected_names.discard("yolo_detect")
        selected_names.discard("yolo_finetune_detect")
        selected_names.discard("yolo_list_finetuned_models")
        selected_names.discard("quantify_objects")

    if mentions_simple_bisque_catalog:
        selected_names.add("search_bisque_resources")
        if not _contains_any(
            text, ("metadata", "dimensions", "header", "uri", "resource uri", "client view")
        ):
            selected_names.discard("load_bisque_resource")
        for tool_name in (
            "upload_to_bisque",
            "bisque_create_dataset",
            "bisque_add_to_dataset",
            "delete_bisque_resource",
            "add_tags_to_resource",
            "bisque_add_gobjects",
            "run_bisque_module",
            "bisque_download_resource",
            "bisque_download_dataset",
            "bisque_advanced_search",
            "bisque_find_assets",
        ):
            if tool_name not in explicit_mentions:
                selected_names.discard(tool_name)

    if has_uploads and not selected_names.intersection(
        detection_core_tools
        | segmentation_core_tools
        | segmentation_interactive_tools
        | depth_tools
    ):
        if has_table_upload and not has_image_upload and not has_video_upload:
            selected_names.update(
                {"analyze_csv", "stats_list_curated_tools", "stats_run_curated_tool"}
            )
        elif has_video_upload and not has_pdf_upload and not has_image_upload:
            selected_names.update(
                {
                    "bioio_load_image",
                    "segment_image_sam3",
                    "estimate_depth_pro",
                    "yolo_detect",
                    "quantify_segmentation_masks",
                }
            )
        elif has_image_upload:
            selected_names.update({"bioio_load_image"})

    if has_image_upload and metadata_first_image_request:
        selected_names.discard("segment_image_sam3")
        selected_names.discard("segment_evaluate_batch")
        selected_names.discard("evaluate_segmentation_masks")
        selected_names.discard("quantify_segmentation_masks")
        selected_names.discard("quantify_objects")
        selected_names.discard("yolo_detect")
        selected_names.discard("estimate_depth_pro")
        selected_names.add("bioio_load_image")

    if len(selected_names) <= 1:
        if has_table_upload:
            selected_names.update(
                {"analyze_csv", "stats_list_curated_tools", "stats_run_curated_tool"}
            )
        elif has_image_upload:
            selected_names.update({"bioio_load_image"})
        elif mentions_code_execution:
            selected_names.update(code_execution_tools)
        else:
            selected_names.update({"bisque_find_assets", "bioio_load_image", "analyze_csv"})

    selected_names.update(explicit_mentions)

    subset: list[dict] = []
    for name in selected_names:
        tool = by_name.get(name)
        if tool is not None:
            subset.append(tool)

    ordered = [tool for tool in all_tools if tool in subset]
    return ordered if ordered else all_tools
