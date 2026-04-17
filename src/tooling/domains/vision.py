"""Vision domain tool schemas (segmentation, depth, detection)."""

BIOIO_LOAD_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "bioio_load_image",
        "description": (
            "Universal scientific image loader powered by bioio. "
            "Loads microscopy/medical formats into normalized arrays and generates preview images "
            "(for 3D/5D data, preview uses middle Z slice). Use this before segmentation/quantification "
            "when format handling or dimensionality is uncertain, especially for scientific volumes such as NIfTI."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Local file path to load."},
                "scene": {
                    "oneOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Optional scene index or scene id.",
                },
                "use_aicspylibczi": {
                    "type": "boolean",
                    "description": "For CZI files, use aicspylibczi mode when available.",
                    "default": False,
                },
                "array_mode": {
                    "type": "string",
                    "enum": ["plane", "volume", "tczyx"],
                    "default": "plane",
                    "description": "plane=single YX/CYX view, volume=ZYX/CZYX, tczyx=full standardized tensor.",
                },
                "t_index": {"type": "integer", "description": "Optional T index."},
                "c_index": {"type": "integer", "description": "Optional C index."},
                "z_index": {"type": "integer", "description": "Optional Z index."},
                "save_array": {"type": "boolean", "default": True},
                "include_array": {
                    "type": "boolean",
                    "default": False,
                    "description": "Inline array payload for small arrays only.",
                },
                "max_inline_elements": {
                    "type": "integer",
                    "default": 16384,
                    "minimum": 64,
                    "maximum": 1000000,
                },
            },
            "required": ["file_path"],
        },
    },
}


SAM2_SEGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "segment_image_sam2",
        "description": (
            "Segment 2D/3D scientific images using MedSAM2. "
            "Internally uses bioio loading for dimensionality-safe input handling and returns mask artifacts "
            "plus coverage statistics. For uploads, prefer preferred_upload_paths (volume masks when available)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image/volume file paths to segment",
                    "items": {"type": "string"},
                },
                "points_per_batch": {
                    "type": "integer",
                    "description": "Deprecated SAM2 argument retained for compatibility.",
                    "default": 64,
                    "minimum": 16,
                    "maximum": 256,
                },
                "save_visualizations": {
                    "type": "boolean",
                    "description": "Whether to save overlay visualizations. Default: true",
                    "default": True,
                },
                "device": {
                    "type": "string",
                    "description": "Execution device: cpu, cuda, cuda:0, or GPU index string. Default: auto-select.",
                },
                "model_id": {
                    "type": "string",
                    "description": "Optional MedSAM2 model id override.",
                },
                "max_slices": {
                    "type": "integer",
                    "description": "Max number of slices to process explicitly for large 3D volumes. Use 0 to process every slice.",
                    "default": 160,
                    "minimum": 0,
                    "maximum": 2000,
                },
            },
            "required": ["file_paths"],
        },
    },
}


SAM2_PROMPT_TOOL = {
    "type": "function",
    "function": {
        "name": "sam2_prompt_image",
        "description": (
            "Run MedSAM2 prompted segmentation on a single 2D/3D image using points and/or bounding boxes. "
            "For 3D inputs, prompts are applied slice-wise. Output includes preferred_upload_path for BisQue upload targets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the image/volume file",
                },
                "input_points": {
                    "type": "array",
                    "description": "List of [x, y] points (single object) or list of list of points (multi-object).",
                    "items": {
                        "oneOf": [
                            {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            {"type": "array", "items": {"type": "array"}},
                        ]
                    },
                },
                "input_labels": {
                    "type": "array",
                    "description": "Labels for points (1=positive, 0=negative).",
                    "items": {"type": "integer"},
                },
                "input_boxes": {
                    "type": "array",
                    "description": "List of [x_min, y_min, x_max, y_max] boxes.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "model_id": {
                    "type": "string",
                    "description": "MedSAM2 model id",
                    "default": "wanglab/MedSAM2",
                },
                "multimask_output": {
                    "type": "boolean",
                    "description": "Whether to return multiple masks (default true).",
                    "default": True,
                },
                "save_visualization": {
                    "type": "boolean",
                    "description": "Save overlay visualization.",
                    "default": True,
                },
                "device": {
                    "type": "string",
                    "description": "Execution device: cpu, cuda, cuda:0, or GPU index string. Default: auto-select.",
                },
                "max_slices": {
                    "type": "integer",
                    "description": "Max number of slices processed explicitly for large 3D inputs. Use 0 to process every slice.",
                    "default": 160,
                    "minimum": 0,
                    "maximum": 2000,
                },
            },
            "required": ["file_path"],
        },
    },
}


SAM2_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "segment_video_sam2",
        "description": (
            "Track and segment objects in videos with SAM-family video models. "
            "Primary platform focus is 2D/3D scientific images; use this for video-specific workloads."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of video file paths to segment (supports mp4, avi, mov, etc.)",
                    "items": {"type": "string"},
                },
                "track_points": {
                    "type": "array",
                    "description": "List of [x, y] coordinate pairs to track. If not provided, uses center of video. Each point tracks one object.",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "default": None,
                },
                "save_visualizations": {
                    "type": "boolean",
                    "description": "Whether to save video with tracked mask overlay. Default: true",
                    "default": True,
                },
                "device": {
                    "type": "string",
                    "description": "Execution device: cpu, cuda, cuda:0, or GPU index string. Default: auto-select.",
                },
                "model_id": {
                    "type": "string",
                    "description": "Optional video model id. Falls back to facebook/sam2-hiera-large if unavailable.",
                },
            },
            "required": ["file_paths"],
        },
    },
}


SAM3_SEGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "segment_image_sam3",
        "description": (
            "Segment 2D/3D/4D scientific images using SAM3 with either automatic prompting "
            "(powerful defaults) or concept prompting (text and/or box prompts). "
            "Use preset for simple tuning; override advanced fields only when needed. "
            "If the user names object(s) to segment in ordinary language, concept_prompt may be inferred from the request context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image/volume file paths to segment.",
                    "items": {"type": "string"},
                },
                "model_id": {
                    "type": "string",
                    "description": "SAM3 model reference (local path recommended in offline deployments).",
                },
                "device": {
                    "type": "string",
                    "description": "Execution device: cpu, cuda, cuda:0, or GPU index string. Default: auto-select.",
                },
                "preset": {
                    "type": "string",
                    "description": "Simple quality/speed preset. Use fast for quick iteration, balanced for default, high_quality for best masks.",
                    "enum": ["fast", "balanced", "high_quality"],
                    "default": "balanced",
                },
                "concept_prompt": {
                    "type": "string",
                    "description": "Optional SAM3 concept text prompt (for example 'nuclei' or 'cell membrane'). Enables concept mode; when omitted, the backend can infer it from a natural-language segmentation request.",
                },
                "input_boxes": {
                    "type": "array",
                    "description": "Optional SAM3 concept boxes in absolute xyxy pixel coordinates. Enables concept mode.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "input_boxes_labels": {
                    "type": "array",
                    "description": "Optional labels for input_boxes (1=positive/include, 0=negative/exclude). Must match input_boxes length.",
                    "items": {"type": "integer"},
                },
                "threshold": {
                    "type": "number",
                    "description": "Concept-mode confidence threshold used by SAM3 post-processing.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "slice_index": {
                    "type": "integer",
                    "description": "Concept mode: optional Z-slice index for 3D/4D inputs (defaults to middle slice).",
                    "minimum": 0,
                },
                "max_slices": {
                    "type": "integer",
                    "description": "Max slices processed explicitly for large 3D/4D inputs.",
                    "default": 192,
                    "minimum": 1,
                    "maximum": 4096,
                },
                "window_size": {
                    "type": "integer",
                    "description": "Sliding-window size for large slices.",
                    "default": 1024,
                    "minimum": 64,
                    "maximum": 4096,
                },
                "window_overlap": {
                    "type": "number",
                    "description": "Sliding-window overlap ratio.",
                    "default": 0.25,
                    "minimum": 0.0,
                    "maximum": 0.9,
                },
                "min_points": {
                    "type": "integer",
                    "description": "Minimum auto-prompt points per window.",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 512,
                },
                "max_points": {
                    "type": "integer",
                    "description": "Maximum auto-prompt points per window.",
                    "default": 64,
                    "minimum": 1,
                    "maximum": 1024,
                },
                "point_density": {
                    "type": "number",
                    "description": "Point density factor used to adapt prompt count to foreground area.",
                    "default": 0.0015,
                    "minimum": 0.00001,
                    "maximum": 0.05,
                },
                "mask_threshold": {
                    "type": "number",
                    "description": "Binarization threshold applied to SAM3 mask logits/probabilities.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "vote_threshold": {
                    "type": "number",
                    "description": "Per-pixel vote ratio threshold used when merging overlapping windows.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "min_component_area_ratio": {
                    "type": "number",
                    "description": "Remove connected components smaller than this fraction of slice area.",
                    "default": 0.0001,
                    "minimum": 0.0,
                    "maximum": 0.2,
                },
                "modality_hint": {
                    "type": "string",
                    "description": "Optional preprocessing profile hint: auto, fluorescence, brightfield, ct_like, or generic.",
                    "default": "auto",
                },
                "preprocess": {
                    "type": "boolean",
                    "description": "Apply classical denoise/contrast preprocessing before prompting/inference.",
                    "default": True,
                },
                "refine_3d": {
                    "type": "boolean",
                    "description": "Apply 3D temporal/component consistency refinement for volumetric outputs.",
                    "default": True,
                },
                "save_visualizations": {
                    "type": "boolean",
                    "description": "Whether to save overlay visualizations.",
                    "default": True,
                },
                "fallback_to_medsam2": {
                    "type": "boolean",
                    "description": "Fallback to MedSAM2 if SAM3 cannot initialize.",
                    "default": True,
                },
                "force_rerun": {
                    "type": "boolean",
                    "description": "If true, bypass cached SAM3 results for identical inputs/parameters.",
                    "default": False,
                },
            },
            "required": ["file_paths"],
        },
    },
}


DEPTH_PRO_ESTIMATE_TOOL = {
    "type": "function",
    "function": {
        "name": "estimate_depth_pro",
        "description": (
            "Estimate monocular depth maps from 2D scientific images using DepthPro. "
            "Outputs normalized depth visualizations (depth map, overlay, side-by-side) and "
            "raw float depth arrays for follow-up analysis or segmentation workflows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image file paths to process.",
                    "items": {"type": "string"},
                },
                "model_id": {
                    "type": "string",
                    "description": "DepthPro model id/path (default from settings).",
                },
                "device": {
                    "type": "string",
                    "description": "Execution device: cpu, cuda, cuda:0, or GPU index string. Default: auto-select.",
                },
                "use_fov_model": {
                    "type": "boolean",
                    "description": "Enable DepthPro field-of-view head when supported by model weights.",
                    "default": True,
                },
                "save_visualizations": {
                    "type": "boolean",
                    "description": "Whether to save depth visualization images.",
                    "default": True,
                },
                "save_raw_depth": {
                    "type": "boolean",
                    "description": "Whether to save raw float32 depth maps as .npy.",
                    "default": True,
                },
                "force_rerun": {
                    "type": "boolean",
                    "description": "If true, bypass cached DepthPro results for identical inputs/parameters.",
                    "default": False,
                },
            },
            "required": ["file_paths"],
        },
    },
}


MEGASEG_SEGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "segment_image_megaseg",
        "description": (
            "Run Megaseg DynUNet inference on multichannel microscopy images. "
            "Use the structure channel for segmentation, optionally include a nucleus channel for context, "
            "and return binary mask artifacts, probability volumes, overlays, quantitative summaries, and an aggregate report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of microscopy image/volume paths to segment with Megaseg.",
                    "items": {"type": "string"},
                },
                "structure_channel": {
                    "type": "integer",
                    "description": (
                        "Structure channel number supplied to Megaseg. Defaults to channel 4 for "
                        "multichannel microscopy, but the tool auto-normalizes single-channel volumes."
                    ),
                    "default": 4,
                    "minimum": 0,
                },
                "nucleus_channel": {
                    "type": ["integer", "null"],
                    "description": (
                        "Optional nucleus channel number used for overlays and intensity context. "
                        "Defaults to channel 6 for multichannel microscopy; single-channel volumes "
                        "automatically omit the nucleus channel."
                    ),
                    "default": 6,
                    "minimum": 0,
                },
                "channel_index_base": {
                    "type": "integer",
                    "description": "Whether channel numbering is 1-based or 0-based.",
                    "enum": [0, 1],
                    "default": 1,
                },
                "mask_threshold": {
                    "type": "number",
                    "description": "Probability threshold used to convert Megaseg probabilities into a binary mask.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "save_visualizations": {
                    "type": "boolean",
                    "description": "Whether to save overlay previews (mid-Z and MIP). Default: true.",
                    "default": True,
                },
                "generate_report": {
                    "type": "boolean",
                    "description": "Whether to save an aggregate Markdown report and CSV summary. Default: true.",
                    "default": True,
                },
                "device": {
                    "type": "string",
                    "description": "Execution device override such as cpu, cuda, cuda:0, or mps. Default: auto.",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Optional Megaseg checkpoint override. Defaults to the configured local benchmark checkpoint.",
                },
                "structure_name": {
                    "type": "string",
                    "description": "Optional label used in technical summaries, for example 'NPM1 structure' or 'Golgi'.",
                },
            },
            "required": ["file_paths"],
        },
    },
}


SEGMENTATION_EVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_segmentation_masks",
        "description": (
            "Evaluate predicted binary segmentation masks against ground-truth masks "
            "using Dice/IoU/precision/recall. Supports .npy, NIfTI, TIFF, and common image formats."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prediction_paths": {
                    "type": "array",
                    "description": "List of predicted mask file paths.",
                    "items": {"type": "string"},
                },
                "ground_truth_paths": {
                    "type": "array",
                    "description": "List of ground-truth mask file paths.",
                    "items": {"type": "string"},
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for binarizing non-binary masks.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "match_by_stem": {
                    "type": "boolean",
                    "description": "Match prediction/ground-truth files by normalized filename stem when true.",
                    "default": True,
                },
                "pair_map": {
                    "type": "object",
                    "description": (
                        "Optional explicit prediction->ground-truth map. Keys can be prediction paths, basenames, or stems. "
                        "Values are ground-truth paths."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "stem_strip_tokens": {
                    "type": "array",
                    "description": "Optional list of trailing stem tokens stripped during stem matching (e.g., input,target,mask,label).",
                    "items": {"type": "string"},
                },
            },
            "required": ["prediction_paths", "ground_truth_paths"],
        },
    },
}


SEGMENT_EVALUATE_BATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "segment_evaluate_batch",
        "description": (
            "Run SAM3 segmentation followed by mask evaluation in one call. "
            "Use only when both source image_paths and ground-truth mask paths are available. "
            "Supports explicit image->ground-truth pairing and returns consolidated metrics/artifacts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "description": "Input image/volume paths to segment.",
                    "items": {"type": "string"},
                },
                "ground_truth_paths": {
                    "type": "array",
                    "description": "Ground-truth mask paths for evaluation.",
                    "items": {"type": "string"},
                },
                "pair_map": {
                    "type": "object",
                    "description": (
                        "Optional explicit image->ground-truth mapping. Keys can be image paths, basenames, or stems. "
                        "Values are ground-truth mask paths."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "save_visualizations": {"type": "boolean", "default": True},
                "model_id": {"type": "string"},
                "device": {"type": "string"},
                "threshold": {
                    "type": "number",
                    "description": "Threshold for binarizing non-binary masks during evaluation.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "match_by_stem": {
                    "type": "boolean",
                    "description": "When no pair_map is provided, pair using normalized filename stems.",
                    "default": True,
                },
                "stem_strip_tokens": {
                    "type": "array",
                    "description": "Optional list of trailing stem tokens stripped during stem matching.",
                    "items": {"type": "string"},
                },
                "force_rerun": {
                    "type": "boolean",
                    "description": "If true, bypass cached segmentation results and rerun segmentation.",
                    "default": False,
                },
            },
            "required": ["image_paths", "ground_truth_paths"],
        },
    },
}


YOLO_DETECT_TOOL = {
    "type": "function",
    "function": {
        "name": "yolo_detect",
        "description": (
            "Run YOLO object detection on one or more images. Returns box predictions and (optionally) overlay visualizations. "
            "Defaults to pretrained YOLO26x (best baseline, configurable via YOLO_DEFAULT_MODEL) unless the user explicitly requests a finetuned model. "
            "For prairie dog or burrow detection, use model_name='yolov5_rarespot' (or 'prairie'), which resolves to the active "
            "shared prairie checkpoint when available and otherwise falls back to RareSpotWeights.pt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image file paths",
                    "items": {"type": "string"},
                },
                "model_name": {
                    "type": "string",
                    "description": "Optional finetuned model name stored in data/models/yolo/finetuned (without .pt). Produced by yolo_finetune_detect. Use 'latest' to auto-select newest finetuned model.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Optional explicit path to a YOLO weights file (.pt). Overrides model_name.",
                },
                "use_latest_finetuned_if_available": {
                    "type": "boolean",
                    "description": "If true and no model is provided, auto-select the newest local finetuned checkpoint.",
                    "default": False,
                },
                "conf": {"type": "number", "default": 0.25, "minimum": 0.0, "maximum": 1.0},
                "iou": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 1.0},
                "include_stability_audit": {
                    "type": "boolean",
                    "description": (
                        "If true, run a post-inference prediction-stability audit on the same images. "
                        "Useful for uncertainty reporting, active learning, or manual-review prioritization."
                    ),
                    "default": False,
                },
                "stability_top_k": {
                    "type": "integer",
                    "description": "Maximum number of highest-priority review candidates to surface from the stability audit.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 20,
                },
                "stability_preservation_ratio": {
                    "type": "number",
                    "description": "Central spectral radius preserved during the optional stability audit.",
                    "default": 0.9,
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "save_visualizations": {"type": "boolean", "default": True},
            },
            "required": ["file_paths"],
        },
    },
}

PREDICTION_STABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_prediction_stability",
        "description": (
            "Audit one or more images for prediction stability and review priority. "
            "Returns a ranked list of images whose detections are most fragile under a controlled perturbation, "
            "along with explanatory figures and review candidates. "
            "The current backend uses spectral feature perturbation with the RareSpot prairie detector."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image file paths to analyze.",
                    "items": {"type": "string"},
                },
                "model_name": {
                    "type": "string",
                    "description": "Optional detector alias. Defaults to the active prairie/RareSpot detector.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Optional explicit YOLO weights path (.pt). Overrides model_name.",
                },
                "method": {
                    "type": "string",
                    "description": "Stability-analysis backend. 'auto' currently resolves to the spectral feature-perturbation method.",
                    "default": "auto",
                    "enum": ["auto", "spectral"],
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of review candidates to highlight.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "preservation_ratio": {
                    "type": "number",
                    "description": "Central spectral radius to preserve in the perturbed feature map.",
                    "default": 0.9,
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "conf": {
                    "type": "number",
                    "description": "Confidence threshold for comparing original and perturbed detections.",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "iou": {
                    "type": "number",
                    "description": "IoU threshold for the internal NMS comparison.",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "imgsz": {
                    "type": "integer",
                    "description": "Inference image size.",
                    "default": 640,
                    "minimum": 64,
                    "maximum": 2048,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size for scoring.",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 64,
                },
            },
            "required": ["file_paths"],
        },
    },
}

SPECTRAL_INSTABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "score_spectral_instability",
        "description": (
            "Score one or more images by detector instability under spectral feature filtering using the RareSpot prairie detector. "
            "This is a standalone active-learning signal: higher scores indicate images whose predictions change more when low-level "
            "spectral content is perturbed, which can help prioritize ambiguous or fragile examples for review."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image file paths to score.",
                    "items": {"type": "string"},
                },
                "model_name": {
                    "type": "string",
                    "description": "Optional detector alias. Defaults to the active prairie/RareSpot detector.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Optional explicit YOLO weights path (.pt). Overrides model_name.",
                },
                "preservation_ratio": {
                    "type": "number",
                    "description": "Central spectral radius to preserve in the perturbed feature map.",
                    "default": 0.9,
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "conf": {
                    "type": "number",
                    "description": "Confidence threshold for comparing original and spectrally filtered detections.",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "iou": {
                    "type": "number",
                    "description": "IoU threshold for the internal NMS pass used by the scorer.",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "imgsz": {
                    "type": "integer",
                    "description": "Inference image size.",
                    "default": 640,
                    "minimum": 64,
                    "maximum": 2048,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size for scoring.",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 64,
                },
            },
            "required": ["file_paths"],
        },
    },
}


YOLO_LIST_MODELS_TOOL = {
    "type": "function",
    "function": {
        "name": "yolo_list_finetuned_models",
        "description": "List local finetuned YOLO models saved under data/models/yolo/finetuned and return the newest model.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of models to return (newest first).",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                }
            },
            "required": [],
        },
    },
}


YOLO_FINETUNE_DETECT_TOOL = {
    "type": "function",
    "function": {
        "name": "yolo_finetune_detect",
        "description": (
            "Finetune a pretrained YOLO26 detection model on a small custom dataset. "
            "Provide image files and matching YOLO-format label .txt files (same base filename). "
            "The tool will organize a temporary COCO-style dataset folder under data/yolo/datasets/ "
            "and save finetuned weights under data/models/yolo/finetuned/."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of image paths and YOLO label .txt paths (can be mixed). Directories are allowed.",
                    "items": {"type": "string"},
                },
                "image_paths": {
                    "type": "array",
                    "description": "Optional explicit list of image file paths (alternative to file_paths).",
                    "items": {"type": "string"},
                },
                "label_paths": {
                    "type": "array",
                    "description": "Optional explicit list of YOLO label .txt paths (alternative to file_paths).",
                    "items": {"type": "string"},
                },
                "class_names": {
                    "type": "array",
                    "description": "Optional class names in index order (e.g., ['cat','dog']). If omitted, uses class0..",
                    "items": {"type": "string"},
                },
                "base_model": {
                    "type": "string",
                    "description": "Pretrained YOLO weights to start from (recommended: yolo26x.pt).",
                    "default": "yolo26x.pt",
                },
                "epochs": {"type": "integer", "default": 10, "minimum": 1, "maximum": 300},
                "imgsz": {"type": "integer", "default": 640, "minimum": 32, "maximum": 2048},
                "batch": {"type": "integer", "default": 4, "minimum": 1, "maximum": 256},
                "val_split": {"type": "number", "default": 0.2, "minimum": 0.05, "maximum": 0.5},
                "seed": {"type": "integer", "default": 42},
                "device": {
                    "type": "string",
                    "description": "Optional device (e.g., 'cpu' or '0').",
                },
                "prepare_only": {"type": "boolean", "default": False},
            },
            "required": [],
        },
    },
}


VISION_TOOL_SCHEMAS = [
    BIOIO_LOAD_IMAGE_TOOL,
    SAM2_SEGMENT_TOOL,
    SAM2_PROMPT_TOOL,
    SAM2_VIDEO_TOOL,
    SAM3_SEGMENT_TOOL,
    MEGASEG_SEGMENT_TOOL,
    DEPTH_PRO_ESTIMATE_TOOL,
    SEGMENTATION_EVAL_TOOL,
    SEGMENT_EVALUATE_BATCH_TOOL,
    YOLO_LIST_MODELS_TOOL,
    YOLO_DETECT_TOOL,
    PREDICTION_STABILITY_TOOL,
    SPECTRAL_INSTABILITY_TOOL,
    YOLO_FINETUNE_DETECT_TOOL,
]


__all__ = [
    "BIOIO_LOAD_IMAGE_TOOL",
    "SAM2_SEGMENT_TOOL",
    "SAM2_PROMPT_TOOL",
    "SAM2_VIDEO_TOOL",
    "SAM3_SEGMENT_TOOL",
    "MEGASEG_SEGMENT_TOOL",
    "DEPTH_PRO_ESTIMATE_TOOL",
    "SEGMENTATION_EVAL_TOOL",
    "SEGMENT_EVALUATE_BATCH_TOOL",
    "YOLO_DETECT_TOOL",
    "PREDICTION_STABILITY_TOOL",
    "SPECTRAL_INSTABILITY_TOOL",
    "YOLO_LIST_MODELS_TOOL",
    "YOLO_FINETUNE_DETECT_TOOL",
    "VISION_TOOL_SCHEMAS",
]
