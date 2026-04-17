"""BisQue domain tool schemas."""


def _named_value_array_schema(*, description: str, value_description: str) -> dict:
    return {
        "type": "array",
        "description": description,
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entry name",
                },
                "value": {
                    "type": "string",
                    "description": value_description,
                },
            },
            "required": ["name", "value"],
        },
    }


def _named_values_array_schema(*, description: str, values_description: str) -> dict:
    return {
        "type": "array",
        "description": description,
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entry name",
                },
                "values": {
                    "type": "array",
                    "description": values_description,
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "values"],
        },
    }


BISQUE_UPLOAD_TOOL = {
    "type": "function",
    "function": {
        "name": "upload_to_bisque",
        "description": (
            "Upload one or more files (images, biological data formats) to the BisQue image repository. "
            "Optionally place the uploaded resources into an existing dataset by URI or by plain dataset name, "
            "or create that dataset if it does not exist. Supports 100+ biological formats including DICOM, "
            "NIFTI, TIFF, PNG, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "List of absolute or relative file paths to upload to BisQue",
                    "items": {"type": "string"},
                },
                "dataset_uri": {
                    "type": "string",
                    "description": "Optional destination dataset URI or resource_uniq for organizing the uploaded resources.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": (
                        "Optional destination dataset name. Use this when the user names a dataset in natural language "
                        "without providing a URI."
                    ),
                },
                "create_dataset_if_missing": {
                    "type": "boolean",
                    "description": (
                        "If true and dataset_name or dataset_uri does not resolve to an existing dataset, "
                        "create a new dataset with that name and place the uploaded resources into it."
                    ),
                    "default": False,
                },
                "dataset_tags": {
                    **_named_value_array_schema(
                        description="Optional key-value tags to add when a new dataset is created from this upload.",
                        value_description="Tag value.",
                    ),
                },
            },
            "required": ["file_paths"],
        },
    },
}


BISQUE_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_ping",
        "description": "Check BisQue connectivity and credentials.",
        "parameters": {"type": "object", "properties": {}},
    },
}


BISQUE_DOWNLOAD_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_download_resource",
        "description": "Download a BisQue resource blob to a local file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "BisQue resource URI, resource_uniq, or view URL",
                },
                "output_path": {
                    "type": "string",
                    "description": "Local path to save the downloaded blob",
                },
            },
            "required": ["resource_uri", "output_path"],
        },
    },
}


BISQUE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_bisque_resources",
        "description": "Search for resources (images, datasets, files, tables) in the BisQue repository. Use this first for simple existence/listing questions. Note: videos are often stored as image resources with file extensions (e.g., .mp4). HDF5, DREAM3D, and similar BisQue table assets can use resource_type 'table' or aliases such as 'hdf5'/'dream3d'; the backend will normalize common file-type cues automatically. Supports BisQue tag_query syntax (e.g., image_num_z:>=160, @created:>=2024-01-01, antibody:*GFP*), plus structured tag/metadata filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "description": "Type of resource to search for (image, dataset, file, table, etc.). HDF5/DREAM3D aliases such as hdf5, h5, and dream3d are accepted.",
                    "default": "image",
                },
                "tag_query": {
                    "type": "string",
                    "description": "BisQue tag_query string. Attributes require '@' (e.g., @created:>=2024-01-01).",
                },
                "tag_filters": {
                    **_named_values_array_schema(
                        description=(
                            "Structured tag filters. Each entry uses {name, values}, where "
                            "values contains one or more comparison tokens such as '>=160' or '<=5'."
                        ),
                        values_description="One or more tag filter values or comparison tokens.",
                    ),
                },
                "metadata_filters": {
                    **_named_values_array_schema(
                        description=(
                            "Convenience metadata filters. Aliases: z/num_z/depth -> image_num_z, "
                            "t/timepoints -> image_num_t, c/channels -> image_num_c, "
                            "x/width -> image_num_x, y/height -> image_num_y."
                        ),
                        values_description="One or more metadata filter values or comparison tokens.",
                    ),
                },
                "text": {
                    "type": "string",
                    "description": "Free-text search term; expands to fuzzy tag query (e.g., '*term*').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of results to skip for pagination (default: 0)",
                    "default": 0,
                },
            },
            "required": ["resource_type"],
        },
    },
}


BISQUE_FIND_ASSETS_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_find_assets",
        "description": (
            "Composite BisQue tool that chains search -> metadata -> download. "
            "Use after search_bisque_resources when you need metadata or downloads, for example "
            "'find the HDF5 table and show metadata' or 'download the first result'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "default": "image",
                    "description": "Resource type to search (image, dataset, file, table). HDF5/DREAM3D aliases are accepted and normalized to table.",
                },
                "tag_query": {"type": "string"},
                "tag_filters": _named_values_array_schema(
                    description=(
                        "Structured tag filters. Each entry uses {name, values}, where values "
                        "contains one or more desired tag values or comparison tokens."
                    ),
                    values_description="One or more tag filter values or comparison tokens.",
                ),
                "metadata_filters": _named_values_array_schema(
                    description=(
                        "Convenience metadata filters using BisQue metadata names or aliases "
                        "such as z, t, c, x, and y."
                    ),
                    values_description="One or more metadata filter values or comparison tokens.",
                ),
                "text": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "offset": {"type": "integer", "default": 0},
                "include_metadata": {"type": "boolean", "default": True},
                "max_metadata": {"type": "integer", "default": 3},
                "download": {"type": "boolean", "default": False},
                "download_dir": {"type": "string", "default": "data/bisque_downloads"},
            },
        },
    },
}


BISQUE_LOAD_TOOL = {
    "type": "function",
    "function": {
        "name": "load_bisque_resource",
        "description": "Load detailed information about a specific BisQue resource by its URI. Returns metadata including tags, dimensions, owner, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "Full URI of the resource to load",
                },
                "view": {
                    "type": "string",
                    "description": "Level of detail to load (short, full, deep). Default: deep",
                    "default": "deep",
                },
            },
            "required": ["resource_uri"],
        },
    },
}


BISQUE_DELETE_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_bisque_resource",
        "description": "Delete a resource from BisQue. WARNING: This action cannot be undone. Use with caution.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "Full URI of the resource to delete",
                }
            },
            "required": ["resource_uri"],
        },
    },
}


BISQUE_TAG_TOOL = {
    "type": "function",
    "function": {
        "name": "add_tags_to_resource",
        "description": "Add metadata tags to a BisQue resource. Tags are key-value pairs used for organization and searchability.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "Full URI of the resource to tag",
                },
                "tags": {
                    "type": "array",
                    "description": "List of tags to add, each with 'name' and 'value'",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["name", "value"],
                    },
                },
            },
            "required": ["resource_uri", "tags"],
        },
    },
}


BISQUE_FETCH_XML_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_fetch_xml",
        "description": "Fetch raw XML from a BisQue resource or endpoint. Useful for debugging and deep inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "BisQue resource URI, resource_uniq, or path (e.g., /data_service/image/...)",
                },
                "view": {
                    "type": "string",
                    "description": "View level (short, full, deep). Default: deep",
                    "default": "deep",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional local path to save the XML",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return inline (preview). Default: 20000",
                    "default": 20000,
                },
            },
            "required": ["resource_uri"],
        },
    },
}


BISQUE_DOWNLOAD_DATASET_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_download_dataset",
        "description": "Download all images in a BisQue dataset to a local directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_uri": {
                    "type": "string",
                    "description": "Dataset URI or resource_uniq",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Local directory to save downloaded dataset members",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional max number of members to download",
                },
                "use_localpath": {
                    "type": "boolean",
                    "description": "If running on BisQue host, use local paths for faster access",
                    "default": False,
                },
            },
            "required": ["dataset_uri", "output_dir"],
        },
    },
}


BISQUE_CREATE_DATASET_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_create_dataset",
        "description": "Create a BisQue dataset from a list of resource URIs (e.g., images) for training or inference.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset name",
                },
                "resource_uris": {
                    "type": "array",
                    "description": "List of BisQue resource URIs or resource_uniq values",
                    "items": {"type": "string"},
                },
                "tags": {
                    **_named_value_array_schema(
                        description="Optional key-value tags to add to the dataset.",
                        value_description="Tag value.",
                    ),
                },
            },
            "required": ["name", "resource_uris"],
        },
    },
}


BISQUE_ADD_TO_DATASET_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_add_to_dataset",
        "description": (
            "Add existing BisQue resources to an existing dataset by URI or dataset name. "
            "Can also create the dataset if it does not exist yet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uris": {
                    "type": "array",
                    "description": "List of BisQue resource URIs or resource_uniq values to organize into a dataset.",
                    "items": {"type": "string"},
                },
                "dataset_uri": {
                    "type": "string",
                    "description": "Optional existing dataset URI or resource_uniq.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Optional existing dataset name to resolve by search before appending resources.",
                },
                "create_dataset_if_missing": {
                    "type": "boolean",
                    "description": (
                        "If true and the dataset cannot be found, create a new dataset using dataset_name "
                        "and add the resources to it."
                    ),
                    "default": False,
                },
                "tags": {
                    **_named_value_array_schema(
                        description="Optional key-value tags to add when creating a new dataset.",
                        value_description="Tag value.",
                    ),
                },
            },
            "required": ["resource_uris"],
        },
    },
}


BISQUE_ADD_GOBJECTS_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_add_gobjects",
        "description": "Add graphical annotations (gobjects) such as points, polygons, rectangles, and circles to a BisQue resource.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_uri": {
                    "type": "string",
                    "description": "Resource URI to annotate",
                },
                "gobjects": {
                    "type": "array",
                    "description": "List of gobject specs (type, vertices, tags)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                            "vertices": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "x": {"type": "number"},
                                                "y": {"type": "number"},
                                                "z": {"type": "number"},
                                                "t": {"type": "number"},
                                            },
                                        },
                                    ]
                                },
                            },
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "value": {"type": "string"},
                                    },
                                    "required": ["name", "value"],
                                },
                            },
                        },
                        "required": ["type", "vertices"],
                    },
                },
                "replace_existing": {
                    "type": "boolean",
                    "description": "Remove existing gobjects before adding new ones",
                    "default": False,
                },
            },
            "required": ["resource_uri", "gobjects"],
        },
    },
}


BISQUE_ADVANCED_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "bisque_advanced_search",
        "description": "Advanced BisQue search with tag filters, metadata aliases, ownership, permissions, date ranges, and ordering.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "default": "image"},
                "tag_query": {
                    "type": "string",
                    "description": "BisQue tag_query (e.g., image_num_z:>=160, antibody:*GFP*). Attributes require '@' (e.g., @created:>=2024-01-01).",
                },
                "tag_filters": {
                    **_named_values_array_schema(
                        description=(
                            "Key-value tags to match. Each entry uses {name, values}; multiple "
                            "values in one entry act like OR for that tag."
                        ),
                        values_description="One or more tag values or comparison tokens.",
                    ),
                },
                "metadata_filters": {
                    **_named_values_array_schema(
                        description=(
                            "Convenience metadata filters. Aliases: z/num_z/depth -> image_num_z, "
                            "t/timepoints -> image_num_t, c/channels -> image_num_c, "
                            "x/width -> image_num_x, y/height -> image_num_y."
                        ),
                        values_description="One or more metadata filter values or comparison tokens.",
                    ),
                },
                "text": {"type": "string", "description": "Free-text search term (fuzzy)."},
                "owner": {"type": "string"},
                "permission": {"type": "string"},
                "created_after": {"type": "string"},
                "created_before": {"type": "string"},
                "modified_after": {"type": "string"},
                "modified_before": {"type": "string"},
                "order_by": {"type": "string"},
                "order": {"type": "string", "default": "desc"},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
                "view": {"type": "string", "default": "short"},
            },
        },
    },
}


BISQUE_RUN_MODULE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_bisque_module",
        "description": "Execute a BisQue module (for example EdgeDetection) using existing BisQue resources or uploaded files. Automatically uploads local inputs, submits module execution, polls MEX status, and can download output artifacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": "Name of the BisQue module to execute (e.g., 'EdgeDetection', 'CellSegment3D')",
                },
                "input_resources": {
                    **_named_value_array_schema(
                        description=(
                            "Module input bindings. Each entry uses {name, value}, where value "
                            "can be a BisQue URI, client view URL, or uploaded filename."
                        ),
                        value_description="BisQue URI, client view URL, or uploaded filename.",
                    ),
                },
                "module_params": {
                    **_named_value_array_schema(
                        description="Optional additional module parameters as key-value pairs.",
                        value_description="Module parameter value.",
                    ),
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "description": "If true (default), poll MEX until terminal status and return module outputs.",
                    "default": True,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum seconds to wait for module completion when wait_for_completion is true.",
                    "default": 600,
                    "minimum": 5,
                },
                "poll_interval_seconds": {
                    "type": "number",
                    "description": "Polling interval in seconds while monitoring MEX status.",
                    "default": 2.0,
                    "minimum": 0.25,
                },
                "download_output": {
                    "type": "boolean",
                    "description": "If true (default), attempt to download module output resource locally.",
                    "default": True,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional local directory for downloaded module outputs.",
                },
                "bisque_user": {
                    "type": "string",
                    "description": "BisQue username (uses environment variable if not provided)",
                },
                "bisque_password": {
                    "type": "string",
                    "description": "BisQue password (uses environment variable if not provided)",
                },
                "bisque_root": {
                    "type": "string",
                    "description": "BisQue server URL (uses environment variable if not provided)",
                },
            },
            "required": ["module_name", "input_resources"],
        },
    },
}


BISQUE_TOOL_SCHEMAS = [
    BISQUE_FIND_ASSETS_TOOL,
    BISQUE_UPLOAD_TOOL,
    BISQUE_PING_TOOL,
    BISQUE_DOWNLOAD_TOOL,
    BISQUE_SEARCH_TOOL,
    BISQUE_LOAD_TOOL,
    BISQUE_DELETE_TOOL,
    BISQUE_TAG_TOOL,
    BISQUE_FETCH_XML_TOOL,
    BISQUE_DOWNLOAD_DATASET_TOOL,
    BISQUE_CREATE_DATASET_TOOL,
    BISQUE_ADD_TO_DATASET_TOOL,
    BISQUE_ADD_GOBJECTS_TOOL,
    BISQUE_ADVANCED_SEARCH_TOOL,
    BISQUE_RUN_MODULE_TOOL,
]


__all__ = [
    "BISQUE_ADD_GOBJECTS_TOOL",
    "BISQUE_ADD_TO_DATASET_TOOL",
    "BISQUE_ADVANCED_SEARCH_TOOL",
    "BISQUE_CREATE_DATASET_TOOL",
    "BISQUE_DELETE_TOOL",
    "BISQUE_DOWNLOAD_DATASET_TOOL",
    "BISQUE_DOWNLOAD_TOOL",
    "BISQUE_FETCH_XML_TOOL",
    "BISQUE_FIND_ASSETS_TOOL",
    "BISQUE_LOAD_TOOL",
    "BISQUE_PING_TOOL",
    "BISQUE_RUN_MODULE_TOOL",
    "BISQUE_SEARCH_TOOL",
    "BISQUE_TAG_TOOL",
    "BISQUE_TOOL_SCHEMAS",
    "BISQUE_UPLOAD_TOOL",
]
