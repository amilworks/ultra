"""Tool schemas for LLM-authored Python code execution."""

NUMPY_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "numpy_calculator",
        "description": (
            "Evaluate a deterministic NumPy-backed numeric expression for arithmetic, trig, linear algebra, "
            "sums, and array math. Use this when you already know the formula and want accurate computation "
            "without launching a full Python job."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A NumPy-style numeric expression. Available names include sin, cos, tan, deg2rad, "
                        "sqrt, log, exp, sum, mean, dot, matmul, array, arange, linspace, pi, and e."
                    ),
                },
                "variables": {
                    "type": "object",
                    "description": ("Optional scalar or array variables used by the expression."),
                    "additionalProperties": True,
                },
            },
            "required": ["expression"],
        },
    },
}

CODEGEN_PYTHON_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "codegen_python_plan",
        "description": (
            "Generate or repair a Python job package for a scientific/data-analysis task. "
            "Returns a persisted job_id with code files, command, dependencies, and expected outputs. "
            "Use this before execute_python_job."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_summary": {
                    "type": "string",
                    "description": "Clear description of the code task to implement.",
                },
                "job_id": {
                    "type": "string",
                    "description": (
                        "Optional existing job_id to update/repair in place instead of creating a new job."
                    ),
                },
                "inputs": {
                    "type": "array",
                    "description": "Optional known input files/artifacts for this job.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "constraints": {
                    "type": "object",
                    "description": (
                        "Optional execution constraints such as timeout, memory, or expected output format."
                    ),
                    "additionalProperties": True,
                },
                "attempt_index": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Current repair attempt index.",
                    "default": 1,
                },
                "previous_failure": {
                    "type": "object",
                    "description": "Optional failure payload from a previous execute_python_job attempt.",
                    "additionalProperties": True,
                },
            },
            "required": ["task_summary"],
        },
    },
}


EXECUTE_PYTHON_JOB_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python_job",
        "description": (
            "Execute a generated Python job in a sandboxed runtime. "
            "Can run durably (continues if client disconnects) and returns logs, artifacts, and error class."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job id returned by codegen_python_plan.",
                },
                "execution_backend": {
                    "type": "string",
                    "enum": ["docker"],
                    "description": "Execution backend. The production release uses docker directly.",
                    "default": "docker",
                },
                "durable_execution": {
                    "type": "boolean",
                    "description": "Submit execution as a durable run before polling for completion.",
                    "default": True,
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "description": (
                        "When true, block until execution reaches terminal status or wait timeout."
                    ),
                    "default": True,
                },
                "wait_timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 86400,
                    "description": "Max time to wait when wait_for_completion=true.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 86400,
                    "description": "Execution timeout for this attempt.",
                },
                "cpu_limit": {
                    "oneOf": [{"type": "number"}, {"type": "string"}],
                    "description": "Docker CPU limit (for example 2 or 0.5).",
                },
                "memory_mb": {
                    "type": "integer",
                    "minimum": 128,
                    "maximum": 262144,
                    "description": "Docker memory limit in MB.",
                },
                "auto_repair": {
                    "type": "boolean",
                    "description": (
                        "Automatically regenerate and retry failed jobs using previous failure payloads."
                    ),
                    "default": True,
                },
                "max_repair_cycles": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                    "description": (
                        "Optional override for maximum code repair cycles during this execution call."
                    ),
                },
            },
            "required": ["job_id"],
        },
    },
}


CODE_EXECUTION_TOOL_SCHEMAS = [
    NUMPY_CALCULATOR_TOOL,
    CODEGEN_PYTHON_PLAN_TOOL,
    EXECUTE_PYTHON_JOB_TOOL,
]


__all__ = [
    "CODEGEN_PYTHON_PLAN_TOOL",
    "CODE_EXECUTION_TOOL_SCHEMAS",
    "EXECUTE_PYTHON_JOB_TOOL",
    "NUMPY_CALCULATOR_TOOL",
]
