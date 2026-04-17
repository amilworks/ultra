import json
from pathlib import Path
from types import SimpleNamespace

from src import tools
from src.science import imaging
from src.training import adapters


def test_models_root_prefers_explicit_yolo_model_root(monkeypatch, tmp_path):
    root = tmp_path / "models" / "yolo"
    monkeypatch.setenv("YOLO_MODEL_ROOT", str(root))
    monkeypatch.delenv("YOLO_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=""),
    )

    resolved = Path(tools._models_root())

    assert resolved == root
    assert resolved.is_dir()


def test_models_root_uses_default_model_parent(monkeypatch, tmp_path):
    model_path = tmp_path / "models" / "yolo" / "yolo26x.pt"
    monkeypatch.delenv("YOLO_MODEL_ROOT", raising=False)
    monkeypatch.setenv("YOLO_DEFAULT_MODEL", str(model_path))
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=""),
    )

    resolved = Path(tools._models_root())

    assert resolved == model_path.parent
    assert resolved.is_dir()


def test_models_root_falls_back_to_rarespot_parent(monkeypatch, tmp_path):
    weights_path = tmp_path / "models" / "yolo" / "RareSpotWeights.pt"
    monkeypatch.delenv("YOLO_MODEL_ROOT", raising=False)
    monkeypatch.delenv("YOLO_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=str(weights_path)),
    )

    resolved = Path(tools._models_root())

    assert resolved == weights_path.parent
    assert resolved.is_dir()


def test_finetuned_dir_uses_shared_science_root_when_legacy_missing(monkeypatch, tmp_path):
    science_root = tmp_path / "science"
    model_path = tmp_path / "readonly-models" / "yolo" / "yolo26x.pt"
    monkeypatch.delenv("YOLO_MODEL_ROOT", raising=False)
    monkeypatch.setenv("YOLO_DEFAULT_MODEL", str(model_path))
    monkeypatch.setenv("SCIENCE_DATA_ROOT", str(science_root))
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(
            prairie_rarespot_weights_path="",
            science_data_root=str(science_root),
        ),
    )

    resolved = Path(tools._finetuned_dir())

    assert resolved == science_root / "yolo" / "models" / "finetuned"
    assert resolved.is_dir()


def test_resolve_yolov5_repo_path_is_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOLOV5_RUNTIME_PATH", raising=False)

    resolved = adapters._resolve_yolov5_repo_path()

    assert resolved == Path(adapters.__file__).resolve().parents[2] / "third_party" / "yolov5"


def test_resolve_yolov5_repo_path_treats_relative_env_as_package_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOLOV5_RUNTIME_PATH", "third_party/yolov5")

    resolved = adapters._resolve_yolov5_repo_path()

    assert resolved == Path(adapters.__file__).resolve().parents[2] / "third_party" / "yolov5"


def test_expand_file_inputs_preserves_ome_zarr_store(tmp_path):
    store = tmp_path / "sample.ome.zarr"
    (store / "0").mkdir(parents=True)
    (store / ".zgroup").write_text("{}", encoding="utf-8")
    (store / "0" / ".zarray").write_text("{}", encoding="utf-8")

    expanded = tools._expand_file_inputs([str(store)])

    assert expanded == [store]


def test_segment_image_megaseg_accepts_remote_s3_ome_zarr(monkeypatch, tmp_path):
    runner_script = tmp_path / "megaseg_runner.py"
    checkpoint = tmp_path / "epoch_650.ckpt"
    runner_script.write_text("# runner stub\n", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    captured_request: dict[str, object] = {}

    def fake_science_output_root(*parts: str) -> str:
        root = tmp_path / "science"
        target = root.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def fake_run(cmd, **_kwargs):
        request_path = Path(str(cmd[-1]))
        captured_request.update(json.loads(request_path.read_text(encoding="utf-8")))
        payload = {
            "files": [
                {
                    "file": "example.ome.zarr",
                    "success": True,
                    "mask_path": str(tmp_path / "mask.tiff"),
                    "probability_path": str(tmp_path / "probability.tiff"),
                    "visualizations": [],
                    "segmentation": {
                        "coverage_percent": 1.25,
                        "object_count": 3,
                        "active_slice_count": 2,
                        "largest_component_voxels": 42,
                    },
                    "intensity_context": {},
                    "technical_summary": "ok",
                }
            ],
            "aggregate": {
                "mean_coverage_percent": 1.25,
                "median_coverage_percent": 1.25,
                "mean_object_count": 3.0,
                "median_object_count": 3.0,
            },
            "summary_csv_path": str(tmp_path / "summary.csv"),
            "report_path": str(tmp_path / "report.md"),
            "warnings": [],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(tools, "_resolve_megaseg_runner_script", lambda: runner_script)
    monkeypatch.setattr(tools, "_resolve_megaseg_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(
        tools, "_resolve_megaseg_checkpoint_path", lambda _explicit=None: str(checkpoint)
    )
    monkeypatch.setattr(tools, "_science_output_root", fake_science_output_root)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools.segment_image_megaseg(
        file_paths=["s3://allencell/aics/example.ome.zarr/"],
        save_visualizations=False,
        generate_report=False,
    )

    assert result["success"] is True
    assert captured_request["file_paths"] == ["s3://allencell/aics/example.ome.zarr/"]


def test_segment_image_megaseg_uses_settings_backed_runtime_paths(monkeypatch, tmp_path):
    runner_script = tmp_path / "megaseg_runner.py"
    checkpoint = tmp_path / "configured.ckpt"
    runtime_python = tmp_path / "megaseg-python"
    input_file = tmp_path / "NPM1_13054_IM.tiff"
    runner_script.write_text("# runner stub\n", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    runtime_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    input_file.write_text("tiff", encoding="utf-8")
    captured_request: dict[str, object] = {}

    def fake_science_output_root(*parts: str) -> str:
        root = tmp_path / "science"
        target = root.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def fake_run(cmd, **_kwargs):
        captured_request["cmd"] = [str(item) for item in cmd]
        request_path = Path(str(cmd[-1]))
        captured_request["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        payload = {
            "files": [
                {
                    "file": str(input_file),
                    "success": True,
                    "mask_path": str(tmp_path / "mask.tiff"),
                    "probability_path": str(tmp_path / "probability.tiff"),
                    "visualizations": [],
                    "segmentation": {
                        "coverage_percent": 2.5,
                        "object_count": 4,
                        "active_slice_count": 3,
                        "largest_component_voxels": 84,
                    },
                    "intensity_context": {},
                    "technical_summary": "ok",
                }
            ],
            "aggregate": {
                "mean_coverage_percent": 2.5,
                "median_coverage_percent": 2.5,
                "mean_object_count": 4.0,
                "median_object_count": 4.0,
            },
            "summary_csv_path": str(tmp_path / "summary.csv"),
            "report_path": str(tmp_path / "report.md"),
            "warnings": [],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.delenv("MEGASEG_PYTHON", raising=False)
    monkeypatch.delenv("CYTODL_PYTHON", raising=False)
    monkeypatch.delenv("MEGASEG_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("MEGASEG_BENCHMARK_ROOT", raising=False)
    monkeypatch.setattr(tools, "_resolve_megaseg_runner_script", lambda: runner_script)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(
            resolved_megaseg_python=str(runtime_python),
            resolved_megaseg_checkpoint_path=str(checkpoint),
            resolved_megaseg_benchmark_root=None,
        ),
    )
    monkeypatch.setattr(tools, "_MEGASEG_DEFAULT_CHECKPOINT", tmp_path / "missing-default.ckpt")
    monkeypatch.setattr(tools, "_MEGASEG_DEFAULT_ALIAS_CHECKPOINT", tmp_path / "missing-alias.ckpt")
    monkeypatch.setattr(tools, "_science_output_root", fake_science_output_root)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools.segment_image_megaseg(
        file_paths=[str(input_file)],
        save_visualizations=False,
        generate_report=False,
    )

    assert result["success"] is True
    assert captured_request["cmd"][:2] == [str(runtime_python), str(runner_script)]
    assert captured_request["request"]["checkpoint_path"] == str(checkpoint)


def test_segment_image_megaseg_auto_adjusts_single_channel_inputs(monkeypatch, tmp_path):
    runner_script = tmp_path / "megaseg_runner.py"
    checkpoint = tmp_path / "configured.ckpt"
    runtime_python = tmp_path / "megaseg-python"
    input_file = tmp_path / "NPM1_13054_IM.tiff"
    runner_script.write_text("# runner stub\n", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    runtime_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    input_file.write_text("tiff", encoding="utf-8")
    captured_request: dict[str, object] = {}

    def fake_science_output_root(*parts: str) -> str:
        root = tmp_path / "science"
        target = root.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def fake_run(cmd, **_kwargs):
        request_path = Path(str(cmd[-1]))
        captured_request["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        payload = {
            "files": [
                {
                    "file": str(input_file),
                    "success": True,
                    "mask_path": str(tmp_path / "mask.tiff"),
                    "probability_path": str(tmp_path / "probability.tiff"),
                    "visualizations": [],
                    "segmentation": {
                        "coverage_percent": 2.5,
                        "object_count": 4,
                        "active_slice_count": 3,
                        "largest_component_voxels": 84,
                    },
                    "intensity_context": {},
                    "technical_summary": "ok",
                }
            ],
            "aggregate": {
                "mean_coverage_percent": 2.5,
                "median_coverage_percent": 2.5,
                "mean_object_count": 4.0,
                "median_object_count": 4.0,
            },
            "summary_csv_path": str(tmp_path / "summary.csv"),
            "report_path": str(tmp_path / "report.md"),
            "warnings": [],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(tools, "_resolve_megaseg_runner_script", lambda: runner_script)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(
            resolved_megaseg_python=str(runtime_python),
            resolved_megaseg_checkpoint_path=str(checkpoint),
            resolved_megaseg_benchmark_root=None,
        ),
    )
    monkeypatch.setattr(
        tools,
        "load_scientific_image",
        lambda **_kwargs: {
            "success": True,
            "axis_sizes": {"C": 1, "Z": 65, "Y": 624, "X": 924},
            "is_multichannel": False,
        },
    )
    monkeypatch.setattr(tools, "_science_output_root", fake_science_output_root)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools.segment_image_megaseg(
        file_paths=[str(input_file)],
        save_visualizations=False,
        generate_report=False,
    )

    assert result["success"] is True
    assert captured_request["request"]["structure_channel"] == 1
    assert captured_request["request"]["nucleus_channel"] is None


def test_segment_image_megaseg_uses_remote_service_and_downloads_artifacts(monkeypatch, tmp_path):
    input_file = tmp_path / "NPM1_13054_IM.tiff"
    input_file.write_text("tiff", encoding="utf-8")
    service_output_dir = "/srv/ultra/megaseg-service/artifacts/job-123/results"
    captured: dict[str, object] = {}

    def fake_science_output_root(*parts: str) -> str:
        root = tmp_path / "science"
        target = root.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    class FakeMegasegClient:
        def __init__(self, *, base_url, api_key=None, timeout_seconds=60.0):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout_seconds"] = timeout_seconds

        def submit_job(self, *, request_payload, local_upload_paths=None):
            captured["request_payload"] = request_payload
            captured["local_upload_paths"] = [str(path) for path in list(local_upload_paths or [])]
            return {"job_id": "job-123", "status": "queued"}

        def wait_for_job(self, *, job_id, poll_interval_seconds=2.0, wait_timeout_seconds=7200.0):
            captured["poll_interval_seconds"] = poll_interval_seconds
            captured["wait_timeout_seconds"] = wait_timeout_seconds
            return {
                "job_id": job_id,
                "status": "succeeded",
                "artifact_manifest": [
                    {"name": "megaseg_report.md", "size_bytes": 12},
                    {"name": "sample/sample__megaseg_mask.tiff", "size_bytes": 12},
                    {"name": "sample/sample__megaseg_probability.tiff", "size_bytes": 12},
                    {"name": "sample/sample__megaseg_summary.json", "size_bytes": 12},
                ],
                "result": {
                    "success": True,
                    "device": "cuda",
                    "checkpoint_path": "/srv/ultra/models/megaseg/epoch_650.ckpt",
                    "output_directory": service_output_dir,
                    "summary_csv_path": f"{service_output_dir}/megaseg_summary.csv",
                    "report_path": f"{service_output_dir}/megaseg_report.md",
                    "warnings": [],
                    "aggregate": {
                        "mean_coverage_percent": 2.5,
                        "median_coverage_percent": 2.5,
                        "mean_object_count": 4.0,
                        "median_object_count": 4.0,
                    },
                    "files": [
                        {
                            "file": "sample",
                            "success": True,
                            "mask_path": f"{service_output_dir}/sample/sample__megaseg_mask.tiff",
                            "probability_path": f"{service_output_dir}/sample/sample__megaseg_probability.tiff",
                            "summary_json_path": f"{service_output_dir}/sample/sample__megaseg_summary.json",
                            "visualizations": [],
                            "segmentation": {
                                "coverage_percent": 2.5,
                                "object_count": 4,
                                "active_slice_count": 3,
                                "largest_component_voxels": 84,
                            },
                            "intensity_context": {},
                            "technical_summary": "ok",
                        }
                    ],
                },
            }

        def download_artifact(self, *, job_id, artifact_name, destination):
            captured.setdefault("downloads", []).append(
                {
                    "job_id": job_id,
                    "artifact_name": artifact_name,
                    "destination": str(destination),
                }
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"artifact:{artifact_name}", encoding="utf-8")
            return destination

    monkeypatch.setattr(tools, "MegasegServiceClient", FakeMegasegClient)
    monkeypatch.setattr(tools, "_science_output_root", fake_science_output_root)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(
            resolved_megaseg_service_url="http://megaseg.example.invalid:8010",
            resolved_megaseg_service_api_key="secret-token",
            megaseg_service_timeout_seconds=15.0,
            megaseg_service_poll_interval_seconds=0.25,
            megaseg_service_wait_timeout_seconds=30.0,
            megaseg_service_download_artifacts=True,
            resolved_megaseg_python=None,
            resolved_megaseg_checkpoint_path=None,
            resolved_megaseg_benchmark_root=None,
        ),
    )

    result = tools.segment_image_megaseg(
        file_paths=[str(input_file), "s3://allencell/aics/example.ome.zarr/"],
        save_visualizations=False,
        generate_report=True,
    )

    assert result["success"] is True
    assert captured["base_url"] == "http://megaseg.example.invalid:8010"
    assert captured["local_upload_paths"] == [str(input_file.resolve())]
    assert captured["request_payload"]["sources"] == [
        {"uri": "s3://allencell/aics/example.ome.zarr/"}
    ]
    assert Path(result["preferred_upload_paths"][0]).exists()
    assert result["checkpoint_path"] == "/srv/ultra/models/megaseg/epoch_650.ckpt"
    assert str(result["report_path"]).startswith(str(tmp_path / "science" / "megaseg_results"))


def test_infer_scientific_image_inputs_prefers_remote_prompt_source():
    prompt = """
aws s3 cp --recursive "s3://allencell/aics/emt_timelapse_dataset/data/1500004526_10_raw_converted.ome.zarr/" "./1500004526_10_raw_converted.ome.zarr"
"""

    candidates = tools.extract_scientific_image_paths_from_text(prompt)
    preferred = tools.infer_scientific_image_inputs_from_text(prompt)

    assert candidates == [
        "s3://allencell/aics/emt_timelapse_dataset/data/1500004526_10_raw_converted.ome.zarr/",
        "./1500004526_10_raw_converted.ome.zarr",
    ]
    assert preferred == [
        "s3://allencell/aics/emt_timelapse_dataset/data/1500004526_10_raw_converted.ome.zarr/",
    ]


def test_execute_tool_call_infers_prompt_image_paths_for_megaseg(monkeypatch):
    captured: dict[str, object] = {}

    def fake_megaseg_tool(file_paths, save_visualizations=True, **_kwargs):
        captured["file_paths"] = list(file_paths)
        captured["save_visualizations"] = bool(save_visualizations)
        return {"success": True, "file_paths": list(file_paths)}

    monkeypatch.setitem(tools.AVAILABLE_TOOLS, "segment_image_megaseg", fake_megaseg_tool)

    result = json.loads(
        tools.execute_tool_call(
            "segment_image_megaseg",
            {},
            uploaded_files=[],
            user_text=(
                "Run MegaSeg on "
                "s3://allencell/aics/emt_timelapse_dataset/data/1500004526_10_raw_converted.ome.zarr/ "
                "and quantify the result."
            ),
        )
    )

    assert result["success"] is True
    assert captured["file_paths"] == [
        "s3://allencell/aics/emt_timelapse_dataset/data/1500004526_10_raw_converted.ome.zarr/",
    ]
    assert captured["save_visualizations"] is True


def test_execute_tool_call_replaces_bisque_resource_uri_with_uploaded_megaseg_input(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}
    input_file = tmp_path / "NPM1_13054_IM.tiff"
    input_file.write_text("tiff", encoding="utf-8")

    def fake_megaseg_tool(file_paths, save_visualizations=True, **_kwargs):
        captured["file_paths"] = list(file_paths)
        captured["save_visualizations"] = bool(save_visualizations)
        return {"success": True, "file_paths": list(file_paths)}

    monkeypatch.setitem(tools.AVAILABLE_TOOLS, "segment_image_megaseg", fake_megaseg_tool)

    result = json.loads(
        tools.execute_tool_call(
            "segment_image_megaseg",
            {"file_paths": ["http://localhost:8080/data_service/00-staleMegasegImage"]},
            uploaded_files=[str(input_file)],
            user_text="Run megaseg on this image and write a report.",
        )
    )

    assert result["success"] is True
    assert captured["file_paths"] == [str(input_file)]
    assert captured["save_visualizations"] is True


def test_execute_tool_call_replaces_bisque_resource_uri_with_selection_context_image(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}
    input_file = tmp_path / "NPM1_13054_IM.tiff"
    input_file.write_text("tiff", encoding="utf-8")

    def fake_megaseg_tool(file_paths, save_visualizations=True, **_kwargs):
        captured["file_paths"] = list(file_paths)
        captured["save_visualizations"] = bool(save_visualizations)
        return {"success": True, "file_paths": list(file_paths)}

    monkeypatch.setitem(tools.AVAILABLE_TOOLS, "segment_image_megaseg", fake_megaseg_tool)

    result = json.loads(
        tools.execute_tool_call(
            "segment_image_megaseg",
            {"file_paths": ["http://localhost:8080/data_service/00-staleMegasegImage"]},
            uploaded_files=[],
            user_text="Run megaseg on the selected image.",
            selection_context={
                "resource_uris": ["http://localhost:8080/data_service/00-currentSelection"],
                "artifact_handles": {"image_files": [str(input_file)]},
            },
        )
    )

    assert result["success"] is True
    assert captured["file_paths"] == [str(input_file.resolve())]
    assert captured["save_visualizations"] is True


def test_load_scientific_image_accepts_remote_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(
        imaging,
        "get_settings",
        lambda: SimpleNamespace(science_data_root=str(tmp_path / "science")),
    )
    monkeypatch.setattr(
        imaging,
        "_load_with_bioio",
        lambda *args, **kwargs: {
            "success": True,
            "metadata": {},
            "warnings": [],
            "reader": "bioio",
            "file_path": args[0] if args else kwargs["file_path"],
        },
    )

    result = imaging.load_scientific_image(
        file_path="s3://allencell/aics/example.ome.zarr/",
        generate_preview=False,
        save_array=False,
    )

    assert result["success"] is True
    assert result["file_path"] == "s3://allencell/aics/example.ome.zarr/"
    assert result["metadata"]["filename_hints"]["tokens"] == ["example", "ome"]
