import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def tracked_paths_under(relative_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", relative_path],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_internal_bisque_platform_checkout_is_not_public_deploy_surface() -> None:
    removed_paths = [
        "platform/bisque",
        "docs/superpowers",
        ".github/workflows/deploy-platform-manual.yml",
        "deploy/env/platform.env.example",
        "deploy/systemd/ultra-platform.service",
        "scripts/deploy_platform_manual.sh",
        "scripts/verify_platform_smoke.sh",
    ]

    for relative_path in removed_paths:
        assert tracked_paths_under(relative_path) == []


def test_go_control_stack_deploy_script_targets_primary_runtime() -> None:
    script = read_repo_file("scripts/deploy_ultra_control_stack.sh")

    assert "backend/controlplane" in script
    assert "go build" in script
    assert "ultra-control migrate" in script
    assert "systemctl restart ultra-control" in script
    assert "systemctl restart ultra-deepagents-worker" in script
    assert "SYSTEMD_UNIT_DIR" in script
    assert "ultra-control-stack.target" in script
    assert "ULTRA_CONTROL_DATABASE_URL" in script
    assert "ULTRA_CONTROL_MIGRATION_DATABASE_URL" in script
    assert "ULTRA_MIGRATION_ENV_FILE" in script
    assert "Move ULTRA_CONTROL_MIGRATION_DATABASE_URL out of" in script
    assert "ULTRA_CONTROL_NATS_URL" in script
    assert "ULTRA_CONTROL_CALPHAD_RUNTIME_IMAGE_ID" in script
    assert "verify_calphad_runtime_image_binding" in script
    assert "docker image inspect --format '{{.Id}}'" in script
    assert "install_sandbox_identity_env" in script
    assert "ULTRA_SANDBOX_IDENTITY_ENV_FILE" in script
    assert "ULTRA_DEEPAGENTS_SANDBOX_IMAGE=%s" in script
    assert "build_kinetics_runtime_image" in script
    assert "verify_kinetics_runtime_image_binding" in script
    assert "ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE=%s" in script
    assert "ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID=%s" in script
    assert "Ultra isolated materials kinetics runtime" in script
    assert 'chmod 0600 "$temporary"' in script
    assert "Using manifest-bound immutable control binary" in script
    assert 'if [ -f "$RELEASE_DIR/release-manifest.json" ]' in script
    assert '--repo-root "$RELEASE_DIR"' in script
    assert "--scope production-full" in script
    assert '"$parity_dir/bundle/release"' in script
    assert '"$parity_dir/bundle/staged"' in script
    assert "/v1/health" in script
    assert "/v2/admin/overview" in script


def test_deepagents_worker_units_override_mutable_tag_with_verified_image_id() -> None:
    for unit in (
        "deploy/systemd/ultra-deepagents-worker.service",
        "deploy/systemd/ultra-deepagents-worker-node.service",
    ):
        text = read_repo_file(unit)
        ordinary_environment = text.index("EnvironmentFile=/etc/ultra/ultra-")
        immutable_environment = text.index(
            "EnvironmentFile=/etc/ultra/ultra-sandbox-image.env"
        )
        assert immutable_environment > ordinary_environment


def test_compose_binds_worker_to_built_kinetics_runtime_identity() -> None:
    compose = read_repo_file("docker-compose.yml")
    makefile = read_repo_file("Makefile")

    assert "materials-kinetics-runtime:" in compose
    assert "deploy/docker/materials-kinetics.Dockerfile" in compose
    assert "ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE:" in compose
    assert "ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID:" in compose
    assert "condition: service_completed_successfully" in compose
    assert "materials-kinetics-image:" in makefile
    assert "docker image inspect --format '{{.Id}}'" in makefile
    assert 'ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID="$$image_id"' in makefile


def test_migration_database_credential_is_separate_from_serving_environment() -> None:
    backend = read_repo_file("deploy/env/ultra-backend.env.example")
    migration = read_repo_file("deploy/env/ultra-migration.env.example")

    assert "ULTRA_CONTROL_MIGRATION_DATABASE_URL" not in backend
    assert "ULTRA_CONTROL_MIGRATION_DATABASE_URL" in migration


def test_frontend_deploy_points_web_root_at_built_dist() -> None:
    script = read_repo_file("scripts/deploy_ultra_frontend.sh")

    assert 'RELEASE_DIR="$ULTRA_RELEASE_ROOT/releases/$RELEASE_SHA/frontend/dist"' in script
    assert 'CURRENT_LINK="$ULTRA_RELEASE_ROOT/frontend-current"' in script


def test_release_artifact_script_builds_immutable_control_stack_bundle() -> None:
    script = read_repo_file("scripts/build_ultra_release_artifact.sh")

    assert "git archive --format=tar" in script
    assert "frontend/dist/index.html" in script
    assert "go build -trimpath" in script
    assert 'GOOS_VALUE="${GOOS:-linux}"' in script
    assert 'GOARCH_VALUE="${GOARCH:-amd64}"' in script
    assert "bin/ultra-control" in script
    assert "frontend/dist" in script
    assert "release-manifest.json" in script
    assert "ultra-release-$RELEASE_SHA.tar.gz" in script
    assert "sha256sum" in script or "shasum -a 256" in script
    assert 'head_sha="$(git rev-parse HEAD)"' in script
    assert 'if [ "$RELEASE_SHA" != "$head_sha" ]' in script
    assert "--ignore-submodules=none" in script
    assert '"control_binary_identity": release_artifacts["control_binary"]' in script
    assert '"frontend_dist_identity": release_artifacts["frontend_dist"]' in script
    assert '"full_materials_production_ready": False' in script
    assert '"required_post_image_gate": "materials-production-readiness"' in script
    assert '"readiness_result_is_candidate_only": True' in script
    assert '"ultra.materials.release-envelope.v1"' in script
    assert '"ultra.materials.production-attestation-verification.v1"' in script
    assert '"requires_github_sigstore_attestation": True' in script
    assert '"raw_mattools_evidence_may_be_public_artifact": False' in script
    assert '"production_parity_scope": module.PRODUCTION_PARITY_SCOPE' in script
    assert "materials_verifier.EVIDENCE_BUNDLE_SCHEMA_VERSION" in script
    assert '"docker_sandbox_backend_combined_stdout_stderr"' in script
    assert "materials_verifier.REQUIRED_CALPHAD_RUNTIME_TEST_COUNT" in script
    assert "materials_verifier.REQUIRED_CALPHAD_TOOLS_TEST_COUNT" in script
    assert '"calphad_cross_language_schema"' in script
    assert '"calphad_cross_language_requires_production_runtime_image": True' in script
    assert '"mattools_runnable_minimum": module.RUNNABLE_MINIMUM' in script
    assert '"mattools_scientific_minimum": module.SCIENTIFIC_MINIMUM' in script


def _release_builder_fixture(tmp_path: Path) -> tuple[Path, str]:
    script = tmp_path / "scripts/build_ultra_release_artifact.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        read_repo_file("scripts/build_ultra_release_artifact.sh"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    frontend = tmp_path / "frontend/dist/index.html"
    frontend.parent.mkdir(parents=True)
    frontend.write_text("<main>fixture</main>\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "release@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Release Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    return script, head


def test_release_builder_rejects_release_sha_different_from_clean_head(
    tmp_path: Path,
) -> None:
    script, head = _release_builder_fixture(tmp_path)
    wrong = "0" * 40 if head != "0" * 40 else "1" * 40
    process = subprocess.run(
        [str(script)],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_SHA": wrong},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "RELEASE_SHA must equal the checked-out HEAD" in process.stderr


def test_release_builder_rejects_untracked_files_at_head(tmp_path: Path) -> None:
    script, head = _release_builder_fixture(tmp_path)
    (tmp_path / "untracked.txt").write_text("not releasable\n", encoding="utf-8")
    process = subprocess.run(
        [str(script)],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_SHA": head},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "dirty or untracked worktree" in process.stderr


def test_pr_ci_workflow_covers_release_gates() -> None:
    workflow = read_repo_file(".github/workflows/pr-ci.yml")

    assert "name: PR CI" in workflow
    assert "scripts/release_codescan.sh" in workflow
    assert "pnpm --dir frontend lint" in workflow
    assert "pnpm --dir frontend typecheck" in workflow
    assert "pnpm --dir frontend test:unit" in workflow
    assert "pnpm --dir frontend build" in workflow
    assert "pnpm --dir frontend bundle:check" in workflow
    assert "pnpm --dir frontend test:smoke" in workflow
    assert "make control-test" in workflow
    assert "make deepagents-test" in workflow
    assert "make deepagents-autonomy-test" in workflow


def test_main_release_workflow_builds_uploadable_release_artifact() -> None:
    workflow = read_repo_file(".github/workflows/release-artifacts.yml")

    assert "name: Release Artifacts" in workflow
    assert "branches:" in workflow
    assert "- main" in workflow
    assert "scripts/release_codescan.sh" in workflow
    assert "scripts/build_ultra_release_artifact.sh" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "ultra-release-${{ github.sha }}" in workflow
    assert "release-manifest-${{ github.sha }}.json" in workflow
    assert "tests/test_materials_readiness_gate.py" in workflow
    assert "Run deterministic materials evidence gate" in workflow
    assert "Assert source artifact makes no materials production-readiness claim" in workflow
    assert 'materials["full_materials_production_ready"] is False' in workflow
    assert "production_parity_evidence_bundle_schema_version" in workflow
    assert "docker_sandbox_backend_combined_stdout_stderr" in workflow
    assert 'materials["required_evidence"]["calphad_runtime_test_count"] == 39' in workflow
    assert 'materials["required_evidence"]["calphad_tool_test_count"] == 56' in workflow
    assert 'materials["required_evidence"]["calphad_cross_language_schema"]' in workflow
    assert "calphad_cross_language_requires_production_runtime_image" in workflow
    assert "ultra_calphad_qualification" in workflow
    assert "make calphad-ledger-qualification" in workflow
    assert "make materials-promotion-envelope-test" in workflow


def test_materials_workflow_has_live_ledger_and_nonpromotable_cross_language_preflight() -> None:
    workflow = read_repo_file(".github/workflows/materials-domain-gate.yml")
    ledger_gate = read_repo_file("scripts/calphad_ledger_gate.py")

    assert "calphad-ledger-postgres:" in workflow
    assert "image: postgres:18-alpine" in workflow
    assert "make calphad-ledger-qualification" in workflow
    assert "tests/test_calphad_ledger_gate.py" in workflow
    assert "make calphad-cross-language-test" in workflow
    assert "make calphad-cross-language-qualification" not in workflow
    assert "Build pinned typed-CALPHAD runtime image" not in workflow
    assert "Upload CALPHAD cross-language qualification evidence" not in workflow
    assert "tests/test_calphad_cross_language_gate.py" in workflow
    assert "TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound" in ledger_gate


def test_systemd_units_run_go_control_and_deepagents_workers() -> None:
    control = read_repo_file("deploy/systemd/ultra-control.service")
    deepagents = read_repo_file("deploy/systemd/ultra-deepagents-worker.service")

    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in control
    assert "ExecStart=/srv/ultra/current/bin/ultra-control serve" in control
    assert "Environment=ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8000" in control

    assert "After=ultra-control.service" in deepagents
    assert "ExecStart=/srv/ultra/current/backend/deepagents_runtime/.venv/bin/python -m ultra_deepagents.nats_worker" in deepagents
    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in deepagents


def test_proxy_templates_route_modern_app_to_go_control_plane() -> None:
    nginx_templates = [
        read_repo_file("deploy/nginx/ultra.conf.template"),
        read_repo_file("deploy/nginx/ultra-single-host.conf.template"),
    ]
    for template in nginx_templates:
        assert "upstream ultra_control" in template
        assert "server 127.0.0.1:8000" in template
        assert "location /v2/" in template
        assert "proxy_pass http://ultra_control" in template
        assert "location = /v1/health" in template
        assert "location /v1/config" in template
        assert "location /v1/session" in template
        assert "proxy_buffering off" in template

    caddy = read_repo_file("deploy/caddy/Caddyfile.single-host.template")
    assert "@api path /v1/* /v2/* /v3/* /docs* /openapi.json" in caddy
    assert "handle @api" in caddy
    assert "reverse_proxy 127.0.0.1:8000" in caddy


def test_staging_env_example_documents_required_control_stack_settings() -> None:
    env = read_repo_file("deploy/env/ultra-backend.env.example")

    required_keys = [
        "ENVIRONMENT=production",
        "ULTRA_CONTROL_AUTH_PROVIDER=workos",
        "ULTRA_CONTROL_WORKOS_CLIENT_ID=",
        "ULTRA_CONTROL_WORKOS_API_KEY=",
        "ULTRA_CONTROL_WORKOS_REDIRECT_URI=https://ultra.example.com/v2/auth/workos/callback",
        "ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD=",
        "ULTRA_CONTROL_SECRET_ENCRYPTION_KEY=",
        "ULTRA_CONTROL_DATABASE_URL=",
        "ULTRA_CONTROL_NATS_URL=",
        "ULTRA_CONTROL_ARTIFACT_ROOT=",
        "ULTRA_CONTROL_BISQUE_ROOT_URL=",
        "ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8000",
        "ULTRA_CONTROL_BASE_URL=http://127.0.0.1:8000",
        "ULTRA_CONTROL_UPLOAD_ROOT=/srv/ultra/shared/uploads",
    ]

    for key in required_keys:
        assert key in env


def test_default_deepagents_sandbox_includes_medical_and_bioimaging_stack() -> None:
    dockerfile = read_repo_file("deploy/docker/deepagents-sandbox.Dockerfile")

    assert (
        "FROM python:3.11-slim@sha256:"
        "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
    ) in dockerfile

    required_system_packages = [
        "build-essential",
        "cmake",
        "dcm2niix",
        "dcmtk",
        "default-jre-headless",
        "libopenslide0",
        "libvips42",
        "openslide-tools",
        "pkg-config",
    ]
    required_python_packages = [
        "SimpleITK",
        "bioio",
        "bioio-bioformats",
        "bioio-czi",
        "bioio-dv",
        "bioio-imageio",
        "bioio-lif",
        "bioio-nd2",
        "bioio-ome-tiff",
        "bioio-ome-zarr",
        "bioio-sldy",
        "bioio-tiff-glob",
        "bioio-tifffile",
        "connected-components-3d",
        "dask[array]",
        "dicom2nifti",
        "dipy",
        "highdicom",
        "imagecodecs",
        "itk",
        "mrcfile",
        "monai",
        "nilearn",
        "numpy==1.26.4",
        "ome-types",
        "ome-zarr",
        "opencv-python-headless==4.10.0.84",
        "openslide-python",
        "pydicom",
        "pynrrd",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
        "pylibjpeg-openjpeg",
        "pylibjpeg-rle",
        "pyvips",
        "roifile",
        "rt-utils",
        "torchio",
        "xarray",
        "zarr",
    ]

    for package in required_system_packages:
        assert package in dockerfile
    for package in required_python_packages:
        assert package in dockerfile
    assert "--no-deps rt-utils" in dockerfile
