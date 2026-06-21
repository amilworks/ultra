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
    assert "ULTRA_CONTROL_NATS_URL" in script
    assert "/v1/health" in script
    assert "/v2/admin/overview" in script


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
        "BISQUE_UPSTREAM=http://127.0.0.1:8080",
        "ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8000",
        "ULTRA_CONTROL_BASE_URL=http://127.0.0.1:8000",
        "ULTRA_CONTROL_UPLOAD_ROOT=/srv/ultra/shared/uploads",
    ]

    for key in required_keys:
        assert key in env


def test_default_deepagents_sandbox_includes_medical_and_bioimaging_stack() -> None:
    dockerfile = read_repo_file("deploy/docker/deepagents-sandbox.Dockerfile")

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
