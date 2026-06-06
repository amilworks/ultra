from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


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
        assert not (ROOT / relative_path).exists()


def test_go_control_stack_deploy_script_targets_primary_runtime() -> None:
    script = read_repo_file("scripts/deploy_ultra_control_stack.sh")

    assert "backend/controlplane" in script
    assert "go build" in script
    assert "ultra-control migrate" in script
    assert "systemctl restart ultra-control" in script
    assert "systemctl restart ultra-deepagents-worker" in script
    assert "systemctl restart ultra-rarespot-worker" in script
    assert "SYSTEMD_UNIT_DIR" in script
    assert "ultra-control-stack.target" in script
    assert "ULTRA_CONTROL_DATABASE_URL" in script
    assert "ULTRA_CONTROL_NATS_URL" in script
    assert "/v1/health" in script
    assert "/v2/admin/overview" in script


def test_systemd_units_run_go_control_and_deepagents_workers() -> None:
    control = read_repo_file("deploy/systemd/ultra-control.service")
    deepagents = read_repo_file("deploy/systemd/ultra-deepagents-worker.service")
    rarespot = read_repo_file("deploy/systemd/ultra-rarespot-worker.service")

    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in control
    assert "ExecStart=/srv/ultra/current/bin/ultra-control serve" in control
    assert "Environment=ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8000" in control

    assert "After=ultra-control.service" in deepagents
    assert "ExecStart=/srv/ultra/current/backend/deepagents_runtime/.venv/bin/python -m ultra_deepagents.nats_worker" in deepagents
    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in deepagents

    assert "After=ultra-control.service" in rarespot
    assert "ExecStart=/srv/ultra/current/backend/deepagents_runtime/.venv/bin/python -m ultra_deepagents.rarespot_worker" in rarespot
    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in rarespot


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
    assert "handle /v2/*" in caddy
    assert "reverse_proxy 127.0.0.1:8000" in caddy
    assert "handle /v1/health" in caddy
    assert "handle /v1/config*" in caddy
    assert "handle /v1/session*" in caddy


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
