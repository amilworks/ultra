import os
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_worker_lock_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in read_repo_file(
        "backend/deepagents_runtime/requirements.worker.lock"
    ).splitlines():
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        requirement = line.split("\\", 1)[0].strip()
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", 1)
        packages[name] = version.split(";", 1)[0].strip()
    return packages


def parse_canonical_lock_versions() -> dict[str, set[str]]:
    lock = tomllib.loads(read_repo_file("backend/deepagents_runtime/uv.lock"))
    packages: dict[str, set[str]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], set()).add(package["version"])
    return packages


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
    assert "verify_sandbox_runtime_image_binding" in script
    assert "docker image inspect --format '{{.Id}}'" in script
    assert "install_sandbox_identity_env" in script
    assert "ULTRA_SANDBOX_IDENTITY_ENV_FILE" in script
    assert "ULTRA_DEEPAGENTS_SANDBOX_IMAGE=%s" in script
    assert 'chmod 0600 "$temporary"' in script
    assert "Using manifest-bound immutable control binary" in script
    assert 'if [ -f "$RELEASE_DIR/release-manifest.json" ]' in script
    assert "/v1/health" in script
    assert "/v2/admin/overview" in script


def test_deepagents_worker_dockerfile_installs_only_hash_locked_dependencies() -> None:
    dockerfile = read_repo_file("backend/deepagents_runtime/Dockerfile.worker")

    assert "requirements.worker.lock" in dockerfile
    assert "--only-binary=:all:" in dockerfile
    assert "--require-hashes" in dockerfile
    assert (
        "pip install --no-cache-dir --no-build-isolation --no-deps "
        "/app/deepagents_runtime"
    ) in dockerfile
    assert "ARG ULTRA_WORKER_NUMPY_VERSION" not in dockerfile
    assert "assert numpy.__version__ == '1.26.4'" in dockerfile
    assert "pip check" in dockerfile
    assert "import ultra_deepagents.agent" in dockerfile
    assert "import ultra_deepagents.nats_worker" in dockerfile


def test_native_worker_deploy_syncs_hash_locked_release_environment_before_symlink() -> None:
    script = read_repo_file("scripts/deploy_ultra_control_stack.sh")

    assert 'DEEPAGENTS_WORKER_LOCK="$DEEPAGENTS_DIR/requirements.worker.lock"' in script
    assert '"$UV_BIN" venv --python "$UV_PYTHON_VERSION" "$DEEPAGENTS_VENV_DIR"' in script
    assert '"$UV_BIN" pip sync' in script
    assert "--require-hashes" in script
    assert "--only-binary=:all:" in script
    assert '"$UV_BIN" pip install' in script
    assert "--no-build-isolation" in script
    assert "--no-deps" in script
    assert '"$UV_BIN" pip check --python "$DEEPAGENTS_VENV_DIR/bin/python"' in script
    assert "assert numpy.__version__ == '1.26.4'" in script
    assert "import ultra_deepagents.agent" in script
    assert "import ultra_deepagents.nats_worker" in script
    sync_index = script.index('"$UV_BIN" pip sync')
    install_index = script.index('"$UV_BIN" pip install')
    check_index = script.index('"$UV_BIN" pip check')
    import_index = script.index('"$DEEPAGENTS_VENV_DIR/bin/python" -c')
    symlink_index = script.index(
        'ln -sfn "$DEEPAGENTS_VENV_DIR" "$DEEPAGENTS_DIR/.venv"'
    )
    restart_index = script.index("systemctl restart ultra-deepagents-worker")
    assert sync_index < install_index < check_index < import_index < symlink_index
    assert symlink_index < restart_index


def test_worker_lock_is_hashed_and_carries_qualified_runtime_versions() -> None:
    constraints = read_repo_file(
        "backend/deepagents_runtime/requirements.worker-constraints.txt"
    )
    lock = read_repo_file("backend/deepagents_runtime/requirements.worker.lock")

    assert "numpy==1.26.4" in constraints
    assert "#    make deepagents-worker-lock" in lock
    for requirement in (
        "deepagents==0.7.5",
        "hatchling==1.28.0",
        "langchain-core==1.5.2",
        "numpy==1.26.4",
    ):
        start = lock.index(requirement)
        assert "--hash=sha256:" in lock[start : start + 1000]


def test_worker_lock_matches_canonical_runtime_except_qualified_numpy() -> None:
    canonical = parse_canonical_lock_versions()
    worker = parse_worker_lock_versions()
    shared = set(canonical).intersection(worker)
    mismatches = {
        name: {"canonical": canonical[name], "worker": {worker[name]}}
        for name in shared - {"numpy"}
        if canonical[name] != {worker[name]}
    }

    assert mismatches == {}
    assert worker["numpy"] == "1.26.4"
    assert {version.split(".", 1)[0] for version in canonical["numpy"]} == {"2"}


def test_worker_build_backend_is_exact_and_part_of_the_hashed_closure() -> None:
    build_input = read_repo_file(
        "backend/deepagents_runtime/requirements.worker-build.txt"
    )
    project = tomllib.loads(read_repo_file("backend/deepagents_runtime/pyproject.toml"))
    worker = parse_worker_lock_versions()

    requirements = [
        line for line in build_input.splitlines() if line and not line.startswith("#")
    ]
    assert requirements == ["hatchling==1.28.0"]
    assert project["build-system"]["requires"] == ["hatchling==1.28.0"]
    assert worker["hatchling"] == "1.28.0"


def test_deepagents_worker_units_override_mutable_tag_with_verified_image_id() -> None:
    for unit in (
        "deploy/systemd/ultra-deepagents-worker.service",
        "deploy/systemd/ultra-deepagents-worker-node.service",
    ):
        text = read_repo_file(unit)
        ordinary_environment = text.index("EnvironmentFile=/etc/ultra/ultra-")
        immutable_environment = text.index("EnvironmentFile=/etc/ultra/ultra-sandbox-image.env")
        assert immutable_environment > ordinary_environment


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


def test_release_manifest_binds_deepagents_worker_lock_and_versions() -> None:
    script = read_repo_file("scripts/build_ultra_release_artifact.sh")

    assert 'Path("backend/deepagents_runtime/requirements.worker.lock")' in script
    assert 'Path("backend/deepagents_runtime/uv.lock")' not in script
    assert '"hatchling",' in script
    assert '"numpy",' in script
    assert '"deepagents_worker_lock": worker_lock_metadata()' in script
    assert '"sha256": hashlib.sha256(payload).hexdigest()' in script
    assert '"python_version": "3.11"' in script
    assert '"python_platform": "x86_64-manylinux_2_28"' in script
    assert '"schema_version": 1' in script


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


def test_only_pr_and_release_ci_check_the_pinned_worker_lock() -> None:
    makefile = read_repo_file("Makefile")
    pr_ci = read_repo_file(".github/workflows/pr-ci.yml")
    release_ci = read_repo_file(".github/workflows/release-artifacts.yml")

    assert "deepagents-worker-lock:" in makefile
    assert "deepagents-worker-lock-check:" in makefile
    assert "deepagents-worker-env-check:" in makefile
    assert "--python-platform x86_64-manylinux_2_28" in makefile
    assert makefile.count("uv run --frozen --python 3.11 --extra dev pytest") == 3
    lock_recipe = makefile.split("deepagents-worker-lock:", 1)[1].split(
        "\ndeepagents-worker-lock-check:", 1
    )[0]
    assert "uv lock --check" in lock_recipe
    assert "uv export --quiet --frozen --no-dev --no-hashes --no-header" in lock_recipe
    assert "--no-emit-project --no-annotate" in lock_recipe
    assert "awk" in lock_recipe
    assert "/^numpy==/" in lock_recipe
    assert "numpy_count != 1" in lock_recipe
    assert "pyproject.toml requirements.worker-build.txt" in lock_recipe
    assert lock_recipe.count("--constraint") == 2
    assert lock_recipe.count("--no-annotate") == 2
    assert "requirements.worker-constraints.txt" in lock_recipe
    assert "--upgrade" not in lock_recipe
    lock_check_recipe = makefile.split("deepagents-worker-lock-check:", 1)[1].split(
        "\ndeepagents-worker-env-check:", 1
    )[0]
    seed_copy = (
        "cp backend/deepagents_runtime/requirements.worker.lock "
        '"$$worker_lock_candidate"'
    )
    assert seed_copy in lock_check_recipe
    assert lock_check_recipe.index(seed_copy) < lock_check_recipe.index(
        "$(MAKE) deepagents-worker-lock"
    )
    assert 'DEEPAGENTS_WORKER_LOCK_OUTPUT="$$worker_lock_candidate"' in lock_check_recipe
    assert "cmp" in lock_check_recipe
    assert "--upgrade" not in lock_check_recipe
    env_check_recipe = makefile.split("deepagents-worker-env-check:", 1)[1].split(
        "\ndeepagents-test:", 1
    )[0]
    assert "mktemp -d /tmp/ultra-worker-env.XXXXXX" in env_check_recipe
    assert "/tmp/ultra-worker-env.*" in env_check_recipe
    assert "uv venv --python 3.11" in env_check_recipe
    assert "uv pip sync" in env_check_recipe
    assert "--require-hashes" in env_check_recipe
    assert "--only-binary=:all:" in env_check_recipe
    assert "uv pip install" in env_check_recipe
    assert "--no-build-isolation" in env_check_recipe
    assert "--no-deps" in env_check_recipe
    assert "uv pip check" in env_check_recipe
    assert "assert numpy.__version__ == '1.26.4'" in env_check_recipe
    assert "import ultra_deepagents.agent" in env_check_recipe
    assert "import ultra_deepagents.nats_worker" in env_check_recipe
    assert 'rm -rf -- "$$worker_env_dir"' in env_check_recipe
    for workflow in (pr_ci, release_ci):
        assert 'version: "0.9.30"' in workflow
        assert "make deepagents-worker-lock-check" in workflow
        assert "make deepagents-worker-env-check" in workflow
        assert workflow.index("make deepagents-worker-lock-check") < workflow.index(
            "make deepagents-worker-env-check"
        )
        assert "docker build" not in workflow
        assert "docker buildx" not in workflow
    assert "uv run --frozen --extra dev pytest -q" in release_ci
    for path in (
        ".github/workflows/autonomy-gate.yml",
        ".github/workflows/deploy-ultra-frontend.yml",
        ".github/workflows/labeler.yml",
    ):
        assert "deepagents-worker-lock-check" not in read_repo_file(path)


def test_systemd_units_run_go_control_and_deepagents_workers() -> None:
    control = read_repo_file("deploy/systemd/ultra-control.service")
    deepagents = read_repo_file("deploy/systemd/ultra-deepagents-worker.service")

    assert "EnvironmentFile=/etc/ultra/ultra-backend.env" in control
    assert "ExecStart=/srv/ultra/current/bin/ultra-control serve" in control
    assert "Environment=ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8000" in control

    assert "After=ultra-control.service" in deepagents
    assert (
        "ExecStart=/srv/ultra/current/backend/deepagents_runtime/.venv/bin/python -m ultra_deepagents.nats_worker"
        in deepagents
    )
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
