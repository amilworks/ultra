import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_bioimageconvert_tree(root: Path) -> None:
    project_text = """\
CONFIG += lib_libgeotiff
CONFIG += lib_proj
CONFIG += lib_libczi
LIBS += $$BIM_LIBS_PLTFM/libjxrglue.a
LIBS += $$BIM_LIBS_PLTFM/libjpegxr.a
LIBS += $$BIM_LIBS_PLTFM/libczi.a
LIBS += -lzstd
"""
    for relative in [
        "src/imgcnv.pro",
        "src_dylib/libimgcnv.pro",
        "libsrc/libbioimg/bioimage.pro",
    ]:
        _write(root / relative, project_text)

    _write(
        root / "Makefile.linux",
        """\
buildlibs: $(LIBS)/libjpegxr.a $(LIBS)/libczi.a $(LIBS)/libgeotiff.a $(LIBS)/libproj.a
$(LIBS)/libczi.a:
\t@echo build czi
$(LIBS)/libgeotiff.a:
\t@echo build geotiff
""",
    )
    _write(
        root / "libsrc/libbioimg/formats/tiff/bim_geotiff_parse.cpp",
        """\
#include <sstream>
#include "bim_geotiff_parse.h"
""",
    )


def test_engine_patch_keeps_czi_enabled_with_duplicate_jpegxr_link_guard(tmp_path: Path) -> None:
    tree = tmp_path / "bioimageconvert"
    _seed_bioimageconvert_tree(tree)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "backend/deepagents_runtime/imaging-engine-build/apply-engine-patches.sh"),
            str(tree),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for relative in [
        "src/imgcnv.pro",
        "src_dylib/libimgcnv.pro",
        "libsrc/libbioimg/bioimage.pro",
    ]:
        patched = (tree / relative).read_text(encoding="utf-8")
        assert "CONFIG += lib_libczi" in patched
        assert "disabled: CZI" not in patched

    for relative in ["src/imgcnv.pro", "src_dylib/libimgcnv.pro"]:
        patched = (tree / relative).read_text(encoding="utf-8")
        assert "QMAKE_LFLAGS += -Wl,--allow-multiple-definition" in patched
        assert patched.count("QMAKE_LFLAGS += -Wl,--allow-multiple-definition") == 1
        assert "LIBS += -lzstd  # libtiff LERC needs zstd independent of CZI" in patched

    makefile = (tree / "Makefile.linux").read_text(encoding="utf-8")
    assert "$(LIBS)/libczi.a" in makefile
    assert "$(LIBS)/libgeotiff.a" not in makefile.splitlines()[0]
    assert "enabled CZI" in result.stdout


def test_image_service_dockerfile_gates_engine_formats_on_czi() -> None:
    dockerfile = (ROOT / "backend/deepagents_runtime/Dockerfile.imaging").read_text(
        encoding="utf-8"
    )

    assert "ARG ULTRA_IMAGE_PLATFORM=linux/amd64" in dockerfile
    assert "FROM --platform=${ULTRA_IMAGE_PLATFORM} ubuntu:24.04 AS engine-build" in dockerfile
    assert "FROM --platform=${ULTRA_IMAGE_PLATFORM} ubuntu:24.04 AS runtime" in dockerfile
    assert "grep -iq dicom" in dockerfile
    assert "grep -iq czi" in dockerfile
    assert "import libbioimage.libbioimage as bim" in dockerfile
    assert '"czi" in formats' in dockerfile
    assert "disable the out-of-scope geospatial chain and CZI" not in dockerfile


def test_dev_image_service_launcher_can_use_local_libbioimage_source_tree() -> None:
    script = (ROOT / "scripts/run_image_service.sh").read_text(encoding="utf-8")

    assert "PYTHONPATH=/app/src:/build/python/src" in script
    assert "[ -f /build/python/pyproject.toml ]" in script
    assert "pip install --quiet pillow fastapi uvicorn nats-py numpy lxml xarray" in script
    assert "import libbioimage.libbioimage as bim" in script
    assert '"czi" not in formats' in script
