#!/usr/bin/env bash
#
# Build fixes for compiling libbioimage (imgcnv) from the public mirror
# (gitlab.com/viqi_public/bioimageconvert), discovered and validated 2026-06-11
# while producing a working imgcnv + libimgcnv.so. Run from the repo root
# (or pass it as $1). Intended for the Linux/CI build (GNU sed).
#
# Fixes 1, 3, and 4 are genuine, upstreamable build bugs; fix 2 is a scope choice.
#
#   1. GeoTIFF is compiled unconditionally (bim_geotiff_parse.cpp lives in the
#      always-on libtiff SOURCES block) and its callers in bim_tiff_format_io.cpp
#      are NOT #ifdef-guarded, so the engine hard-requires the geospatial libs --
#      but the qmake Makefile.linux PROJ recipe is broken for the pinned PROJ
#      (calls a missing ./autogen.sh). We guard the file with no-op stubs so it
#      builds with geospatial disabled.
#   2. Disable the out-of-scope geospatial (PROJ/GeoTIFF) format + its lib build.
#   3. `-lzstd` is bundled only inside the CZI link block, but libtiff's LERC
#      needs ZSTD_compress regardless; link it independently.
#   4. Keep CZI enabled. The public mirror's libCZI static archive and the
#      standalone JPEG-XR archives both vendor the same JPEG-XR glue symbols on
#      Linux. We tolerate those duplicate static symbols at final link time so
#      both .czi and .jxr readers stay registered.
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"

# 1. Guard bim_geotiff_parse.cpp with no-op stubs when BIM_GEOTIFF_FORMAT is off.
python3 - <<'PY'
p = "libsrc/libbioimg/formats/tiff/bim_geotiff_parse.cpp"
s = open(p).read()
if "#ifndef BIM_GEOTIFF_FORMAT" not in s:
    stub = (
        '#ifndef BIM_GEOTIFF_FORMAT\n'
        '#include "bim_geotiff_parse.h"\n'
        'void geotiff_append_metadata(bim::FormatHandle *, bim::TagMap *) {}\n'
        'bool isGeoTiff(TIFF *) { return false; }\n'
        'int GTIFFromBuffer(const std::vector<char> &, TIFF *) { return 1; }\n'
        'int BufferFromGTIF(TIFF *, std::vector<char> &) { return 0; }\n'
        '#else\n\n#include <sstream>'
    )
    s = s.replace("#include <sstream>", stub, 1) + "\n#endif // BIM_GEOTIFF_FORMAT\n"
    open(p, "w").write(s)
    print("patched bim_geotiff_parse.cpp")
PY

# 2. Disable geospatial config across the three project files + drop the
#    broken libgeotiff.a (and its libproj.a prereq) from buildlibs.
python3 - <<'PY'
from pathlib import Path
import re

project_files = [
    Path("src/imgcnv.pro"),
    Path("src_dylib/libimgcnv.pro"),
    Path("libsrc/libbioimg/bioimage.pro"),
]

for path in project_files:
    s = path.read_text()
    for config in ("lib_libgeotiff", "lib_proj"):
        s = re.sub(
            rf"^CONFIG \+= {config}.*$",
            lambda m: f"#{m.group(0)} (disabled: geospatial out of scope)",
            s,
            flags=re.MULTILINE,
        )
    path.write_text(s)

makefile = Path("Makefile.linux")
s = makefile.read_text()
s = s.replace(" $(LIBS)/libgeotiff.a", "")
makefile.write_text(s)
PY
echo "disabled geospatial (PROJ/GeoTIFF)"

# 3. Link zstd independently (libtiff LERC needs it).
python3 - <<'PY'
from pathlib import Path

line = "LIBS += -lzstd  # libtiff LERC needs zstd independent of CZI"
for path in [Path("src/imgcnv.pro"), Path("src_dylib/libimgcnv.pro")]:
    s = path.read_text()
    if line not in s:
        s = s.rstrip() + f"\n\n{line}\n"
        path.write_text(s)
PY
echo "linked zstd independently"

# 4. Keep CZI enabled and tolerate duplicated static JPEG-XR glue symbols at
#    final link. This preserves the standalone .jxr reader and the libCZI reader.
python3 - <<'PY'
from pathlib import Path
import re

project_files = [
    Path("src/imgcnv.pro"),
    Path("src_dylib/libimgcnv.pro"),
    Path("libsrc/libbioimg/bioimage.pro"),
]
for path in project_files:
    s = path.read_text()
    s = re.sub(
        r"^#(CONFIG \+= lib_libczi).*$",
        r"\1",
        s,
        flags=re.MULTILINE,
    )
    path.write_text(s)

linker_flag = "QMAKE_LFLAGS += -Wl,--allow-multiple-definition"
for path in [Path("src/imgcnv.pro"), Path("src_dylib/libimgcnv.pro")]:
    s = path.read_text()
    if linker_flag not in s:
        s = s.rstrip() + f"\n\n{linker_flag}\n"
        path.write_text(s)
PY
echo "enabled CZI with duplicate JPEG-XR static symbol link guard"
