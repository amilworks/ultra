#!/usr/bin/env bash
#
# Build fixes for compiling libbioimage (imgcnv) from the public mirror
# (gitlab.com/viqi_public/bioimageconvert), discovered and validated 2026-06-11
# while producing a working imgcnv + libimgcnv.so. Run from the repo root
# (or pass it as $1). Intended for the Linux/CI build (GNU sed).
#
# Fixes 1 and 3 are genuine, upstreamable build bugs; fix 2 is a scope choice.
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
#
# CZI: the public mirror's libCZI static lib is missing symbols (CreateCZIReader,
# CompressionModeToCompressionIdentifier) -> link failure. If that occurs, also
# run the CZI-disable sed printed at the end (we validated locally with CZI off).
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
for f in src/imgcnv.pro src_dylib/libimgcnv.pro libsrc/libbioimg/bioimage.pro; do
  sed -i 's/^CONFIG += lib_libgeotiff.*/#& (disabled: geospatial out of scope)/' "$f"
  sed -i 's/^CONFIG += lib_proj.*/#& (disabled: geospatial out of scope)/' "$f"
done
sed -i 's# \$(LIBS)/libgeotiff.a##' Makefile.linux
echo "disabled geospatial (PROJ/GeoTIFF)"

# 3. Link zstd independently (libtiff LERC needs it).
for f in src/imgcnv.pro src_dylib/libimgcnv.pro; do
  grep -q '^LIBS += -lzstd  # libtiff LERC' "$f" || printf '\nLIBS += -lzstd  # libtiff LERC needs zstd independent of CZI\n' >> "$f"
done
echo "linked zstd independently"

echo
echo "Done. If the libCZI static link fails, also disable CZI:"
echo "  sed -i 's/^CONFIG += lib_libczi/#&/' src/imgcnv.pro src_dylib/libimgcnv.pro libsrc/libbioimg/bioimage.pro"
echo "  sed -i 's# \$(LIBS)/libczi.a##' Makefile.linux   # if buildlibs lists it"
