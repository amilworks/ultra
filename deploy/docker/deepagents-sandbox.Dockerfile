# --- three.js bundle builder (build-time network only; nothing from this
# stage runs in the sandbox). Produces ONE classic-script IIFE exposing the
# global `THREE` — including OrbitControls, which nearly every scientific
# scene needs — so reports can inline bespoke 3D under the reading canvas's
# CSP (inline scripts allowed, external fetches blocked) with zero imports,
# no import maps, and no CSP loosening. Versions pinned; MIT license ships
# beside the bundle. node:22-alpine digest resolved 2026-08-02.
FROM node:22-alpine@sha256:c610fcdfb1d5b4740dd70c284ed3cb16bb857e0f7166196e36a5501df7a3aa32 AS threejs-bundle
WORKDIR /build
RUN npm install --no-audit --no-fund three@0.172.0 esbuild@0.24.2 \
    && printf '%s\n' \
        "export * from 'three';" \
        "export { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';" \
        > entry.js \
    && npx esbuild entry.js --bundle --minify --format=iife --global-name=THREE \
        --legal-comments=none --outfile=three.iife.min.js \
    && cp node_modules/three/LICENSE THREE_LICENSE \
    && node -e "const s=require('fs').statSync('three.iife.min.js').size; if (s < 400000 || s > 1400000) { throw new Error('three bundle size out of expected range: ' + s); } console.log('three.iife.min.js', s, 'bytes')"

# Multi-architecture python:3.11-slim index resolved 2026-07-11
# (3.11.15-slim-trixie). Revisions require an explicit reviewed digest update.
FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        dcm2niix \
        dcmtk \
        default-jre-headless \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libopenslide0 \
        libvips42 \
        openslide-tools \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pin torch + torchvision as a matched CUDA (cu130) pair, BEFORE the imaging stack
# below so it resolves against this exact torch. torchvision is a hard YOLOv5 dep
# (utils/general.py: torchvision.ops.nms) and is otherwise absent; without an
# explicit pin a later imaging-lib bump silently moves torch and breaks NMS parity.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.12.1+cu130" \
        "torchvision==0.27.1+cu130"

RUN python -m pip install --no-cache-dir \
        SimpleITK \
        bioio \
        bioio-bioformats \
        bioio-czi \
        bioio-dv \
        bioio-imageio \
        bioio-lif \
        bioio-nd2 \
        bioio-ome-tiff \
        bioio-ome-zarr \
        bioio-sldy \
        bioio-tiff-glob \
        bioio-tifffile \
        connected-components-3d \
        "dask[array]" \
        dicom2nifti \
        dipy \
        h5py \
        highdicom \
        imageio \
        imagecodecs \
        itk \
        matplotlib \
        monai \
        mrcfile \
        networkx \
        nibabel \
        nilearn \
        numpy==1.26.4 \
        ome-types \
        ome-zarr \
        opencv-python-headless==4.10.0.84 \
        openslide-python \
        "pandas<3" \
        pillow \
        pyarrow \
        pydicom \
        pylibjpeg \
        pylibjpeg-libjpeg \
        pylibjpeg-openjpeg \
        pylibjpeg-rle \
        pynrrd \
        pyvips \
        roifile \
        scikit-image \
        scikit-learn \
        scipy \
        seaborn \
        tifffile \
        torchio \
        xarray \
        zarr \
    && python -m pip install --no-cache-dir --no-deps rt-utils

# Bake the RareSpot prairie-dog detection assets so the prairie-dog-detection Skill
# runs entirely in the --network none sandbox (which mounts only /workspace+/outputs).
# A minimal ultra_deepagents tree (empty __init__, standalone config.py, the rarespot
# package) preserves the ultra_deepagents.rarespot import path WITHOUT dragging
# nats/langgraph; yolov5 is the vendored runtime (IPython import made optional). The
# 88MB weights are lab data (data/ is untracked except a .gitkeep), so they bake in
# only when present in the build context: production/lab builds have them, a clean
# clone builds a weights-less sandbox and the Skill fails fast at runtime instead of
# the image failing to build. The CLI imports the package and the detect subprocess
# sets YOLOv5_AUTOINSTALL=false (no offline pip).
COPY backend/deepagents_runtime/src/ultra_deepagents/config.py /opt/rarespot/ultra_deepagents/config.py
COPY backend/deepagents_runtime/src/ultra_deepagents/rarespot /opt/rarespot/ultra_deepagents/rarespot
COPY backend/deepagents_runtime/skills/prairie-dog-detection/rarespot_detect.py /opt/rarespot/rarespot_detect.py
COPY third_party/yolov5 /opt/rarespot/yolov5
COPY data/models/yolo/ /opt/rarespot/weights-src/
RUN if [ -f /opt/rarespot/weights-src/RareSpotWeights.pt ]; then \
        mv /opt/rarespot/weights-src/RareSpotWeights.pt /opt/rarespot/RareSpotWeights.pt \
        && echo "RareSpot weights baked into the sandbox"; \
    else \
        echo >&2 "NOTE: data/models/yolo/RareSpotWeights.pt not in build context — sandbox built WITHOUT RareSpot weights (prairie-dog detection unavailable at runtime)"; \
    fi \
    && rm -rf /opt/rarespot/weights-src
RUN : > /opt/rarespot/ultra_deepagents/__init__.py \
    && python -c "import sys; sys.path.insert(0, '/opt/rarespot'); import torchvision; from ultra_deepagents.rarespot.inference import run_rarespot_inference; print('rarespot bake OK', torchvision.__version__)"

# Bake the medical-volume-slices CLI: the medical-volume-slices Skill runs it to turn a
# 3D/4D volume (CT/MRI/fMRI/DICOM/tractography) into VLM-optimal 2D slices. Uses the imaging
# stack already installed above (nibabel, SimpleITK, dipy, pillow) — no new deps.
COPY backend/deepagents_runtime/skills/medical-volume-slices/volume_slices.py /opt/medvol/volume_slices.py
RUN python -c "import nibabel, SimpleITK, PIL, numpy; import ast; ast.parse(open('/opt/medvol/volume_slices.py').read()); print('medvol bake OK')"

# Computational-biology stack for the main interpreter: single-cell analysis (scanpy/
# anndata), SOTA community detection (Leiden via leidenalg + python-igraph), ligand-receptor
# inference (liana), and graph neural networks (torch-geometric, resolved against the exact
# cu130 torch above). The install RE-ASSERTS the imaging stack's load-bearing pins so the
# resolver cannot silently move them — numpy==1.26.4 (yolov5/RareSpot NMS + np.int parity),
# zarr>=3.1 and dask>=2026.6 (the OME-Zarr imaging path via bioio-ome-zarr). This exact set
# was empirically verified to leave numpy/zarr/dask/torch/torchvision/opencv/bioio-ome-zarr
# untouched. squidpy is deliberately excluded here: its working builds need numpy>=2 while
# numpy==1.26.4 forces an old build that breaks on dask 2026 — it lives in the isolated
# /opt/biograph conda env below instead.
RUN python -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "zarr>=3.1" \
        "dask>=2026.6" \
        anndata \
        scanpy \
        leidenalg \
        python-igraph \
        liana \
        torch-geometric \
    && python -c "import numpy, zarr, dask, torch, torchvision, cv2, bioio_ome_zarr; assert numpy.__version__ == '1.26.4', numpy.__version__; print('bio main-env no-regression OK: numpy', numpy.__version__, 'zarr', zarr.__version__, 'dask', dask.__version__)" \
    && python -c "import scanpy, anndata, leidenalg, igraph, liana; from torch_geometric.nn import GCNConv; print('bio main-env imports OK: scanpy', scanpy.__version__, 'liana', liana.__version__)"

# OCR + video toolchain for the ocr-reader subagent. Deliberately a LATE layer:
# adding it to the base apt block would invalidate the torch/conda cache chain
# and force a full rebuild for a 60MB addition. Verified at build time so a
# broken toolchain fails the build, not a run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir pytesseract \
    && tesseract --version | head -1 \
    && ffmpeg -version | head -1 \
    && python -c "import pytesseract; print('ocr toolchain OK:', pytesseract.get_tesseract_version())"

# graph-tool (Bayesian nested stochastic block models + minimum-description-length model
# selection — the principled test for whether detected community structure is real signal or
# an artifact of the algorithm) and squidpy (spatial neighbor graphs, neighborhood
# enrichment, Moran's I, co-occurrence) cannot live in the pinned main env: graph-tool is a
# C++/Boost library with no pip wheel, and squidpy's working builds require numpy>=2. They
# live together in a self-contained conda-forge environment at /opt/biograph (numpy 2,
# fully isolated from the default interpreter). micromamba is arch-detected so the image
# builds on both arm64 and amd64. The computational-biology Skill points the agent at
# /opt/biograph/bin/python for SBM and spatial-graph work; everything else stays in the
# default interpreter. /opt/biograph is NOT placed on PATH so it never shadows the main env.
RUN set -eux; \
    export MAMBA_ROOT_PREFIX=/opt/mamba; \
    arch="$(uname -m)"; \
    case "$arch" in \
        aarch64 | arm64) mamba_platform=linux-aarch64 ;; \
        x86_64 | amd64) mamba_platform=linux-64 ;; \
        *) echo "unsupported arch for micromamba: $arch" >&2; exit 1 ;; \
    esac; \
    curl -Ls "https://micro.mamba.pm/api/micromamba/${mamba_platform}/latest" | tar -xj -C /usr/local bin/micromamba; \
    micromamba create -y -p /opt/biograph -c conda-forge \
        "python=3.11" \
        graph-tool \
        squidpy \
        scanpy \
        anndata \
        leidenalg \
        python-igraph \
        matplotlib; \
    micromamba clean --all --yes; \
    rm -rf /opt/mamba/pkgs /usr/local/bin/micromamba; \
    /opt/biograph/bin/python -c "import graph_tool.all as gt, squidpy, scanpy, anndata, leidenalg, igraph; print('biograph env OK: graph-tool', gt.__version__.split()[0], '| squidpy', squidpy.__version__, '| scanpy', scanpy.__version__)"

# Computational-ecology / geospatial stack for the main interpreter: vector + raster IO and
# CRS handling (geopandas via the pyogrio engine, rasterio, rioxarray, shapely, pyproj),
# spatial statistics (PySAL — libpysal/esda/pointpats/spreg/spopt: Moran's I, LISA, Getis-Ord,
# spatial regression, point patterns), landscape/fragmentation metrics (pylandstats), zonal
# stats (rasterstats), terrain + hydrology (xarray-spatial + pysheds), movement ecology
# (movingpandas), and cartographic helpers (contextily, mapclassify). The install RE-ASSERTS
# the imaging pins so the resolver can't move them — empirically verified to leave
# numpy==1.26.4 / zarr / dask / torch / opencv / bioio-ome-zarr intact (no isolated env
# needed). Two packages are deliberately OMITTED and must not be reached for: fiona (its sdist
# needs GDAL dev headers absent from this image — the pyogrio engine replaces it) and richdem
# (no py3.11 wheel; builds against the removed CPython PyFrameObject C-API — pysheds +
# xarray-spatial replace its flow-routing). The geo wheels bundle GDAL/GEOS/PROJ (no apt GDAL).
RUN python -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "zarr>=3.1" \
        "dask>=2026.6" \
        geopandas \
        pyogrio \
        shapely \
        pyproj \
        rasterio \
        rioxarray \
        libpysal \
        esda \
        pointpats \
        spreg \
        spopt \
        pylandstats \
        rasterstats \
        xarray-spatial \
        pysheds \
        movingpandas \
        contextily \
        mapclassify \
    && python -c "import numpy, zarr, dask, torch, torchvision, cv2, bioio_ome_zarr; assert numpy.__version__ == '1.26.4', numpy.__version__; print('geo main-env no-regression OK: numpy', numpy.__version__, 'zarr', zarr.__version__)" \
    && python -c "import geopandas, pyogrio, shapely, pyproj, rasterio, rioxarray, libpysal, esda, pointpats, spreg, spopt, pylandstats, rasterstats, xrspatial, pysheds, movingpandas, contextily, mapclassify; print('geo main-env imports OK: geopandas', geopandas.__version__, 'esda', esda.__version__)"

# numba/llvmlite mis-detects the CPU's feature set on some virtualized hosts (notably the arm64
# Docker VM on Apple Silicon), emitting an illegal instruction (SIGILL) on the FIRST @njit
# Headless Chromium for the report-preview skill: the agent renders its own
# outputs/report.html to a screenshot + console log (skills/report-preview)
# before delivery, catching broken figure refs and script errors the reader
# would otherwise hit. Browsers download at BUILD time (runtime stays
# network-isolated; the renderer drives file:// only and aborts every
# outbound request). Verified end-to-end in this image lineage before this
# line landed: pip playwright + install --with-deps chromium, then
# render_report.py returns 0/2/3 semantics with a correct findings log.
RUN python -m pip install --no-cache-dir playwright \
    && playwright install --with-deps chromium

# Report-authoring extras (both offline-safe for the reader's sandboxed
# canvas, which blocks all network):
# - latex2mathml: LaTeX -> native MathML. Chromium (reader AND the headless
#   preview) renders MathML core natively, so equations need zero vendored
#   JS/CSS/fonts — this replaces the hand-rolled HTML/CSS math the first real
#   report resorted to.
# - plotly: interactive figures inlined into self-contained reports via
#   include_plotlyjs="inline" (~3.5MB of inline JS; the canvas CSP allows
#   inline scripts). Static figure exports stay on matplotlib — deliberately
#   NO kaleido, which would bundle a second private Chromium.
RUN python -m pip install --no-cache-dir latex2mathml plotly

# Bespoke 3D for reports: the vendored three.js IIFE (global THREE, incl.
# THREE.OrbitControls) built in the stage above. Reports inline the FILE
# CONTENTS in a classic <script> — never a CDN tag, which is a silently dead
# tag under the reading canvas's CSP. ~0.7MB minified; MIT license alongside.
COPY --from=threejs-bundle /build/three.iife.min.js /opt/report-assets/three.iife.min.js
COPY --from=threejs-bundle /build/THREE_LICENSE /opt/report-assets/THREE_LICENSE

# compile — which hard-crashes every numba-JIT path: umap / scanpy neighbors (bio),
# xarray-spatial terrain (ecology), esda conditional permutations. Forcing
# generic baseline-ISA codegen makes the JIT correct on every host; the modest loss of
# CPU-specific vectorization is a worthwhile price for not crashing. Set late so it does not
# invalidate the expensive install layers above (it only affects runtime code execution).
ENV NUMBA_CPU_NAME=generic \
    NUMBA_CPU_FEATURES=""

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Ultra Deep Agents scientific sandbox" \
      org.opencontainers.image.description="Network-isolated production code-execution environment" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/amilworks/ultra"

WORKDIR /workspace
