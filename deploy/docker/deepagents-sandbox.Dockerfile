FROM python:3.11-slim

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
# nats/langgraph; yolov5 is the vendored runtime (IPython import made optional); the
# 88MB weights are an immutable, versioned asset. The CLI imports the package and the
# detect subprocess sets YOLOv5_AUTOINSTALL=false (no offline pip).
COPY backend/deepagents_runtime/src/ultra_deepagents/config.py /opt/rarespot/ultra_deepagents/config.py
COPY backend/deepagents_runtime/src/ultra_deepagents/rarespot /opt/rarespot/ultra_deepagents/rarespot
COPY backend/deepagents_runtime/skills/prairie-dog-detection/rarespot_detect.py /opt/rarespot/rarespot_detect.py
COPY third_party/yolov5 /opt/rarespot/yolov5
COPY data/models/yolo/RareSpotWeights.pt /opt/rarespot/RareSpotWeights.pt
RUN : > /opt/rarespot/ultra_deepagents/__init__.py \
    && python -c "import sys; sys.path.insert(0, '/opt/rarespot'); import torchvision; from ultra_deepagents.rarespot.inference import run_rarespot_inference; print('rarespot bake OK', torchvision.__version__)"

WORKDIR /workspace
