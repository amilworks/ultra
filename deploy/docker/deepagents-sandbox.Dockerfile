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

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
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

WORKDIR /workspace
