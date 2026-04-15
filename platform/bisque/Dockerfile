# Updated by Wahid Sadique Koly on 2025-07-29 to align with the new upgraded codebase.

FROM ubuntu:22.04 AS imgcnv-build

LABEL maintainer="amil@ucsb.edu"

ENV DEBIAN_FRONTEND=noninteractive

# Follow upstream BioImageConvert Ubuntu build instructions so imgcnv is
# compiled natively for the target container architecture instead of copying a
# prebuilt x86_64 archive that fails on arm64 hosts.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    cmake \
    meson \
    qt5-qmake \
    qt5-qmake-bin \
    qconf \
    libfftw3-dev \
    libbz2-dev \
    liblzma-dev \
    libavcodec-dev \
    libavcodec-extra \
    libavdevice-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswresample-dev \
    libswscale-dev \
    libhdf5-dev \
    libgdcm-dev \
    libraw-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libtiff5-dev \
    libtiffxx5 \
    libopenslide-dev \
    libexiv2-dev \
    libgeotiff-dev \
    libproj-dev \
    libpugixml-dev \
    libwebp-dev \
    liblcms2-dev \
    libzstd-dev \
    libeigen3-dev \
    libsqlite3-dev \
    sqlite3 \
    zlib1g-dev \
    libjbig-dev \
    libdeflate-dev \
    liblerc-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libxml2-dev && \
    python3 -m pip install --no-cache-dir --upgrade pip virtualenv pytz python-dateutil future && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG BIOIMAGECONVERT_REPO=https://gitlab.com/viqi_public/bioimageconvert/bioimageconvert.git
ARG BIOIMAGECONVERT_REF=v3.19.0

WORKDIR /opt
RUN git clone --branch "${BIOIMAGECONVERT_REF}" --depth 1 "${BIOIMAGECONVERT_REPO}" bioimageconvert && \
    cd /opt/bioimageconvert && \
    git submodule update --init --recursive

WORKDIR /opt/bioimageconvert
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      echo "Patching BioImageConvert qmake flags for arm64"; \
      find /opt/bioimageconvert \
        \( \
          -path '*/src/imgcnv.pro' -o \
          -path '*/src_dylib/libimgcnv.pro' -o \
          -path '*/src_bimread/bimread.pro' -o \
          -path '*/projects/imgcnv/imgcnv.pro' -o \
          -path '*/projects/libimgcnv/libimgcnv.pro' -o \
          -path '*/libsrc/libbioimg/bioimage.pro' \
        \) \
        -type f \
        -exec sed -i \
          -e 's/-m64//g' \
          -e 's/-msse2//g' \
          -e 's/  */ /g' \
          {} +; \
      for file in /opt/bioimageconvert/src/imgcnv.pro /opt/bioimageconvert/src_dylib/libimgcnv.pro; do \
        sed -i '/LIBS += -lhdf5_serial$/a\    LIBS += -lhdf5_serial_cpp' "$file"; \
      done; \
    fi
RUN make -j"$(nproc)" && \
    test -x /opt/bioimageconvert/bin/imgcnv && \
    test -e /opt/bioimageconvert/bin/libimgcnv.so

FROM ubuntu:22.04

LABEL maintainer="amil@ucsb.edu"
LABEL build_date="2025-03-01"

# Set noninteractive frontend to avoid hanging
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Install Bisque System"

########################################################################################
# Linux System Package Installs for BisQue (Ubuntu 22.04)
########################################################################################

# Consolidated package installation for M8.2b.
# Includes core runtime/dev packages, Docker engine CLI, and imgcnv runtime deps.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    wget \
    gnupg \
    lsb-release \
    && apt-add-repository multiverse \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-scipy \
    python3-openslide \
    python3-minimal \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libmysqlclient-dev \
    postgresql \
    postgresql-client \
    graphviz \
    libgraphviz-dev \
    openslide-tools \
    libfftw3-dev \
    libgdcm3.0 \
    xvfb \
    firefox \
    tightvncserver \
    x11vnc \
    xfonts-base \
    ffmpeg \
    git \
    locales \
    less \
    libasound2 \
    libasound2-data \
    libblas3 \
    libbz2-1.0 \
    libgdbm-compat4 \
    libgdk-pixbuf2.0-0 \
    libgdk-pixbuf2.0-common \
    libgfortran5 \
    libglib2.0-0 \
    libglib2.0-data \
    libblosc1 \
    libgomp1 \
    libice6 \
    libjbig0 \
    libjpeg-turbo8 \
    liblapack3 \
    liblzo2-2 \
    libmagic1 \
    libogg0 \
    libopenslide0 \
    libopenslide-dev \
    liborc-0.4-0 \
    libpixman-1-0 \
    libpng16-16 \
    libpq5 \
    libsm6 \
    libsqlite3-0 \
    libstdc++6 \
    libtheora0 \
    libtiff5-dev \
    libx11-6 \
    libx11-data \
    libxau6 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shm0 \
    libxdmcp6 \
    libxext6 \
    libxml2 \
    libxrender1 \
    libxslt1.1 \
    libxvidcore4 \
    libfreeimage3 \
    mercurial \
    openjdk-8-jdk \
    vim \
    sudo \
    libjpeg-dev \
    libjpeg8-dev \
    docker.io \
    libavformat58 \
    libavcodec58 \
    libswscale5 \
    libavutil56 \
    libhdf5-103 \
    libhdf5-cpp-103 \
    libfftw3-3 \
    liblzma5 \
    libraw20 \
    libtiff5 \
    libtiffxx5 \
    libopenjp2-7 \
    libexiv2-27 \
    libwebp7 \
    liblcms2-2 \
    libzstd1 \
    zlib1g \
    libdeflate0 \
    liblerc3 \
    && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Fix ImarisConvert library dependencies - create symlink for older JPEG library version
RUN set -eux; \
    libdir="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)"; \
    ln -sf "${libdir}/libjpeg.so.8" "${libdir}/libjpeg.so.62"

# Set Java environment for bioformats
# `/usr/lib/jvm/default-java` resolves correctly on both amd64 and arm64.
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
RUN locale

########################################################################################
# Install Image Converter for BisQue (Ubuntu 22.04)
########################################################################################

WORKDIR /var/opt

# imgcnv runtime dependencies are installed in the consolidated package layer
# above. Copy a native build from the upstream BioImageConvert source tree so
# the binary matches the container architecture. Recreate the shared-library
# symlink layout explicitly because wildcard COPY flattens symlinks into plain
# files, which prevents later BisQue setup steps from resolving libimgcnv.so.
COPY --from=imgcnv-build /opt/bioimageconvert/bin/imgcnv /usr/local/bin/
COPY --from=imgcnv-build /opt/bioimageconvert/bin/libimgcnv.so* /tmp/imgcnv-libs/
ENV LD_LIBRARY_PATH=/usr/local/lib
RUN set -eux; \
    real_lib="$(find /tmp/imgcnv-libs -maxdepth 1 -type f -name 'libimgcnv.so.*' -printf '%f\n' | sort -V | tail -n1)"; \
    if [ -z "${real_lib}" ]; then \
      echo "imgcnv shared library was not copied from the build stage" >&2; \
      exit 1; \
    fi; \
    install -m 0755 "/tmp/imgcnv-libs/${real_lib}" "/usr/local/lib/${real_lib}"; \
    base_lib="$(printf '%s' "${real_lib}" | awk -F. '{print $1"."$2}')"; \
    abi_lib="$(printf '%s' "${real_lib}" | awk -F. '{print $1"."$2"."$3}')"; \
    abi_minor_lib="$(printf '%s' "${real_lib}" | awk -F. '{print $1"."$2"."$3"."$4}')"; \
    ln -sf "${real_lib}" "/usr/local/lib/${base_lib}"; \
    ln -sf "${real_lib}" "/usr/local/lib/${abi_lib}"; \
    if [ -n "${abi_minor_lib}" ] && [ "${abi_minor_lib}" != "${abi_lib}" ]; then \
      ln -sf "${real_lib}" "/usr/local/lib/${abi_minor_lib}"; \
    fi; \
    echo /usr/local/lib > /etc/ld.so.conf.d/imgcnv.conf; \
    ldconfig; \
    imgcnv -v; \
    python3 - <<'PY'
import ctypes
ctypes.cdll.LoadLibrary("libimgcnv.so")
print("libimgcnv ok")
PY

########################################################################################
# COPY BASH Scripts for BisQue
#   - Set workdir early  as may wipe out contents
########################################################################################

WORKDIR /source
COPY run-bisque.sh bq-admin-setup.sh virtualenv.sh /builder/
COPY start-bisque.sh /builder/start-scripts.d/R50-start-bisque.sh
COPY builder/ /builder/build-scripts.d/
COPY boot/ /builder/boot-scripts.d/

########################################################################################
# RUN BASH Scripts for BisQue
#   - Install virtual ENV
#   - Set biodev pip index and install pip dependencies
########################################################################################

RUN /builder/virtualenv.sh
# ENV PY_INDEX=https://vat.example.com/py/bisque/xenial/+simple

# Set custom Python package index and configure pip
# ENV PY_INDEX=https://vat.example.com/py/bisque/xenial/+simple
# RUN mkdir -p /root/.pip && echo "\
# [global]\n\
# index-url = $PY_INDEX\n\
# trusted-host = biodev.example.com\n" > /root/.pip/pip.conf

# # Install certifi for updated CA certificates
# RUN pip install certif
# ENV SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

########################################################################################
# COPY Source Code
#   - Install requirements.txt first to install all python deps
#   - Added at the end for easy updates to source code
########################################################################################

COPY source/ /source/
RUN mkdir -p /source/data /source/staging /source/reports /source/public
RUN /builder/run-bisque.sh build

# Build-time guard for M8.1: keep duplicate install disabled by default.
ARG BISQUE_BUILD_DOUBLE_INSTALL=0
ENV BISQUE_BUILD_DOUBLE_INSTALL=${BISQUE_BUILD_DOUBLE_INSTALL}
# Keep post-install config generation enabled unless explicitly disabled.
ARG BISQUE_BUILD_POST_FULLCONFIG=1
ENV BISQUE_BUILD_POST_FULLCONFIG=${BISQUE_BUILD_POST_FULLCONFIG}
RUN /builder/bq-admin-setup.sh

########################################################################################
# Fix: bioformats and third-party tools integration
########################################################################################

# Ensure bioformats tools are properly configured
RUN if [ -f /usr/lib/bisque/bin/bioformats_package.jar ]; then \
    echo "Bioformats JAR found, testing functionality..." && \
    java -cp /usr/lib/bisque/bin/bioformats_package.jar loci.formats.tools.ImageInfo 2>/dev/null || echo "Bioformats JAR needs runtime configuration"; \
    fi

# Create environment script for bioformats tools
RUN echo '#!/bin/bash\n\
    export BF_JAR_DIR=/usr/lib/bisque/bin\n\
    export JAVA_HOME=/usr/lib/jvm/default-java\n\
    export PATH=$JAVA_HOME/bin:$PATH\n' > /etc/bioformats-env.sh && \
    chmod +x /etc/bioformats-env.sh

# Test ImarisConvert functionality
RUN if [ -f /usr/lib/bisque/bin/ImarisConvert ]; then \
    echo "Testing ImarisConvert..." && \
    /usr/lib/bisque/bin/ImarisConvert --help >/dev/null 2>&1 && echo "ImarisConvert working" || echo "ImarisConvert needs additional libraries"; \
    fi

# Verify OpenSlide installation
RUN python3 -c "import openslide; print('OpenSlide version:', openslide.__version__)" && \
    echo "OpenSlide verification complete"

# Create proper wrapper scripts for third-party tools in system PATH
RUN echo '#!/bin/sh\nSUB=ImarisConvert_glibc2.11\nexport LD_LIBRARY_PATH="/usr/lib/bisque/bin/$SUB"\nexec /usr/lib/bisque/bin/$SUB/ImarisConvert "$@"' > /usr/local/bin/ImarisConvert && \
    echo '#!/usr/bin/env bash\nBF_PROG=loci.formats.tools.ImageInfo /usr/lib/bisque/bin/bf.sh "$@"' > /usr/local/bin/showinf && \
    echo '#!/usr/bin/env bash\nBF_PROG=loci.formats.tools.ImageConverter /usr/lib/bisque/bin/bf.sh "$@"' > /usr/local/bin/bfconvert && \
    chmod +x /usr/local/bin/ImarisConvert /usr/local/bin/showinf /usr/local/bin/bfconvert && \
    echo "Third-party tools wrapper scripts created"

# Set container environment markers for BisQue setup scripts
ENV DOCKER_CONTAINER=1
ENV BQ_CONTAINER_BUILD=1

########################################################################################
# Install Minio and Argo CLI
########################################################################################

# RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc &&  mv mc /usr/bin/mc

# COPY config.json /root/.mc/config.json

# Download the binary
RUN curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.8/argo-linux-amd64.gz

# Unzip
RUN gunzip argo-linux-amd64.gz

# Make binary executable
RUN chmod +x argo-linux-amd64 && mv ./argo-linux-amd64 /usr/local/bin/argo

########################################################################################

ENTRYPOINT ["/builder/run-bisque.sh"]

CMD [ "bootstrap","start"]
