# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# ==================================================================

FROM python:3.6.15-buster

ENV DEBIAN_FRONTEND=noninteractive

# Debian buster is archived; remap apt sources so legacy module images stay buildable.
RUN sed -i \
      -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
      -e 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' \
      /etc/apt/sources.list && \
    printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99no-check-valid

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# ===================Module Dependencies============================

RUN pip3 install "opencv-python-headless<4.3"
# https://stackoverflow.com/questions/64650877/install-of-opencv-python-headless-takes-a-long-time 

RUN pip3 install imageio matplotlib numpy Pillow pyparsing python-dateutil scipy 

# ===================Copy Source Code===============================

RUN mkdir /module
WORKDIR /module

COPY src /module/src

# ===============bqapi for python3 Dependencies=====================
# pip install in this exact order
RUN pip3 install six
RUN pip3 install lxml
RUN pip3 install requests==2.18.4
RUN pip3 install requests-toolbelt

# =====================Build Directory Structure====================

COPY PythonScriptWrapper.py /module/
COPY bqapi/ /module/bqapi

# Replace the following line with your {ModuleName}.xml
COPY EdgeDetection.xml /module/EdgeDetection.xml

ENV PATH=/module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=$PYTHONPATH:/module/src
