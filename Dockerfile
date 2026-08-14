FROM docker.io/rocm/onnxruntime:rocm7.2.4_ub24.04_ort1.23_torch2.10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV HSA_OVERRIDE_GFX_VERSION="10.3.0"
ENV YOLO_AUTOINSTALL=false
ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages:$PYTHONPATH

RUN apt update && apt install -y locales fish \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN /opt/venv/bin/pip install onnx onnxslim opencv-python ultralytics boxmot rerun_sdk

WORKDIR /workspaces/multiview_3d_tracking
