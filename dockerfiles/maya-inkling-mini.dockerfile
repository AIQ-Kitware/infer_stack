# syntax=docker/dockerfile:1
# Build for a single NVIDIA GPU. No vLLM or PyTorch dependency.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
# Supply an exact upstream commit (see docs/maya-inkling-mini.md).
ARG MAYA_REF
ARG CUDA_ARCHITECTURES=86
ARG BUILD_JOBS=8
LABEL org.opencontainers.image.source="https://github.com/alesha-pro/maya-inkling-mini" \
      org.opencontainers.image.revision="${MAYA_REF}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake ninja-build build-essential python3 python3-venv tini \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir \
    numpy jinja2 regex pyyaml tqdm requests pillow psutil 'jsonschema>=4.18,<5' huggingface_hub
ENV PATH="/opt/venv/bin:${PATH}" PYTHONUNBUFFERED=1

WORKDIR /opt/maya
RUN test -n "$MAYA_REF" \
    && git init \
    && git remote add origin https://github.com/alesha-pro/maya-inkling-mini.git \
    && git fetch --depth 1 origin "$MAYA_REF" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$MAYA_REF"
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DSTRATA_BUILD_TESTS=OFF \
    && cmake --build build --parallel "$BUILD_JOBS" --target maya-ik2

COPY dockerfiles/maya_inkling_launcher.py /usr/local/bin/maya_inkling_launcher.py
ENV HF_HOME=/models/.hf MAYA_MODEL_DIR=/models
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/venv/bin/python", "/usr/local/bin/maya_inkling_launcher.py"]
CMD ["serve"]
