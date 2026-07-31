# A vLLM-shaped mock inference server, deployable as an infer-stack endpoint.
#
# Presents the same command line and HTTP surface as `vllm serve`, so an
# endpoint can point at this image through `runtime.image` and be acquired,
# converged and released exactly like a real one -- on a host with no GPU
# and no model weights.
#
#   docker build -f dockerfiles/mock-vllm.dockerfile -t aiq-mock-vllm:latest .
#
# Then, in a catalog:
#
#   endpoints:
#     mock-smol:
#       engine: vllm
#       model: smol135
#       runtime:
#         image: aiq-mock-vllm:latest
#         max_model_len: 2048
#
# A fixture can be mounted at /mock/config.yaml to give the simulator an
# answer key; without one it still serves, using --mock-ability.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/infer-stack
COPY pyproject.toml README.md ./
COPY infer_stack ./infer_stack
RUN pip install --no-cache-dir -e .

# vLLM's image takes the model id positionally and serves on 8000; matching
# that is what lets infer-stack treat this as a drop-in.
EXPOSE 8000
ENTRYPOINT ["python", "-m", "infer_stack.mockserver.vllm_serve"]
CMD ["--help"]

HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=12 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"
