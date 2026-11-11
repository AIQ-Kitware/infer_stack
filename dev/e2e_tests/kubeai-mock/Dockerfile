FROM python:3.13-slim
RUN pip install --no-cache-dir pyyaml ubelt
COPY infer_stack /app/infer_stack
COPY vllm /app/vllm
COPY mock.yaml /app/mock.yaml
WORKDIR /app
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
