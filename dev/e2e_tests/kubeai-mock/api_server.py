"""KubeAI's vLLM module entrypoint, backed by the repository simulator."""
import sys

from infer_stack.mockserver.vllm_serve import main

raise SystemExit(main(sys.argv[1:] + ['--mock-config=/app/mock.yaml']))
