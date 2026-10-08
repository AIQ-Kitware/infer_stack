#!/usr/bin/env bash
# Build a minimal context; never send benchmark artifacts or credentials to Docker.
set -euo pipefail
fixture_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$fixture_dir/../../.." && pwd)
build_dir=$(mktemp -d -t infer-stack-kubeai-mock.XXXXXXXX)
trap 'rm -rf -- "$build_dir"' EXIT
mkdir -p "$build_dir/infer_stack" "$build_dir/vllm/entrypoints/openai"
cp -R "$repo_dir/infer_stack/mockserver" "$build_dir/infer_stack/"
printf '%s\n' '"""Standalone test simulator package."""' > "$build_dir/infer_stack/__init__.py"
cp "$fixture_dir/api_server.py" "$build_dir/vllm/entrypoints/openai/"
cp "$fixture_dir/Dockerfile" "$fixture_dir/mock.yaml" "$build_dir/"
docker build -t "${1:-infer-stack-kubeai-mock:local}" "$build_dir"
