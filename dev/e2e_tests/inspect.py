#!/usr/bin/env python3
"""Tiny rendered-compose inspector for the e2e harness assertions.

Usage: ``python3 inspect.py <docker-compose.yml>``

Prints flat ``key: value`` lines the shell tiers assert on with ``expect_out`` /
``expect_re`` — keeping the YAML parsing out of brittle bash. Covers the front
door wiring (which services exist, how Open WebUI is connected) and each Ollama
daemon's host port + GPU pinning, which is what the new leasing features and the
GPU-pin bug fix need to be checked on real hardware.
"""

from __future__ import annotations

import sys

import yaml


def main() -> None:
    d = yaml.safe_load(open(sys.argv[1])) or {}
    svcs = d.get('services', {}) or {}
    print('services:', ' '.join(sorted(svcs)))
    print('has litellm:', 'litellm' in svcs)
    print('has open-webui:', 'open-webui' in svcs)

    ow_svc = svcs.get('open-webui') or {}
    ow_ports = ow_svc.get('ports') or []
    if ow_ports:
        print('open-webui host_port:', str(ow_ports[0]).split(':')[0])
    ow = ow_svc.get('environment', {}) or {}
    for k in (
        'ENABLE_OPENAI_API',
        'OPENAI_API_BASE_URL',
        'ENABLE_OLLAMA_API',
        'OLLAMA_BASE_URL',
    ):
        print(f'open-webui {k}: {ow.get(k)}')

    for name, svc in sorted(svcs.items()):
        if not name.startswith('ollama-'):
            continue
        ports = svc.get('ports') or ['?:?']
        host_port = str(ports[0]).split(':')[0]
        env = svc.get('environment', {}) or {}
        devices = (
            (((svc.get('deploy') or {}).get('resources') or {}).get(
                'reservations', {}
            ) or {}).get('devices')
            or [{}]
        )
        device_ids = devices[0].get('device_ids')
        print(f'{name} host_port: {host_port}')
        print(f'{name} device_ids: {device_ids}')
        print(f'{name} CUDA_VISIBLE_DEVICES: {env.get("CUDA_VISIBLE_DEVICES")}')


if __name__ == '__main__':
    main()
