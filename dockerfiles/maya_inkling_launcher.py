"""Prepare the pinned Maya checkout and launch its OpenAI-compatible server.

The runtime contract is supplied by infer-stack's generic command/env/mounts.
Weights live on the persistent mount; the generated config and tokenizer are
container-local. Keep Maya's model-specific flags in its own upstream configs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/opt/maya')
MODEL_REPO = 'unsloth/Inkling-2-Mini-GGUF'
MODEL_FILE = 'Inkling-2-Mini-UD-Q4_K_XL.gguf'
DRAFT_REPO = 'thinkingmachines/Inkling-2-Mini'
DRAFT_FILE = 'mtp.safetensors'


def settings(env):
    """Validate the launcher contract before downloading or allocating VRAM."""
    draft = env.get('MAYA_DSPARK', '1')
    if draft not in {'0', '1'}:
        raise ValueError('MAYA_DSPARK must be 0 or 1')
    draft = draft == '1'
    context = int(env.get('MAYA_MAX_CONTEXT', '196608'))
    limit = 262144 if draft else 344064
    if not 0 < context <= limit:
        raise ValueError(f'MAYA_MAX_CONTEXT must be in 1..{limit} (DSpark={draft})')
    port = int(env.get('PORT', '8000'))
    if not 0 < port <= 65535:
        raise ValueError('PORT must be in 1..65535')
    name = env.get('MAYA_MODEL_NAME', 'inkling-2-mini-maya')
    if not name.strip():
        raise ValueError('MAYA_MODEL_NAME must not be empty')
    model_dir = Path(env.get('MAYA_MODEL_DIR', '/models')).resolve()
    return draft, context, port, name, model_dir


def ensure_file(repo, filename, model_dir):
    """HF's local-dir downloader handles resumable downloads and file locks.

    Existing complete files can be used offline. To replace weights, explicitly
    remove them (or choose a new model directory), rather than updating silently.
    """
    path = model_dir / filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    from huggingface_hub import hf_hub_download

    print(f'Maya: downloading {repo}/{filename} into {model_dir}', flush=True)
    return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir))


def make_config(root, model_dir, *, draft, context, name):
    """Use upstream's tested FP8/tuning configuration, minus CPU vision."""
    configs = root / 'docs/inkling2-mini'
    profile = 'ik2-mini.json' if draft else 'ik2-mini-text.json'
    config = json.loads((configs / profile).read_text())
    config.pop('vision', None)
    config['exe'] = str(root / 'build/maya-ik2')
    config['args'] = [
        '--model', str(model_dir / MODEL_FILE), '--max-context', str(context),
    ]
    if draft:
        config['args'] += ['--draft', str(model_dir / DRAFT_FILE)]
    config['tokenizer'] = str(root / 'ik2/tokenizer')
    config['model_name'] = name
    # The lease controller owns engine lifecycle; do not offer Maya's separate
    # model-management API or let it unload weights behind the controller.
    config['model_management'] = False
    config['idle_unload_s'] = 0
    # Upstream defaults to high; allow each OpenAI request to override it.
    config['reasoning_effort'] = 'high'
    return config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (['serve'], ['prepare']):
        raise SystemExit('usage: maya_inkling_launcher.py {serve|prepare}')
    try:
        draft, context, port, name, model_dir = settings(os.environ)
    except ValueError as exc:
        raise SystemExit(f'Maya: {exc}') from exc
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = ensure_file(MODEL_REPO, MODEL_FILE, model_dir)
    if draft:
        ensure_file(DRAFT_REPO, DRAFT_FILE, model_dir)
    if argv == ['prepare']:
        return
    os.chdir(ROOT)
    subprocess.run([
        sys.executable, 'tools/strata_tokenizer.py', '--gguf', str(weights),
        '--out', 'ik2', '--chat-template', 'tools/ik2/chat_template.jinja', '--check',
    ], check=True)
    config = make_config(ROOT, model_dir, draft=draft, context=context, name=name)
    config_path = ROOT / 'infer-stack-ik2.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Maya: {name}, context={context}, DSpark={draft}, port={port}', flush=True)
    os.execv(sys.executable, [sys.executable, 'serve/server.py', '--engine', 'maya',
                            '--config', str(config_path), '--host', '0.0.0.0',
                            '--port', str(port)])


if __name__ == '__main__':
    main()
