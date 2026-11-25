"""The standalone image build and Compose share one packaged recipe path."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from infer_stack.runtime_images import (
    IMAGE_RECIPES,
    MAYA_IMAGE,
    RECIPE_NAMES,
    RuntimeImageError,
    ensure_managed_image,
)


def test_prepare_image_without_catalog_or_lease(tmp_path, monkeypatch):
    from infer_stack.leasing import compose

    monkeypatch.chdir(tmp_path)
    calls = []
    present = False

    def run(args, **kwargs):
        nonlocal present
        calls.append((args, kwargs))
        if args[1:3] == ['image', 'inspect']:
            if not present:
                raise subprocess.CalledProcessError(1, args)
            return 'sha256:cached'
        assert args[1] == 'build'
        present = True
        return ''

    monkeypatch.setattr(compose, '_default_docker_run', run)
    messages = []
    assert ensure_managed_image('maya-inkling-mini', report=messages.append) == MAYA_IMAGE
    assert ensure_managed_image(MAYA_IMAGE, report=messages.append) == MAYA_IMAGE
    assert [args[1] for args, _ in calls] == ['image', 'build', 'image']
    args, kwargs = calls[1]
    assert args[3:5] == ['--tag', MAYA_IMAGE]
    assert Path(args[-1]) == IMAGE_RECIPES[MAYA_IMAGE]
    assert kwargs['timeout'] == compose.DOCKER_TIMEOUT_BUILD
    assert messages[-1] == f'image already present: {MAYA_IMAGE}'
    assert not list(tmp_path.iterdir())


def test_unknown_recipe_never_calls_docker():
    def run(*args, **kwargs):
        pytest.fail('unknown recipe should not invoke Docker')

    with pytest.raises(RuntimeImageError, match='Unknown bundled image recipe'):
        ensure_managed_image('arbitrary-registry/image:latest', run=run)


def test_cli_image_commands(monkeypatch, capsys):
    from infer_stack.cli import ManageCLI
    from infer_stack import runtime_images

    calls = []
    monkeypatch.setattr(runtime_images, 'ensure_managed_image', calls.append)
    assert ManageCLI.main(argv=['image', 'list']) in (None, 0)
    assert f'maya-inkling-mini  {RECIPE_NAMES["maya-inkling-mini"]}' in capsys.readouterr().out
    assert ManageCLI.main(argv=['image', 'build', 'maya-inkling-mini']) in (None, 0)
    assert calls == ['maya-inkling-mini']
