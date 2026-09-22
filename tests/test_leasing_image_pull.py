"""A missing image is pulled before `up`, with progress, not silently inside it.

Reported from a real host: the first acquire of a model on a new image sat for
many minutes with no output, because `docker compose up` pulled it with its
output captured. The ledger then showed the deployment LIVE with no container.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from infer_stack.leasing.compose import PullProgress
from test_leasing_compose import vllm
from test_leasing_selective_apply import backend

PULL_LINES = [
    'test: Pulling from vllm/vllm-openai',
    'aaaaaaaaaaaa: Pulling fs layer',
    'bbbbbbbbbbbb: Already exists',
    'aaaaaaaaaaaa: Download complete',
    'aaaaaaaaaaaa: Pull complete',
    'Digest: sha256:0123',
]
MANIFEST = [{
    'Descriptor': {'platform': {'os': 'linux', 'architecture': 'amd64'}},
    'OCIManifest': {'layers': [
        {'digest': 'sha256:' + 'a' * 64, 'size': 3 * 10**9},
        {'digest': 'sha256:' + 'b' * 64, 'size': 10**9},
    ]},
}]


class MissingImages:
    """Wraps the fake daemon: some images are absent until pulled."""

    def __init__(self, docker, missing, *, pull_fails=False):
        self.docker = docker
        self.missing = set(missing)
        self.pull_fails = pull_fails
        self.order: list[str] = []

    def __call__(self, args, **kw):
        if args[:3] == ['docker', 'image', 'inspect']:
            if args[-1] in self.missing:
                raise subprocess.CalledProcessError(1, args)
            return 'sha256:x\n'
        if args[:3] == ['docker', 'manifest', 'inspect']:
            return json.dumps(MANIFEST)
        if args[:2] == ['docker', 'pull']:
            self.order.append('pull')
            if self.pull_fails:
                raise subprocess.CalledProcessError(1, args)
            for line in PULL_LINES:
                kw['stdout_lines'](line)
            self.missing.discard(args[-1])
            return ''
        if args[:3] == ['docker', 'rm', '-f']:
            self.order.append('rm')
        if args[:2] == ['docker', 'compose'] and 'up' in args:
            self.order.append('up')
        return self.docker(args, **kw)

    def __getattr__(self, name):
        return getattr(self.docker, name)


def test_a_missing_image_is_pulled_first_with_progress(tmp_path, monkeypatch):
    import platform

    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    be = backend(tmp_path)
    run = MissingImages(be.run, {'vllm/vllm-openai:test'})
    be.run = run
    seen = []
    be.progress = seen.append

    be.converge([vllm('a')])

    assert run.order[:2] == ['pull', 'up']
    assert seen[0] == 'pulling vllm/vllm-openai:test (4.0 GB): not present locally'
    assert 'pulling vllm/vllm-openai:test: 2 of 2 layers downloaded, 4.0 GB of 4.0 GB' in seen
    assert seen[-1] == 'pulled vllm/vllm-openai:test'


def test_a_present_image_is_not_pulled(tmp_path):
    be = backend(tmp_path)
    run = MissingImages(be.run, set())
    be.run = run
    be.converge([vllm('a')])
    assert 'pull' not in run.order


def test_a_failed_pull_removes_nothing(tmp_path):
    """The pull runs before departing containers go, so failing it keeps them."""
    be = backend(tmp_path)
    be.converge([vllm('a')])
    before = set(be.run.containers)
    run = MissingImages(be.run, {'vllm/vllm-openai:other'}, pull_fails=True)
    be.run = run
    be.images['vllm'] = 'vllm/vllm-openai:other'   # `a` must be recreated on it

    with pytest.raises(Exception):
        be.converge([vllm('a')])
    assert run.order == ['pull']                       # failed there, before any rm
    assert set(run.docker.containers) == before


def test_progress_without_sizes_counts_layers_only():
    p = PullProgress('img')
    p.feed('aaaaaaaaaaaa: Pulling fs layer')
    p.feed('bbbbbbbbbbbb: Pulling fs layer')
    assert p.feed('aaaaaaaaaaaa: Download complete') == \
        'pulling img: 1 of 2 layers downloaded'
    assert p.feed('aaaaaaaaaaaa: Pull complete') is None     # already counted
    assert p.feed('Digest: sha256:0123') is None             # not a layer line
