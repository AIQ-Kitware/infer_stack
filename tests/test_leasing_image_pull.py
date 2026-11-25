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

    def __init__(self, docker, missing, *, pull_fails=False, build_fails=False):
        self.docker = docker
        self.missing = set(missing)
        self.pull_fails = pull_fails
        self.build_fails = build_fails
        self.build_commands = []
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
        if args[:2] == ['docker', 'build']:
            self.order.append('build')
            self.build_commands.append((args, kw))
            if self.build_fails:
                raise subprocess.CalledProcessError(1, args)
            self.missing.discard(args[args.index('--tag') + 1])
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


def test_managed_image_built_before_start_and_reused(tmp_path):
    from infer_stack.leasing.compose import DOCKER_TIMEOUT_BUILD
    from infer_stack.runtime_images import IMAGE_RECIPES, MAYA_IMAGE, MAYA_REF

    be = backend(tmp_path)
    run = MissingImages(be.run, {MAYA_IMAGE})
    be.run = run
    seen = []
    be.progress = seen.append
    dep = vllm('maya')
    dep.spec['runtime']['image'] = MAYA_IMAGE
    be.converge([dep])
    assert run.order == ['build', 'up']
    assert seen[0].startswith(f'building {MAYA_IMAGE}')
    assert seen[-1] == f'built {MAYA_IMAGE}'
    args, kw = run.build_commands[0]
    recipe = IMAGE_RECIPES[MAYA_IMAGE]
    assert args == ['docker', 'build', '--progress=plain', '--tag', MAYA_IMAGE,
                    str(recipe)]
    assert kw['timeout'] == DOCKER_TIMEOUT_BUILD
    assert f'ARG MAYA_REF={MAYA_REF}' in (recipe / 'Dockerfile').read_text()
    assert (recipe / 'launcher.py').is_file()
    assert MAYA_REF[:12] in MAYA_IMAGE
    be.converge([])
    be.converge([dep])
    assert run.order.count('build') == 1
    assert 'pull' not in run.order


def test_failed_managed_image_build_leaves_existing_container(tmp_path):
    from infer_stack.runtime_images import MAYA_IMAGE

    be = backend(tmp_path)
    dep = vllm('a')
    be.converge([dep])
    before = set(be.run.containers)
    run = MissingImages(be.run, {MAYA_IMAGE}, build_fails=True)
    be.run = run
    dep.spec['runtime']['image'] = MAYA_IMAGE
    with pytest.raises(Exception):
        be.converge([dep])
    assert run.order == ['build']
    assert set(run.docker.containers) == before


def test_publish_prepares_managed_image_without_registry_pull(tmp_path):
    from infer_stack.runtime_images import MAYA_IMAGE

    be = backend(tmp_path)
    run = MissingImages(be.run, {MAYA_IMAGE})
    be.run = run
    be.pull_images([MAYA_IMAGE])
    be.pull_images([MAYA_IMAGE])
    assert run.order == ['build']
