"""Built-in, versioned Docker recipes shipped with infer-stack.

Only exact, code-owned image references can trigger a build. Catalogs cannot
supply arbitrary Dockerfiles or build contexts. Bump the recipe tag when its
contents change; a new upstream revision also requires a new tag. The small
build context is available even in an installed wheel (no checkout required).
"""
import subprocess
from pathlib import Path
from typing import Callable

MAYA_REF = '27daec5e608aa59dab45184e82a1f0611ea7c1aa'
MAYA_IMAGE = f'local/infer-stack-maya-inkling-mini:{MAYA_REF[:12]}-cuda128-sm86-r2'

IMAGE_RECIPES = {
    MAYA_IMAGE: Path(__file__).resolve().parent / 'maya_inkling',
}

# Stable recipe names are separate from versioned Docker tags.
RECIPE_NAMES = {'maya-inkling-mini': MAYA_IMAGE}


class RuntimeImageError(RuntimeError):
    """An unknown or incomplete packaged build recipe."""


def resolve_recipe(name: str) -> str:
    """Accept a stable recipe name or its exact bundled image reference."""
    image = RECIPE_NAMES.get(name, name)
    if image not in IMAGE_RECIPES:
        raise RuntimeImageError(
            f'Unknown bundled image recipe {name!r}; '
            'see `infer-stack image list`.'
        )
    return image


def image_present(image: str, *, run: Callable[..., str]) -> bool:
    args = ['docker', 'image', 'inspect', '--format', '{{.Id}}', image]
    try:
        try:
            run(args, stderr_lines=lambda _line: None)
        except TypeError:  # an injected runner without the keyword
            run(args)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_managed_image(name: str, *, run=None, report=print,
                         report_cached: bool = True) -> str:
    """Prepare a bundled image once, without touching the deployment state.

    Both explicit builds and automatic Compose preparation use this function.
    No catalog, GPU inventory, lease, or checked-out source tree is necessary.
    Only recipes installed with infer-stack can be built here.
    """
    image = resolve_recipe(name)

    from ..leasing.compose import DOCKER_TIMEOUT_BUILD, _default_docker_run
    run = run or _default_docker_run
    if image_present(image, run=run):
        if report_cached:
            report(f'image already present: {image}')
        return image
    context = IMAGE_RECIPES[image]
    if not (context / 'Dockerfile').is_file():
        raise RuntimeImageError(
            f'packaged Docker build recipe missing for {image}: {context}'
        )
    report(f'building {image}: first use, compiling the bundled CUDA engine')
    run(['docker', 'build', '--progress=plain', '--tag', image, str(context)],
        timeout=DOCKER_TIMEOUT_BUILD)
    report(f'built {image}')
    return image
