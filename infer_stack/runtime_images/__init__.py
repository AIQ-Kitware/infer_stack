"""Built-in, versioned Docker recipes shipped with infer-stack.

Only exact, code-owned image references can trigger a build. Catalogs cannot
supply arbitrary Dockerfiles or build contexts. Bump the recipe tag when its
contents change; a new upstream revision also requires a new tag. The small
build context is available even in an installed wheel (no checkout required).
"""
from pathlib import Path

MAYA_REF = '27daec5e608aa59dab45184e82a1f0611ea7c1aa'
MAYA_IMAGE = f'local/infer-stack-maya-inkling-mini:{MAYA_REF[:12]}-cuda128-sm86-r1'

IMAGE_RECIPES = {
    MAYA_IMAGE: Path(__file__).resolve().parent / 'maya_inkling',
}
