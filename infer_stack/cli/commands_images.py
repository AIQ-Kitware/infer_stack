"""Prepare infer-stack's bundled runtime images independently of leases."""
from __future__ import annotations

import kwconf as kw


class ImageListCLI(kw.Config):
    """List bundled build recipes and their versioned Docker image tags."""

    __command__ = 'list'

    @classmethod
    def main(cls, argv=True, **kwargs):
        cls.cli(argv=argv, data=kwargs)
        from ..runtime_images import RECIPE_NAMES

        for name, image in sorted(RECIPE_NAMES.items()):
            print(f'{name}  {image}')
        return 0


class ImageBuildCLI(kw.Config):
    """Build a bundled runtime image if absent, without acquiring an endpoint."""

    __command__ = 'build'
    name = kw.Value(None, position=1, type=str,
                    help='Recipe name from `infer-stack image list`, or its exact image tag.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        from ..runtime_images import RuntimeImageError, ensure_managed_image

        if not config.name:
            raise SystemExit('Specify an image recipe; see `infer-stack image list`.')
        try:
            ensure_managed_image(config.name)
        except RuntimeImageError as exc:
            raise SystemExit(str(exc)) from None
        except FileNotFoundError as exc:
            if exc.filename != 'docker':
                raise
            raise SystemExit('Docker is required to build runtime images.') from None
        return 0


class ImageModalCLI(kw.ModalCLI):
    """Discover and prepare bundled inference runtime images on this Docker host."""

    __command__ = 'image'
    list_command = ImageListCLI
    build = ImageBuildCLI
