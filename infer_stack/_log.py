"""loguru-based "behind the scenes" feedback for the leasing CLI.

This is deliberately kept off the ``--help`` import path (loguru is ~50ms to
import): nothing at module scope in the CLI imports this. It is pulled in only
at runtime by the leasing verbs (``configure_logging``) and the converge/
reconcile code that narrates what it is doing.

The ``infer_stack`` package is ``disable``d here on import, so importing the
library (or running the test suite, which calls subcommands directly) stays
silent until the CLI explicitly calls :func:`configure_logging`.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

__all__ = ['logger', 'configure_logging']

# Silent by default; the CLI entry points re-enable via configure_logging().
logger.disable('infer_stack')
_CONFIGURED = False


def configure_logging(level: str | None = None, *, force: bool = False):
    """Route infer-stack's narration to stderr (idempotent).

    Level resolves to ``level`` arg > ``$INFER_STACK_LOG_LEVEL`` > ``INFO``.
    Narration goes to stderr so it never pollutes a command's stdout (JSON,
    ``$(infer-stack env KEY)``, etc.). Colorized only on a real terminal.
    """
    global _CONFIGURED
    logger.enable('infer_stack')
    if _CONFIGURED and not force:
        return logger
    level = (level or os.environ.get('INFER_STACK_LOG_LEVEL') or 'INFO').upper()
    colorize = sys.stderr.isatty() and os.environ.get('NO_COLOR') is None
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=colorize,
        format=(
            '<dim>{time:HH:mm:ss}</dim> '
            '<level>{level: <7}</level> {message}'
        ),
    )
    _CONFIGURED = True
    return logger
