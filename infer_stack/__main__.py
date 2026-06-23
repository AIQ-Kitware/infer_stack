"""Enable ``python -m infer_stack`` to run the CLI."""

from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())
