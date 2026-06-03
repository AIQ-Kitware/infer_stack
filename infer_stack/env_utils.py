from __future__ import annotations

import secrets
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    """Return a flat dict of KEY=value pairs from a .env file.

    Comments, blank lines, and malformed lines (no '=') are skipped. Both
    managed and unknown keys are returned together.
    """
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        k, v = stripped.split('=', 1)
        data[k.strip()] = v
    return data


def ensure_secret(
    env: dict[str, str], key: str, length: int = 32, prefix: str = ''
) -> str:
    """Return ``env[key]`` if present (and matches ``prefix``), else a fresh secret.

    The ``prefix`` requirement is checked against any existing value: if the
    stored value does not start with ``prefix``, a new secret is generated.
    This matters for keys whose downstream consumer enforces a format
    (e.g. LiteLLM rejects auth tokens that don't start with ``sk-``).
    """
    value = env.get(key, '').strip()
    if value and (not prefix or value.startswith(prefix)):
        return value
    return prefix + secrets.token_urlsafe(length)


def _parse_env_lines(text: str) -> list[tuple[str, str | None, str]]:
    """Tokenize a .env file into ordered records.

    Each record is a tuple ``(kind, key, raw_line)`` where ``kind`` is one of
    ``"kv"`` (a syntactically valid ``KEY=VALUE`` line) or ``"other"`` (a blank
    line, comment, or malformed line). For ``"kv"`` records, ``key`` is the
    variable name; otherwise ``key`` is None. ``raw_line`` is the verbatim
    line (without trailing newline) to preserve formatting on rewrite.
    """
    records: list[tuple[str, str | None, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            records.append(('other', None, line))
            continue
        k, _ = stripped.split('=', 1)
        key = k.strip()
        if not key or any(c.isspace() for c in key):
            records.append(('other', None, line))
            continue
        records.append(('kv', key, line))
    return records


def write_env_file(path: Path, values: dict[str, str]) -> None:
    """Write the .env file non-destructively.

    - Existing unknown ``KEY=value`` pairs are preserved verbatim.
    - Comments and blank lines are preserved in their original positions.
    - Keys present in ``values`` are updated in place (preserving line order).
    - Keys in ``values`` not yet present in the file are appended at the end.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_text = path.read_text(encoding='utf-8') if path.exists() else ''
    records = _parse_env_lines(existing_text)

    seen: set[str] = set()
    out_lines: list[str] = []
    for kind, key, raw in records:
        if kind == 'kv' and key in values:
            out_lines.append(f'{key}={values[key]}')
            seen.add(key)
        else:
            out_lines.append(raw)

    for key, value in values.items():
        if key in seen:
            continue
        out_lines.append(f'{key}={value}')

    text = '\n'.join(out_lines)
    if text and not text.endswith('\n'):
        text += '\n'

    print(f'Write .env to {path}')
    path.write_text(text, encoding='utf-8')
