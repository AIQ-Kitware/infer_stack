"""Render-time diff confirmation.

Builds a unified diff of every file the renderer is about to overwrite, prints
each one as its own panel using Rich syntax-highlighting, and asks the user to
accept the changes interactively unless ``assume_yes=True`` is passed.
"""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import Mapping

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.text import Text


def _read(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def _diff(old: str, new: str, label: Path) -> str:
    return ''.join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f'a/{label}',
            tofile=f'b/{label}',
            n=3,
        )
    )


def confirm_writes(
    planned: Mapping[Path, str],
    *,
    assume_yes: bool,
    console: Console | None = None,
    title: str = 'Pending render changes',
) -> bool:
    """Display a per-file diff and prompt for acceptance.

    Returns True if the writes should proceed, False if the user declined.
    A planned write whose new content matches the existing file is shown as
    "no changes" and never blocks acceptance on its own.
    """
    console = console or Console()

    panels: list[Panel] = []
    any_changes = False
    new_files: list[Path] = []
    modified_files: list[Path] = []
    for path, new_content in planned.items():
        old = _read(path)
        if old == new_content:
            panels.append(
                Panel(
                    Text('(no changes)', style='dim'),
                    title=str(path),
                    border_style='green',
                    title_align='left',
                )
            )
            continue
        any_changes = True
        diff_text = _diff(old, new_content, path)
        if not diff_text:
            # New file — synthesize a diff against an empty original.
            diff_text = _diff('', new_content, path)
        if not old:
            new_files.append(path)
            border = 'yellow'
            tag = '[bold yellow]NEW[/bold yellow]'
        else:
            modified_files.append(path)
            border = 'cyan'
            tag = '[bold cyan]MOD[/bold cyan]'
        body = Syntax(
            diff_text,
            'diff',
            theme='ansi_dark',
            background_color='default',
            word_wrap=False,
            line_numbers=False,
        )
        panels.append(
            Panel(
                body,
                title=f'{tag} {path}',
                border_style=border,
                title_align='left',
            )
        )

    console.rule(f'[bold]{title}[/bold]')
    console.print(Group(*panels))

    if not any_changes:
        console.print('[green]No changes to apply.[/green]')
        return True

    summary = (
        f'[cyan]{len(modified_files)} modified[/cyan], '
        f'[yellow]{len(new_files)} new[/yellow]'
    )
    console.print(f'Summary: {summary}')

    if assume_yes:
        console.print('[green]--yes provided; applying changes.[/green]')
        return True

    return Confirm.ask('Apply these changes?', default=False, console=console)
