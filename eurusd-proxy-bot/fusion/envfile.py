"""Surgical edits to the .env file that holds your credentials.

Two rules shape everything here:

* **An edit touches only the line it owns.** This file also carries the Supabase
  service_role key, so a careless rewrite is not a cosmetic bug. Comments, blank
  lines and every unrelated setting survive byte-for-byte.
* **A secret that has been written is never printed back in full.** Callers get
  masked labels; the real value only ever goes into the file.

Dependency-free and small, because the same code runs on Termux.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

# Matches `KEY=value`, `# KEY=value` and `  KEY = value`.
_ASSIGNMENT = re.compile(r"^(\s*)(#\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _unquote(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def get_value(lines: Iterable[str], key: str) -> str:
    """Return the live value for ``key``, or "" when unset or commented out."""
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if match and match.group(3) == key and not line.lstrip().startswith("#"):
            return _unquote(line.partition("=")[2])
    return ""


def set_value(
    lines: Sequence[str],
    key: str,
    value: str,
    *,
    comment: str | None = None,
) -> list[str]:
    """Return ``lines`` with ``key=value`` set exactly once.

    An existing live assignment is replaced in place. A placeholder that is
    still commented out is promoted to a live line, and any further commented
    copies are dropped so the file does not collect duplicates. When the key is
    absent entirely it is appended, optionally under a ``comment`` header.
    """
    result: list[str] = []
    replaced = False
    commented_indexes: list[int] = []

    for line in lines:
        match = _ASSIGNMENT.match(line)
        if match and match.group(3) == key:
            if line.lstrip().startswith("#"):
                commented_indexes.append(len(result))
                result.append(line)
                continue
            if replaced:
                # A second live copy would be ambiguous; drop it.
                continue
            result.append(f"{key}={value}")
            replaced = True
            continue
        result.append(line)

    if replaced:
        # Remove the commented placeholders now that a live line exists.
        for index in sorted(commented_indexes, reverse=True):
            del result[index]
        return result

    if commented_indexes:
        # Promote the first placeholder and drop the rest.
        first = commented_indexes[0]
        result[first] = f"{key}={value}"
        for index in sorted(commented_indexes[1:], reverse=True):
            del result[index]
        return result

    while result and not result[-1].strip():
        result.pop()
    if comment:
        result.append("")
        result.append(f"# {comment}")
    result.append(f"{key}={value}")
    return result


def remove_key(lines: Sequence[str], key: str) -> list[str]:
    """Return ``lines`` with every assignment for ``key`` removed."""
    return [
        line
        for line in lines
        if not (_ASSIGNMENT.match(line) and _ASSIGNMENT.match(line).group(3) == key)
    ]


def write(path: str | Path, lines: Sequence[str]) -> None:
    """Write the file atomically and restrict it to the owner.

    Atomic because a half-written .env at boot is worse than an unchanged one,
    and owner-only because the file holds the service_role key. chmod is best
    effort: some Android filesystems refuse it, and that must not be fatal.
    """
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text("\n".join(str(line) for line in lines).rstrip("\n") + "\n", encoding="utf-8")
    os.replace(temporary, target)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
