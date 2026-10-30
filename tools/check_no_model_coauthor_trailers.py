#!/usr/bin/env python3
"""No commit in this repository may credit a model as a co-author.

A trailer, by git's own convention, is a `Key: value` line in the LAST
paragraph of the commit message. A commit message that QUOTES such a line
in its body is not crediting anybody -- and a checker that cannot tell the
two apart fires on every document that describes the problem, which is how
a tripwire gets switched off.

The check reads the history and reports what GitHub counts as
contributions. It also checks itself: a detector that cannot see a trailer
any more is worse than no detector, because it reports green.

Usage:
    python3 tools/check_no_model_coauthor_trailers.py [repo_root]

Exits 1 if a real trailer credits a model, 0 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TRAILER_KEY = "Co-Authored-By:"

#: The addresses that resolve to a GitHub account belonging to a model.
FORBIDDEN = (
    "noreply@anthropic.com",
    "noreply@mavis.local",
    "noreply@minimax",
)

#: The refs that are not the repository's history: filter-branch's backup
#: of a pre-rewrite history holds the very commits a rewrite removed.
SKIP_REF_PREFIXES = (
    "refs/original/",
    "refs/filter-repo/",
    "refs/bisect/",
    "refs/worktree/",
)


def _refs(root: Path) -> list[str]:
    """The refs whose history is the repository's history."""
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "for-each-ref", "--format=%(refname)"],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git for-each-ref failed in {root}: {proc.stderr.strip()}")
    refs = [
        r.strip()
        for r in proc.stdout.splitlines()
        if r.strip() and not r.startswith(SKIP_REF_PREFIXES)
    ]
    return refs or ["HEAD"]


def git_messages(root: Path) -> list[tuple[str, str]]:
    """Every commit message this checkout can see."""
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "log", "--format=%H%x00%B%x00", *_refs(root)],
        cwd=root, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git log failed in {root}: {proc.stderr.strip()}")
    records = [r for r in proc.stdout.split("\x00") if r.strip()]
    return list(zip(records[0::2], records[1::2]))


def trailer_block(message: str) -> list[str]:
    """The trailer block of one message: its last paragraph."""
    blocks = [b for b in message.strip().split("\n\n") if b.strip()]
    return blocks[-1].split("\n") if blocks else []


def credits(message: str, forbidden: tuple[str, ...] = FORBIDDEN) -> list[str]:
    """The trailer lines of *message* that credit a model."""
    return [
        line
        for line in trailer_block(message)
        if line.startswith(TRAILER_KEY) and any(f in line.lower() for f in forbidden)
    ]


def self_check() -> None:
    """The detector must still be able to see what it was written for.

    Three controls, all of them about the check and not about the history:
    a real trailer is seen, a quote of one in prose is not, and a human
    co-author is not a violation.
    """
    real = "Subject\n\nbody\n\nCo-Authored-By: Claude Opus <noreply@anthropic.com>"
    if not credits(real):
        raise SystemExit("the detector is broken: it does not see a real trailer")
    # A quote in prose: the line is indented, as a quoted block is, and
    # the message goes on after it. The last paragraph is then the text
    # that follows -- not the line.
    quoted = (
        "Subject\n\n"
        "The body quotes the old trailer:\n\n"
        "    Co-Authored-By: Claude Opus <noreply@anthropic.com>\n\n"
        "and says it was removed.\n"
    )
    if credits(quoted):
        raise SystemExit("the detector is broken: it fires on a quote in prose")
    human = "Subject\n\nbody\n\nCo-Authored-By: Vaskes <vaskes@mail.ru>"
    if credits(human):
        raise SystemExit("the detector is broken: it fires on a human co-author")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    self_check()
    records = git_messages(root)
    hits = [(sha, line.strip()) for sha, message in records for line in credits(message)]
    if not hits:
        print(f"no commit credits a model as a co-author ({len(records)} message(s) read)")
        return 0
    print(f"{len(hits)} commit message(s) credit a model as a REAL trailer (last paragraph):")
    for sha, line in hits:
        print(f"  {sha[:12]}  {line}")
    print("\nThese are the commits GitHub counts as contributions.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
