#!/usr/bin/env python3
"""Extract the complete File blocks from the Markdown audit into a clean worktree.

Dry-run by default. Review the audit and printed paths before adding --write.
The Markdown file should be kept outside the target worktree.
"""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

AUDITED_COMMIT = "500e7acc1adfcdc9171f179d08df8aae1f7ae482"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--allow-other-commit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    def git(*command):
        return subprocess.run(["git", *command], cwd=root, check=True, text=True,
                              capture_output=True).stdout.strip()
    if git("rev-parse", "HEAD") != AUDITED_COMMIT and not args.allow_other_commit:
        parser.error("base commit differs; review differences before explicitly allowing another commit")
    if git("status", "--porcelain", "--untracked-files=normal"):
        parser.error("target worktree must be clean; commit/stash changes and keep the audit outside it")
    pattern = r"^### File: `([^`\n]+)`\n\n```[^\n]*\n(.*?)^```[ \t]*$"
    blocks = re.findall(pattern, args.audit.read_text(encoding="utf-8"), re.MULTILINE|re.DOTALL)
    if not blocks:
        parser.error("no complete File blocks found")
    seen = set()
    destinations = []
    for relative, content in blocks:
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or ".git" in p.parts or "\\" in relative:
            parser.error(f"unsafe target path: {relative}")
        target = root.joinpath(*p.parts)
        if not target.resolve().is_relative_to(root) or relative in seen:
            parser.error(f"duplicate or escaping path: {relative}")
        seen.add(relative)
        destinations.append((target, content))
        print(("WRITE " if args.write else "WOULD WRITE ")+relative)
    if not args.write:
        print(f"Dry-run: {len(destinations)} files. Add --write only after review.")
        return
    for target, content in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    print("Applied locally. Review git diff and run tests; no commit or push was performed.")


if __name__ == "__main__":
    main()