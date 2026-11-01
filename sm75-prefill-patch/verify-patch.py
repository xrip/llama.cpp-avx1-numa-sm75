#!/usr/bin/env python3
"""Read-only pre-apply check against the user's actual checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git exited {result.returncode}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Local repository directory")
    args = parser.parse_args()
    if shutil.which("git") is None:
        parser.error("git is not on PATH")
    package = Path(__file__).resolve().parent
    patch = package / "sm75-ninfer-prefill-af084752.patch"
    manifest = json.loads((package / "base-manifest.json").read_text(encoding="utf-8"))
    try:
        repo = Path(git(args.repo.resolve(), "rev-parse", "--show-toplevel"))
        head = git(repo, "rev-parse", "HEAD")
        print(f"Local HEAD: {head}")
        print(f"Patch base: {manifest['base_commit']}")
        exact_files = True
        for relative, expected in manifest["original_blob_sha1"].items():
            path = repo / relative
            if not path.is_file():
                print(f"MISSING: {relative}")
                exact_files = False
                continue
            # Use the working-tree path so Git applies the repository's clean filters.
            actual = git(repo, "hash-object", f"--path={relative}", "--", str(path))
            matched = actual == expected
            print(f"{'MATCH' if matched else 'DIFFERS'}: {relative}")
            exact_files = exact_files and matched
        if not exact_files:
            print("Some files differ from the public base. The context check below decides whether the patch applies.")
        if head != manifest["base_commit"]:
            print("HEAD differs from the public base; no branch switch or history change will be made.")
        git(repo, "apply", "--check", "--whitespace=error-all", str(patch))
        print("PASS: git apply --check on this working tree. No files were changed.")
        print("This does not validate CUDA compilation, numerical behavior, speed, or available VRAM.")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
