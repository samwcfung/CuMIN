"""Clean notebooks/CUMIN.V6.ipynb for committing.

Two jobs:

1. Repoint the config path at the single source of truth. The notebook runs with
   its working directory set to notebooks/, so a bare "config.yaml" resolves to
   notebooks/config.yaml -- a duplicate that silently drifts from
   config/config.yaml. This rewrites it to "../config/config.yaml" so the
   notebook and the CLI always read the same file.

2. Strip execution counts and cell outputs. Jupyter rewrites these on every
   save, so committing them produces huge diffs that bury the actual changes
   and can leak data in figures.

Usage:
    python clean_notebook.py notebooks/CUMIN.V6.ipynb
    python clean_notebook.py notebooks/CUMIN.V6.ipynb --keep-outputs
"""

import argparse
import json
import sys
from pathlib import Path


def clean(path: Path, strip_outputs: bool = True) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))

    repointed = 0
    stripped = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        # 1. repoint config path
        new_source = []
        for line in cell.get("source", []):
            original = line
            if "self.config" in line and "config.yaml" in line:
                indent = line[: len(line) - len(line.lstrip())]
                line = (
                    f'{indent}self.config = "../config/config.yaml"'
                    "  # single source of truth (shared with the CLI)\n"
                )
                if not original.endswith("\n"):
                    line = line.rstrip("\n")
                repointed += 1
            new_source.append(line)
        cell["source"] = new_source

        # 2. strip outputs
        if strip_outputs:
            if cell.get("outputs"):
                stripped += 1
            cell["outputs"] = []
            cell["execution_count"] = None

    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"repointed": repointed, "cells_stripped": stripped}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", type=Path)
    ap.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Repoint the config path but leave cell outputs in place",
    )
    args = ap.parse_args()

    if not args.notebook.exists():
        sys.exit(f"No such notebook: {args.notebook}")

    result = clean(args.notebook, strip_outputs=not args.keep_outputs)
    print(f"config path rewritten in {result['repointed']} cell(s)")
    print(f"outputs cleared from {result['cells_stripped']} cell(s)")
    print(
        "\nnotebooks/config.yaml is now unused. Remove it with:\n"
        "    git rm notebooks/config.yaml"
    )


if __name__ == "__main__":
    main()
