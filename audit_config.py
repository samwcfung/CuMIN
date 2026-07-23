"""Audit config.get() calls in CuMIN source against keys present in config.yaml.

Reports:
  1. ORPHANED READS  - code reads a key that appears nowhere in the YAML
                       => the hardcoded default silently wins (the evoked bug)
  2. DEAD CONFIG KEYS - YAML supplies a key that no code ever reads
                       => you think you're tuning something; you aren't
"""
import ast
import sys
from pathlib import Path

import yaml


def yaml_keys(path):
    """Every key name appearing anywhere in the YAML tree."""
    keys = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                keys.add(str(k))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(yaml.safe_load(open(path, encoding="utf-8")))
    return keys


CONFIG_RECEIVERS = ("config", "cfg", "conf", "settings", "opts", "thresholds")


def _is_config_receiver(node):
    """Heuristic filter: only count .get() on dicts that look like config.

    Excludes result dicts (spont_params, evoked_params, properties, ...) which
    would otherwise dominate the report with false positives.
    """
    f = node.func.value
    if isinstance(f, ast.Call):
        return True  # chained: config.get("x", {}).get("y", ...)
    name = None
    if isinstance(f, ast.Name):
        name = f.id
    elif isinstance(f, ast.Attribute):
        name = f.attr
    if not name:
        return False
    parts = name.lower().split("_")
    return any(r in parts for r in CONFIG_RECEIVERS)


def code_reads(pyfile):
    """Every literal key read via .get('key', ...) or ['key'], with line numbers."""
    src = Path(pyfile).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  !! could not parse {pyfile}: {e}", file=sys.stderr)
        return []

    found = []
    for node in ast.walk(tree):
        # x.get("key", default)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_config_receiver(node)
        ):
            default = None
            if len(node.args) > 1:
                default = ast.unparse(node.args[1])
            found.append((node.args[0].value, node.lineno, default))
    return found


def main(repo, config_path):
    repo = Path(repo)
    supplied = yaml_keys(config_path)

    pyfiles = [repo / "pipeline.py"] + sorted((repo / "modules").glob("*.py"))

    all_reads = {}          # key -> list of (file, line, default)
    for f in pyfiles:
        for key, line, default in code_reads(f):
            all_reads.setdefault(key, []).append(
                (f.relative_to(repo).as_posix(), line, default)
            )

    read_keys = set(all_reads)

    orphaned = sorted(read_keys - supplied)
    dead = sorted(supplied - read_keys)

    print("=" * 72)
    print("1. ORPHANED READS — code reads it, config never supplies it")
    print("   (the hardcoded default always wins)")
    print("=" * 72)
    for k in orphaned:
        sites = all_reads[k]
        defaults = {d for _, _, d in sites if d is not None}
        dstr = " | ".join(sorted(defaults)) if defaults else "(no default -> None)"
        print(f"\n  {k}")
        print(f"      default in force: {dstr}")
        for f, line, _ in sites[:4]:
            print(f"      read at {f}:{line}")
        if len(sites) > 4:
            print(f"      ... and {len(sites)-4} more")

    print()
    print("=" * 72)
    print("2. DEAD CONFIG KEYS — config supplies it, no code reads it")
    print("=" * 72)
    for k in dead:
        print(f"  {k}")

    print()
    print("=" * 72)
    print(f"SUMMARY: {len(orphaned)} orphaned reads, {len(dead)} dead config keys")
    print(f"         ({len(read_keys)} distinct keys read, {len(supplied)} supplied)")
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
