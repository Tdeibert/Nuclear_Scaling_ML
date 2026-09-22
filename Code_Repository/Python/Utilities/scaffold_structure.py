#!/usr/bin/env python3
"""
scaffold_structure.py - build and audit the Nuclear Scaling directory layout
from Repository_Structure.json.

Purpose: reusable maintenance tool, not a one-off debug run. It
  1. creates every directory the layout defines for THIS machine that doesn't exist yet
  2. audits the existing tree and reports folders (and root-level files) the layout
     doesn't define, so you can move them by hand

Safety: it only ever creates directories (and, with --gitkeep, empty .gitkeep files).
It never moves, renames, or deletes anything. Dry run by default; --apply to create.

Usage:
  python scaffold_structure.py                    # dry run: what would be created + audit
  python scaffold_structure.py --apply            # create missing directories
  python scaffold_structure.py --apply --gitkeep  # also mark empty tracked dirs so git keeps them
  python scaffold_structure.py --audit-only       # only report stray folders/files

Layout rules it follows (see _schema in the JSON):
  - the keys in METADATA_KEYS are metadata; every other key is a subdirectory
  - _machines is inherited from the parent when a node doesn't set it
  - only nodes whose _created_by is "scaffold" or "git clone" are created;
    anything created at run time (pipeline, importer) is left alone
  - a node with no subdirectories is "open": anything may live inside it
    (Runs/, Models/, Training_Data/, Deprecated/Python/ ...), so it isn't audited
  - stray FILES are only flagged in folders whose node lists _files
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_LAYOUT = Path.home() / "Projects" / "Nuclear_Scaling" / "Repository_Structure.json"
ROOT_KEYS = ("code_root", "data_root")
CREATABLE = {"scaffold", "git clone"}                        # _created_by values we may create
IGNORED_NAMES = {".git", ".gitkeep", "__pycache__", ".ipynb_checkpoints"}


# ---------------------------------------------------------------- layout helpers

def detect_machine():
    """Local machines mirror /data/user/<user> as a symlink; on Cheaha it's a real directory."""
    user_data = Path("/data/user") / os.environ.get("USER", "tdeibert")
    return "local" if user_data.is_symlink() else "cheaha"


# Metadata keys are listed explicitly (not "anything starting with _"), so that real
# folders whose names start with an underscore, like _to_delete/, are treated as folders.
METADATA_KEYS = {"_schema", "_path", "_desc", "_machines", "_git", "_created_by",
                 "_pattern", "_files", "_run_template", "_TODO"}


def subdirs(node):
    """Child directory nodes of a layout node (every dict-valued key that isn't metadata)."""
    return {k: v for k, v in node.items() if k not in METADATA_KEYS and isinstance(v, dict)}


def flatten(node, path, inherited_machines):
    """Depth-first list of (path, node, machines) for every directory in the layout."""
    machines = node.get("_machines", inherited_machines)
    out = [(path, node, machines)]
    for name, child in subdirs(node).items():
        out.extend(flatten(child, path / name, machines))
    return out


def resolve_root(root_node, override):
    """Root path from --code-root/--data-root, else the JSON _path with $HOME/~ expanded."""
    raw = override if override else root_node["_path"]
    return Path(os.path.expandvars(os.path.expanduser(raw)))


# ---------------------------------------------------------------- plan + apply

def plan(layout, machine, overrides):
    """Sort every layout directory into: create / exists / skipped / conflict."""
    result = {"create": [], "exists": [], "skipped": [], "conflict": []}
    for key in ROOT_KEYS:
        root_node = layout[key]
        root_path = resolve_root(root_node, overrides.get(key))
        for path, node, machines in flatten(root_node, root_path, ["local", "cheaha"]):
            created_by = node.get("_created_by", "scaffold")
            if machine not in machines:
                result["skipped"].append((path, f"not on {machine}"))
            elif created_by not in CREATABLE:
                result["skipped"].append((path, f"created at run time by {created_by}"))
            elif path.is_dir():
                result["exists"].append((path, node))
            elif path.exists():
                result["conflict"].append((path, "a FILE exists where a directory should be"))
            else:
                result["create"].append((path, node))
    return result


def apply(result, gitkeep):
    """Create missing directories; optionally drop .gitkeep in empty git-tracked ones."""
    for path, _ in result["create"]:
        path.mkdir(parents=True, exist_ok=True)
    kept = []
    if gitkeep:
        for path, node in result["create"] + result["exists"]:
            if node.get("_git") == "tracked" and path.is_dir() and not any(path.iterdir()):
                (path / ".gitkeep").touch()
                kept.append(path)
    return kept


# ---------------------------------------------------------------- audit

def audit(node, path, strays):
    """Report directories (and files, where _files is defined) the layout doesn't know about."""
    if not path.is_dir():
        return
    children = subdirs(node)
    if not children:                       # open node: any contents allowed
        return
    expected_files = set(node.get("_files", []))
    for entry in sorted(path.iterdir()):
        name = entry.name
        if name in IGNORED_NAMES or name.endswith(".egg-info"):   # .egg-info: made by pip install -e
            continue
        if entry.is_dir():
            if name in children:
                audit(children[name], entry, strays)
            elif not name.startswith("."):  # skip hidden tool dirs (.vscode, .Rproj.user ...)
                strays.append(("dir ", entry))
        elif expected_files and name not in expected_files and not name.startswith("."):
            strays.append(("file", entry))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT, help="path to Repository_Structure.json")
    ap.add_argument("--machine", choices=["local", "cheaha"], default=None, help="override auto-detection")
    ap.add_argument("--apply", action="store_true", help="actually create directories (default: dry run)")
    ap.add_argument("--gitkeep", action="store_true", help="with --apply, add .gitkeep to empty tracked dirs")
    ap.add_argument("--audit-only", action="store_true", help="skip creation, only report strays")
    ap.add_argument("--code-root", default=None, help="override code_root _path (testing)")
    ap.add_argument("--data-root", default=None, help="override data_root _path (testing)")
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text())
    machine = args.machine or detect_machine()
    overrides = {"code_root": args.code_root, "data_root": args.data_root}
    print(f"Layout:  {args.layout}\nMachine: {machine}\n")

    if not args.audit_only:
        result = plan(layout, machine, overrides)

        if result["conflict"]:
            print("CONFLICTS (fix by hand before continuing):")
            for path, why in result["conflict"]:
                print(f"  ! {path}  ({why})")
            print()

        verb = "Creating" if args.apply else "Would create"
        print(f"{verb} {len(result['create'])} director{'y' if len(result['create']) == 1 else 'ies'}:")
        for path, _ in result["create"]:
            print(f"  + {path}")
        print(f"\nAlready present: {len(result['exists'])}")
        print(f"Skipped: {len(result['skipped'])}")
        for path, why in result["skipped"]:
            print(f"  - {path}  ({why})")

        if args.apply:
            if result["conflict"]:
                print("\nNot applying: resolve the conflicts above first.")
                sys.exit(1)
            kept = apply(result, args.gitkeep)
            print("\nDone.")
            if kept:
                print(f"Added .gitkeep to {len(kept)} empty tracked director{'y' if len(kept) == 1 else 'ies'}.")
        else:
            print("\nDry run only. Re-run with --apply to create.")

    strays = []
    for key in ROOT_KEYS:
        audit(layout[key], resolve_root(layout[key], overrides.get(key)), strays)
    print(f"\nNot in the layout ({len(strays)}) - move, deprecate, or quarantine by hand:")
    for kind, path in strays:
        print(f"  ? {kind}  {path}")
    if not strays:
        print("  (none)")


if __name__ == "__main__":
    main()
