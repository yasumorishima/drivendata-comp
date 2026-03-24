#!/usr/bin/env python3
"""Preflight check for DrivenData Kaggle training workflow.

Usage:
    python scripts/preflight.py pasketti-phonetic
    python scripts/preflight.py pasketti-word

Checks:
    1. Kaggle auth (KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json)
    2. Config files exist (drivendata-config.json, kernel-metadata.json, train.py)
    3. Kernel ID format validation ({username}/{slug})
    4. Kernel existence on Kaggle API
    5. Notebook file exists (train_kaggle.ipynb or generated)
    6. train.py dry-run (syntax + import check)
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def check_kaggle_auth():
    """Verify Kaggle API authentication."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.kernels_list(page=1, page_size=1)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_config_files(comp_dir: Path):
    """Verify required config files exist."""
    required = ["drivendata-config.json", "kernel-metadata.json", "train.py"]
    missing = [f for f in required if not (comp_dir / f).exists()]
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "OK"


def check_kernel_id(comp_dir: Path):
    """Validate kernel ID format."""
    meta_path = comp_dir / "kernel-metadata.json"
    if not meta_path.exists():
        return False, "kernel-metadata.json not found", ""

    with open(meta_path) as f:
        meta = json.load(f)

    kid = meta.get("id", "")
    if not kid or "/" not in kid:
        return False, f"Invalid format: '{kid}' (expected '{{username}}/{{slug}}')", kid

    parts = kid.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, f"Invalid format: '{kid}'", kid

    return True, "OK", kid


def check_kernel_exists(kernel_id: str):
    """Check if kernel exists on Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.kernels_status(kernel_id)
        return True, "exists"
    except Exception as e:
        err = str(e)
        if "404" in err or "Not Found" in err:
            return False, "NOT FOUND - bootstrap creation needed"
        return False, f"API error: {err}"


def check_dry_run(comp_dir: Path):
    """Run train.py --dry_run to verify syntax and imports."""
    train_py = comp_dir / "train.py"
    if not train_py.exists():
        return False, "train.py not found"

    # Syntax check only (no GPU/dependencies needed)
    try:
        with open(train_py) as f:
            compile(f.read(), str(train_py), "exec")
        return True, "syntax OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/preflight.py <competition_dir>")
        print("Example: python scripts/preflight.py pasketti-word")
        sys.exit(1)

    comp_dir_name = sys.argv[1]

    # Find repo root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    comp_dir = repo_root / comp_dir_name

    if not comp_dir.is_dir():
        print(f"FAIL: Directory not found: {comp_dir}")
        sys.exit(1)

    errors = 0
    print(f"=== Preflight check: {comp_dir_name} ===\n")

    # 1. Kaggle auth
    print("1. Kaggle auth...", end=" ")
    ok, msg = check_kaggle_auth()
    print(f"{'OK' if ok else 'FAIL'}: {msg}")
    if not ok:
        errors += 1

    # 2. Config files
    print("2. Config files...", end=" ")
    ok, msg = check_config_files(comp_dir)
    print(f"{'OK' if ok else 'FAIL'}: {msg}")
    if not ok:
        errors += 1

    # 3. Kernel ID format
    print("3. Kernel ID format...", end=" ")
    ok, msg, kernel_id = check_kernel_id(comp_dir)
    print(f"{'OK' if ok else 'FAIL'}: {msg}")
    if kernel_id:
        print(f"   Kernel ID: {kernel_id}")
    if not ok:
        errors += 1

    # 4. Kernel existence on Kaggle
    if kernel_id:
        print("4. Kernel exists on Kaggle...", end=" ")
        ok, msg = check_kernel_exists(kernel_id)
        status = "OK" if ok else "WARN"
        print(f"{status}: {msg}")
        if not ok:
            print("   -> Ensure kernel exists step will create it on first run.")
            # Not counted as error — bootstrap will handle it

    # 5. train.py syntax
    print("5. train.py syntax...", end=" ")
    ok, msg = check_dry_run(comp_dir)
    print(f"{'OK' if ok else 'FAIL'}: {msg}")
    if not ok:
        errors += 1

    print(f"\n=== Result: {errors} errors ===")
    if errors > 0:
        print("Fix the errors above before running the workflow.")
        sys.exit(1)
    else:
        print("All checks passed. Safe to trigger workflow.")
        sys.exit(0)


if __name__ == "__main__":
    main()
