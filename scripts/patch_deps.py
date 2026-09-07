#!/usr/bin/env python3
"""Post-install compatibility patches for pinned upstream packages.

Idempotent: safe to re-run, and a no-op once applied. Invoked by `make setup`.
"""
import sys
from pathlib import Path

PATCHES = [
    {
        "package": "pytorchvideo/transforms/augmentations.py",
        "old": "import torchvision.transforms.functional_tensor as F_t",
        "new": "import torchvision.transforms.functional as F_t",
        "why": (
            "torchvision removed transforms.functional_tensor in 0.17. We need "
            "torchvision 0.22 for sm_120 GPUs, so redirect the import. This module "
            "only uses F_t.affine, which the public functional module still provides."
        ),
    },
]


def site_packages():
    for entry in sys.path:
        candidate = Path(entry)
        if candidate.name == "site-packages" and candidate.is_dir():
            return candidate
    raise SystemExit("Could not locate site-packages; run this with the venv python.")


def main():
    root = site_packages()
    applied = skipped = 0

    for patch in PATCHES:
        target = root / patch["package"]
        if not target.exists():
            print(f"  - {patch['package']}: not installed, skipping")
            continue

        text = target.read_text()
        if patch["new"] in text:
            print(f"  = {patch['package']}: already patched")
            skipped += 1
            continue
        if patch["old"] not in text:
            print(f"  ! {patch['package']}: expected import not found, leaving alone")
            continue

        target.write_text(text.replace(patch["old"], patch["new"]))
        print(f"  + {patch['package']}: patched")
        print(f"      {patch['why']}")
        applied += 1

    print(f"\n  {applied} applied, {skipped} already in place")


if __name__ == "__main__":
    main()
