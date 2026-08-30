#!/usr/bin/env python3
"""Widen --quality-metric's choices to the new gradient/Jacobian cosine names.

env.py's quality_metric() now understands grad_cosine and jac_cosine and treats
plain "cosine" as a deprecated alias.  argparse would reject the new names
before env.py ever saw them, so the choices list has to learn them too.

Idempotent.
"""
from __future__ import annotations

import re
import sys

PATHS = ["src/alphagrad/approx/ppo.py"]

OLD = re.compile(
    r'choices=\[\s*"auto"\s*,\s*"loss_drop"\s*,\s*"cosine"\s*,\s*"none"\s*\]'
)
NEW = ('choices=["auto", "loss_drop", "grad_cosine", "jac_cosine", '
       '"cosine", "none"]')

STALE_HELP = '" (Jacobian cosine vs the exact reference)"'
NEW_HELP = ('" (gradient cosine at init vs the exact reference; "\n'
            '                  "\'cosine\' is a deprecated alias)"')


def main():
    rc = 0
    for path in PATHS:
        try:
            with open(path) as fh:
                src = fh.read()
        except OSError as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue

        if "grad_cosine" in src:
            print(f"{path}: already patched")
            continue

        new_src, n = OLD.subn(NEW, src)
        if n == 0:
            print(f"{path}: CHOICES ANCHOR NOT FOUND", file=sys.stderr)
            rc = 2
            continue
        new_src = new_src.replace(STALE_HELP, NEW_HELP)

        with open(path, "w") as fh:
            fh.write(new_src)
        print(f"patched {path} ({n} choices list)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
