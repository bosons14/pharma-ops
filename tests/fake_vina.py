#!/usr/bin/env python3
"""A stand-in for the Vina binary, for testing the pipeline without Vina.

Emits a real-format mode table and a real-format output PDBQT, with a
deterministic pseudo-affinity derived from the ligand's atom count so that
ranking is reproducible and testable.

This exists only so the orchestration, parsing and ranking code can be
exercised in CI. It computes nothing physical.
"""

import argparse
import hashlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("receptor", "ligand", "out", "scoring"):
        ap.add_argument(f"--{key}")
    for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
        ap.add_argument(f"--{key}", type=float)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--num_modes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", type=int, default=1)
    args = ap.parse_args()

    lines = open(args.ligand, encoding="utf-8").read().splitlines()
    atoms = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
    n = len(atoms)
    if n == 0:
        print("ERROR: no atoms in ligand", file=sys.stderr)
        return 1

    h = int(hashlib.sha256(args.ligand.encode()).hexdigest()[:8], 16)
    base = -4.0 - 0.18 * n - (h % 220) / 100.0

    modes = []
    for i in range(args.num_modes):
        aff = round(base + 0.33 * i, 2)
        modes.append((i + 1, aff, round(0.0 if i == 0 else 1.1 + 0.6 * i, 3),
                      round(0.0 if i == 0 else 2.0 + 0.8 * i, 3)))

    with open(args.out, "w", encoding="utf-8") as fh:
        for mode, aff, lb, ub in modes:
            fh.write(f"MODEL {mode}\n")
            fh.write(f"REMARK VINA RESULT:    {aff:>8.2f}  {lb:>8.3f}  {ub:>8.3f}\n")
            for ln in atoms:
                fh.write(ln + "\n")
            fh.write("ENDMDL\n")

    print("AutoDock Vina v1.2.5 (fake test stub)")
    print("mode |   affinity | dist from best mode")
    print("     | (kcal/mol) | rmsd l.b.| rmsd u.b.")
    print("-----+------------+----------+----------")
    for mode, aff, lb, ub in modes:
        print(f"{mode:>5}{aff:>13.2f}{lb:>11.3f}{ub:>11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
