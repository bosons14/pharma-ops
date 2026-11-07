"""Receptor preparation and binding-site detection, in pure Python.

Handles the unglamorous but error-prone part of structure-based screening:

* strip waters, ions, cryoprotectants and buffer components
* keep one chain / one altloc so the grid is unambiguous
* find the co-crystallised ligand and use it to define the docking box
* fall back to a geometric pocket search when the structure is apo

PDB parsing here is intentionally minimal and column-based rather than via a
heavyweight dependency. It covers ATOM/HETATM records, which is all that is
needed to build a Vina box and a receptor PDBQT.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("pharma_ops.receptor")

# Things that are in the file but are not the protein and not the ligand.
SOLVENT = {
    "HOH", "DOD", "WAT", "H2O", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE",
    "MPD", "DMS", "TRS", "EPE", "MES", "ACT", "ACY", "FMT", "CIT", "TLA",
    "IOD", "BR", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI",
    "CO", "CD", "HG", "NO3", "AZI", "CYN", "BME", "DTT", "IMD", "BU3",
    "1PE", "2PE", "P6G", "PE4", "SIN", "OXL", "MLI", "FLC", "UNX", "UNL",
}

AMINO = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}

NUCLEIC = {"A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU"}


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_pdb_line(self) -> str:
        return (
            f"{self.record:<6}{self.serial:>5} {self.name:<4}{self.altloc:1}"
            f"{self.resname:>3} {self.chain:1}{self.resseq:>4}{self.icode:1}   "
            f"{self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}"
            f"{self.occupancy:>6.2f}{self.bfactor:>6.2f}          "
            f"{self.element:>2}"
        )


@dataclass
class Ligand:
    resname: str
    chain: str
    resseq: int
    atoms: list[Atom] = field(default_factory=list)

    @property
    def n_heavy(self) -> int:
        return sum(1 for a in self.atoms if a.element.strip().upper() != "H")

    @property
    def centroid(self) -> np.ndarray:
        return np.mean([a.xyz for a in self.atoms], axis=0)

    @property
    def key(self) -> str:
        return f"{self.resname}:{self.chain}:{self.resseq}"


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source: str

    def as_dict(self) -> dict:
        return {
            "center_x": round(self.center[0], 3),
            "center_y": round(self.center[1], 3),
            "center_z": round(self.center[2], 3),
            "size_x": round(self.size[0], 2),
            "size_y": round(self.size[1], 2),
            "size_z": round(self.size[2], 2),
            "source": self.source,
        }


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def _f(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except ValueError:
        return default


def _i(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except ValueError:
        return default


def parse_pdb(path: str, model: int = 1) -> list[Atom]:
    """Read ATOM/HETATM records from a PDB file, first MODEL only by default."""
    atoms: list[Atom] = []
    current_model = 1
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tag = line[:6]
            if tag.startswith("MODEL"):
                current_model = _i(line[10:14].strip(), 1)
                continue
            if tag.startswith("ENDMDL"):
                if current_model >= model:
                    break
                continue
            if not (tag.startswith("ATOM") or tag.startswith("HETATM")):
                continue
            if current_model != model:
                continue

            element = line[76:78].strip()
            name = line[12:16].strip()
            if not element:
                # Guess from the atom name: first alphabetic char, roughly
                element = "".join(c for c in name if c.isalpha())[:1].upper()

            atoms.append(
                Atom(
                    record=tag.strip(),
                    serial=_i(line[6:11]),
                    name=name,
                    altloc=line[16:17].strip(),
                    resname=line[17:20].strip(),
                    chain=line[21:22].strip() or "A",
                    resseq=_i(line[22:26]),
                    icode=line[26:27].strip(),
                    x=_f(line[30:38]),
                    y=_f(line[38:46]),
                    z=_f(line[46:54]),
                    occupancy=_f(line[54:60], 1.0),
                    bfactor=_f(line[60:66]),
                    element=element,
                )
            )
    if not atoms:
        raise ValueError(f"no ATOM/HETATM records parsed from {path}")
    return atoms


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------


def find_ligands(atoms: list[Atom], min_heavy: int = 8, max_heavy: int = 120) -> list[Ligand]:
    """Candidate co-crystallised ligands, largest first.

    A real ligand is a HETATM residue that is not solvent/buffer/ion and has
    a sensible heavy-atom count. Modified residues and sugars will sometimes
    slip through; the caller should sanity-check the chosen one.
    """
    groups: dict[tuple[str, str, int], Ligand] = {}
    for a in atoms:
        if a.record != "HETATM":
            continue
        rn = a.resname.upper()
        if rn in SOLVENT or rn in AMINO or rn in NUCLEIC:
            continue
        key = (rn, a.chain, a.resseq)
        groups.setdefault(key, Ligand(rn, a.chain, a.resseq)).atoms.append(a)

    out = [lg for lg in groups.values() if min_heavy <= lg.n_heavy <= max_heavy]
    out.sort(key=lambda lg: lg.n_heavy, reverse=True)
    return out


def clean_receptor(
    atoms: list[Atom],
    chain: str | None = None,
    keep_ligand: str | None = None,
    drop_hydrogens: bool = True,
    keep_altloc: str = "A",
) -> tuple[list[Atom], dict]:
    """Return protein-only atoms plus a report of what was removed."""
    counts: Counter = Counter()
    kept: list[Atom] = []

    chains_present = sorted({a.chain for a in atoms if a.resname.upper() in AMINO})
    if chain is None and len(chains_present) > 1:
        # Choose the chain with the most residues -- usually the biological unit
        per_chain = Counter(a.chain for a in atoms if a.resname.upper() in AMINO)
        chain = per_chain.most_common(1)[0][0]
        log.info("multiple chains %s; keeping the largest (%s)", chains_present, chain)

    for a in atoms:
        rn = a.resname.upper()
        if a.altloc and a.altloc.upper() not in ("", keep_altloc.upper()):
            counts["altloc_dropped"] += 1
            continue
        if drop_hydrogens and a.element.strip().upper() == "H":
            counts["hydrogens_dropped"] += 1
            continue
        if rn in AMINO:
            if chain and a.chain != chain:
                counts["other_chain_dropped"] += 1
                continue
            kept.append(a)
            counts["protein_atoms"] += 1
            continue
        if keep_ligand and rn == keep_ligand.upper():
            kept.append(a)
            counts["ligand_kept"] += 1
            continue
        if rn in SOLVENT:
            counts["solvent_dropped"] += 1
        else:
            counts["hetero_dropped"] += 1

    report = {
        "chain_kept": chain or (chains_present[0] if chains_present else None),
        "chains_present": chains_present,
        **dict(counts),
    }
    if counts["protein_atoms"] == 0:
        raise ValueError("no standard amino-acid atoms survived cleaning; is this a protein structure?")
    return kept, report


def write_pdb(atoms: list[Atom], path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("REMARK   prepared by pharma-ops\n")
        for i, a in enumerate(atoms, start=1):
            a.serial = i
            fh.write(a.to_pdb_line() + "\n")
        fh.write("END\n")
    return path


# ----------------------------------------------------------------------------
# Box definition
# ----------------------------------------------------------------------------


def box_from_ligand(ligand: Ligand, padding: float = 5.0, min_size: float = 18.0) -> Box:
    coords = np.array([a.xyz for a in ligand.atoms])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center = (lo + hi) / 2.0
    size = np.maximum((hi - lo) + 2 * padding, min_size)
    return Box(
        center=tuple(float(v) for v in center),  # type: ignore[arg-type]
        size=tuple(float(v) for v in size),  # type: ignore[arg-type]
        source=f"co-crystallised ligand {ligand.key} ({ligand.n_heavy} heavy atoms)",
    )


def box_from_residues(atoms: list[Atom], resids: list[int], chain: str | None = None,
                      padding: float = 6.0, min_size: float = 18.0) -> Box:
    sel = [a for a in atoms if a.resseq in set(resids) and (chain is None or a.chain == chain)]
    if not sel:
        raise ValueError(f"no atoms found for residues {resids} (chain={chain})")
    coords = np.array([a.xyz for a in sel])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center = (lo + hi) / 2.0
    size = np.maximum((hi - lo) + 2 * padding, min_size)
    return Box(
        center=tuple(float(v) for v in center),  # type: ignore[arg-type]
        size=tuple(float(v) for v in size),  # type: ignore[arg-type]
        source=f"residues {sorted(set(resids))}",
    )


def detect_pocket(
    atoms: list[Atom],
    spacing: float = 1.2,
    probe: float = 1.6,
    buried_cutoff: float = 0.78,
    box_size: float = 22.0,
) -> Box:
    """Geometric pocket finder for apo structures.

    A coarse LIGSITE-style grid scan: mark grid points that are outside the
    protein but enclosed by it in many directions, cluster the most buried
    points, and take the largest cluster.

    This is a last resort. It finds *a* concave site, not necessarily *the*
    functional site. If you know the catalytic residues, pass them with
    ``--site-residues`` instead -- it is far more reliable.
    """
    coords = np.array([a.xyz for a in atoms if a.element.strip().upper() != "H"])
    if len(coords) < 50:
        raise ValueError("too few heavy atoms for pocket detection")

    lo = coords.min(axis=0) - 3.0
    hi = coords.max(axis=0) + 3.0

    axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
    # Cap the grid so very large structures stay tractable
    total = np.prod([len(a) for a in axes])
    if total > 4_000_000:
        spacing *= (total / 4_000_000) ** (1 / 3)
        axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]

    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Occupancy: grid points within probe+vdW of any protein atom are inside
    occupied = np.zeros(len(grid), dtype=bool)
    cell = spacing
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, c in enumerate(coords):
        key = tuple((c // (cell * 4)).astype(int))
        buckets[key].append(i)

    r_block = (probe + 1.8) ** 2
    for gi, g in enumerate(grid):
        key = tuple((g // (cell * 4)).astype(int))
        near: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        if near:
            d2 = ((coords[near] - g) ** 2).sum(axis=1)
            if (d2 < r_block).any():
                occupied[gi] = True

    shape = tuple(len(a) for a in axes)
    occ3 = occupied.reshape(shape)

    # Buriedness: fraction of the 13 axis/diagonal directions that hit protein
    directions = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    ]
    free = ~occ3
    buried = np.zeros(shape, dtype=np.float32)
    for d in directions:
        fwd = _ray_hits(occ3, d)
        bwd = _ray_hits(occ3, tuple(-v for v in d))  # type: ignore[arg-type]
        buried += (fwd & bwd).astype(np.float32)
    buried /= len(directions)

    pocket_mask = free & (buried >= buried_cutoff)
    if not pocket_mask.any():
        raise ValueError(
            "no buried cavity found. Supply --site-residues or --box-center explicitly."
        )

    idx = np.argwhere(pocket_mask)
    pts = np.stack([axes[0][idx[:, 0]], axes[1][idx[:, 1]], axes[2][idx[:, 2]]], axis=1)
    labels = _cluster(pts, eps=spacing * 1.9, min_pts=8)
    if labels.max() < 0:
        center = pts.mean(axis=0)
        n = len(pts)
    else:
        sizes = Counter(labels[labels >= 0])
        best = sizes.most_common(1)[0][0]
        sel = pts[labels == best]
        center = sel.mean(axis=0)
        n = len(sel)

    return Box(
        center=tuple(float(v) for v in center),  # type: ignore[arg-type]
        size=(box_size, box_size, box_size),
        source=f"geometric pocket detection ({n} buried grid points) -- VERIFY THIS SITE",
    )


def _ray_hits(occ: np.ndarray, d: tuple[int, int, int], max_steps: int = 30) -> np.ndarray:
    """For each grid point, does a ray in direction d hit protein within range?"""
    hit = np.zeros(occ.shape, dtype=bool)
    cur = occ.copy()
    for _ in range(max_steps):
        cur = _shift(cur, d)
        hit |= cur
    return hit


def _shift(arr: np.ndarray, d: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for ax, step in enumerate(d):
        if step > 0:
            src[ax] = slice(step, None)
            dst[ax] = slice(None, -step)
        elif step < 0:
            src[ax] = slice(None, step)
            dst[ax] = slice(-step, None)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _cluster(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Tiny DBSCAN via sklearn if available, else single-linkage fallback."""
    try:
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(pts)
    except Exception:  # noqa: BLE001
        return np.zeros(len(pts), dtype=int)


# ----------------------------------------------------------------------------
# PDBQT (receptor)
# ----------------------------------------------------------------------------

# AutoDock atom types for a rigid receptor. Vina only needs the broad classes.
_AD_TYPE = {
    "C": "C", "N": "N", "O": "O", "S": "S", "P": "P",
    "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "MG": "Mg", "MN": "Mn", "ZN": "Zn", "CA": "Ca", "FE": "Fe",
}

# Backbone/sidechain atoms that are hydrogen-bond donors or acceptors get the
# polar variants; Vina's scoring distinguishes N/NA and O/OA.
_ACCEPTOR_O = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT"}
_ACCEPTOR_N = {"ND1", "NE2"}


def _ad_type(atom: Atom) -> str:
    el = atom.element.strip().upper()
    base = _AD_TYPE.get(el, el.capitalize() or "C")
    if el == "O":
        return "OA"
    if el == "N":
        return "NA" if atom.name.upper() in _ACCEPTOR_N else "N"
    if el == "S":
        return "SA"
    return base


# Kollman-style united-atom partial charges would need a force field; Vina
# ignores receptor charges entirely for its scoring function, so zeros are
# correct here rather than a shortcut.
def write_receptor_pdbqt(atoms: list[Atom], path: str) -> str:
    """Write a rigid-receptor PDBQT.

    Vina's scoring function does not use receptor partial charges, so the
    charge column is zero by design. Atom typing distinguishes polar N/O
    (NA/OA/SA), which Vina *does* use for its hydrogen-bond term.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("REMARK  rigid receptor prepared by pharma-ops\n")
        for i, a in enumerate(atoms, start=1):
            if a.element.strip().upper() == "H":
                continue
            fh.write(
                f"ATOM  {i:>5} {a.name:<4}{'':1}{a.resname:>3} {a.chain:1}{a.resseq:>4}{'':1}   "
                f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
                f"{1.00:>6.2f}{a.bfactor:>6.2f}    {0.000:>6.3f} {_ad_type(a):<2}\n"
            )
        fh.write("TER\n")
    return path


def prepare(
    pdb_path: str,
    out_dir: str,
    chain: str | None = None,
    ligand_resname: str | None = None,
    site_residues: list[int] | None = None,
    box_center: tuple[float, float, float] | None = None,
    box_size: float = 22.0,
    padding: float = 5.0,
) -> dict:
    """Full receptor prep. Returns paths, the box, and a provenance report."""
    os.makedirs(out_dir, exist_ok=True)
    atoms = parse_pdb(pdb_path)
    ligands = find_ligands(atoms)

    info: dict = {
        "input": pdb_path,
        "n_atoms_in": len(atoms),
        "candidate_ligands": [{"id": lg.key, "n_heavy": lg.n_heavy} for lg in ligands[:8]],
    }

    # Clean first, so the box is defined against the chain we actually keep.
    clean, clean_report = clean_receptor(atoms, chain=chain)
    chain = chain or clean_report.get("chain_kept")

    # --- box ---------------------------------------------------------------
    if box_center is not None:
        box = Box(center=box_center, size=(box_size,) * 3, source="user-specified centre")
    elif site_residues:
        box = box_from_residues(atoms, site_residues, chain, padding=padding)
    elif ligands:
        same_chain = [lg for lg in ligands if lg.chain == chain] or ligands
        chosen = same_chain[0]
        if ligand_resname:
            match = [lg for lg in ligands if lg.resname.upper() == ligand_resname.upper()]
            if not match:
                raise ValueError(
                    f"ligand {ligand_resname!r} not found; candidates were "
                    f"{[lg.key for lg in ligands]}"
                )
            chosen = match[0]
        box = box_from_ligand(chosen, padding=padding)
        info["reference_ligand"] = chosen.key
        # Save it -- useful for redocking validation
        write_pdb(chosen.atoms, os.path.join(out_dir, "reference_ligand.pdb"))
    else:
        log.warning("apo structure: falling back to geometric pocket detection")
        box = detect_pocket(atoms, box_size=box_size)

    # --- sanity checks on the box -----------------------------------------
    warnings: list[str] = []
    vol = box.size[0] * box.size[1] * box.size[2]
    if max(box.size) > 40:
        warnings.append(
            f"box is {box.size[0]:.0f}x{box.size[1]:.0f}x{box.size[2]:.0f} A -- that is very large "
            "for a single site. Vina's search quality degrades badly in big boxes; "
            "narrow the site definition or dock into sub-boxes."
        )
    if vol > 27000:
        warnings.append(f"box volume {vol:.0f} A^3 exceeds the ~27000 A^3 where Vina stays reliable")
    if "pocket detection" in box.source:
        warnings.append(
            "box came from geometric pocket detection, which finds the largest cavity, "
            "not necessarily the functional site. Verify it against the literature."
        )

    # --- receptor ----------------------------------------------------------
    pdb_out = write_pdb(clean, os.path.join(out_dir, "receptor_clean.pdb"))
    pdbqt_out = write_receptor_pdbqt(clean, os.path.join(out_dir, "receptor.pdbqt"))

    info.update(
        cleaning=clean_report,
        box=box.as_dict(),
        receptor_pdb=pdb_out,
        receptor_pdbqt=pdbqt_out,
        n_atoms_out=len(clean),
        warnings=warnings,
    )
    for w in warnings:
        log.warning(w)
    return info
