"""Shared cheminformatics helpers: I/O, standardisation, fingerprints.

Keeping all RDKit contact in one module makes the three tools easy to test
and keeps RDKit's noisy logging in a single place.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger("pharma_ops.chem")

# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------


@dataclass
class MolRecord:
    """A single compound as it moves through the pipeline."""

    mol_id: str
    smiles: str
    mol: Chem.Mol | None = None
    props: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mol is not None


# ----------------------------------------------------------------------------
# Standardisation
# ----------------------------------------------------------------------------

_SALT_REMOVER = SaltRemover.SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST = rdMolStandardize.LargestFragmentChooser()


def standardise(mol: Chem.Mol, neutralise: bool = True) -> Chem.Mol | None:
    """Clean a molecule: strip salts, keep the largest fragment, neutralise.

    Returns ``None`` if the molecule cannot be sanitised afterwards. This is
    deliberately conservative -- a compound we cannot standardise is better
    dropped with a recorded error than silently scored.
    """
    try:
        mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
        mol = _LARGEST.choose(mol)
        if neutralise:
            mol = _UNCHARGER.uncharge(mol)
        mol = rdMolStandardize.Cleanup(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception as exc:  # noqa: BLE001 - RDKit raises many types
        log.debug("standardisation failed: %s", exc)
        return None


def mol_from_smiles(smiles: str, do_standardise: bool = True) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if do_standardise:
        mol = standardise(mol)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Input readers
# ----------------------------------------------------------------------------


def _iter_smiles_file(path: str, smiles_col: str | None, id_col: str | None) -> Iterator[tuple[str, str]]:
    """Yield (mol_id, smiles) from .smi / .csv / .tsv.

    Format sniffing: if the first line contains a column that looks like a
    SMILES header, treat the file as delimited with a header row. Otherwise
    assume whitespace-separated ``SMILES [id]`` as in classic .smi files.
    """
    import csv

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        sample = fh.readline()
        fh.seek(0)

        delim = None
        for cand in (",", "\t", ";"):
            if cand in sample:
                delim = cand
                break

        header_like = any(
            tok.strip().strip('"').lower() in {"smiles", "canonical_smiles", "structure"}
            for tok in (sample.split(delim) if delim else sample.split())
        )

        if delim and header_like:
            reader = csv.DictReader(fh, delimiter=delim)
            fields = reader.fieldnames or []
            scol = smiles_col or _pick(fields, ["smiles", "canonical_smiles", "structure"])
            icol = id_col or _pick(fields, ["id", "mol_id", "name", "compound_id", "chembl_id", "title"])
            if scol is None:
                raise ValueError(f"no SMILES column found in {path}; columns were {fields}")
            for i, row in enumerate(reader):
                smi = (row.get(scol) or "").strip()
                if not smi:
                    continue
                mid = (row.get(icol) or "").strip() if icol else ""
                yield (mid or f"mol_{i + 1}", smi)
        else:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smi = parts[0]
                mid = parts[1] if len(parts) > 1 else f"mol_{i + 1}"
                yield (mid, smi)


def _pick(fields: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {f.lower().strip(): f for f in fields}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _iter_sdf(path: str, id_col: str | None) -> Iterator[tuple[str, str, Chem.Mol]]:
    supplier = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
    for i, mol in enumerate(supplier):
        if mol is None:
            yield (f"sdf_{i + 1}", "", None)  # type: ignore[misc]
            continue
        mid = None
        if id_col and mol.HasProp(id_col):
            mid = mol.GetProp(id_col)
        elif mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            mid = mol.GetProp("_Name").strip()
        yield (mid or f"sdf_{i + 1}", Chem.MolToSmiles(mol), mol)


_CHEM_EXTENSIONS = (".smi", ".smiles", ".csv", ".tsv", ".txt", ".sdf", ".sdf.gz", ".mol", ".mol2")


def _looks_like_path(source: str) -> bool:
    """Is this meant to be a file rather than a SMILES string?

    SMILES never contain path separators and effectively never end in a
    chemistry file extension, so either signal means the user intended a
    file that is not there.
    """
    s = source.strip()
    if os.sep in s or (os.altsep and os.altsep in s) or s.startswith("~"):
        return True
    return s.lower().endswith(_CHEM_EXTENSIONS)


def load_molecules(
    source: str,
    smiles_col: str | None = None,
    id_col: str | None = None,
    do_standardise: bool = True,
    limit: int | None = None,
) -> list[MolRecord]:
    """Load compounds from a file path or a single SMILES string.

    Supported: ``.smi``, ``.csv``, ``.tsv``, ``.txt``, ``.sdf``, ``.sdf.gz``,
    or a bare SMILES passed on the command line.
    """
    records: list[MolRecord] = []

    if not os.path.exists(source):
        # A typo'd path must not be silently reinterpreted as a SMILES string,
        # or the user gets "unparseable SMILES" when the real problem is a
        # missing file.
        if _looks_like_path(source):
            raise FileNotFoundError(f"no such file: {source}")

        # Treat as a literal SMILES (or a dot/space separated handful).
        for i, smi in enumerate(s for s in source.replace(",", " ").split() if s):
            mol = mol_from_smiles(smi, do_standardise)
            rec = MolRecord(mol_id=f"input_{i + 1}", smiles=smi, mol=mol)
            if mol is None:
                rec.errors.append("unparseable SMILES")
            records.append(rec)
        if not records:
            raise FileNotFoundError(f"{source!r} is neither an existing file nor a valid SMILES")
        return records

    ext = source.lower()
    if ext.endswith((".sdf", ".sdf.gz", ".mol")):
        for mid, smi, mol in _iter_sdf(source, id_col):
            if mol is not None and do_standardise:
                mol = standardise(mol)
            rec = MolRecord(mol_id=mid, smiles=smi, mol=mol)
            if mol is None:
                rec.errors.append("unreadable SDF record")
            records.append(rec)
            if limit and len(records) >= limit:
                break
    else:
        for mid, smi in _iter_smiles_file(source, smiles_col, id_col):
            mol = mol_from_smiles(smi, do_standardise)
            rec = MolRecord(mol_id=mid, smiles=smi, mol=mol)
            if mol is None:
                rec.errors.append("unparseable SMILES")
            records.append(rec)
            if limit and len(records) >= limit:
                break

    return records


# ----------------------------------------------------------------------------
# Fingerprints (used by tool 2)
# ----------------------------------------------------------------------------


def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP4-style count-folded bit vector as a float array."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def descriptor_vector(mol: Chem.Mol, names: Sequence[str] | None = None) -> np.ndarray:
    """A small, fast, interpretable descriptor block to append to the FP."""
    names = names or DEFAULT_DESCRIPTORS
    out = []
    for name in names:
        fn = getattr(Descriptors, name, None)
        try:
            out.append(float(fn(mol)) if fn else 0.0)
        except Exception:  # noqa: BLE001
            out.append(0.0)
    return np.asarray(out, dtype=np.float32)


DEFAULT_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAromaticRings",
)


def featurise(
    mols: Iterable[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    with_descriptors: bool = True,
) -> np.ndarray:
    rows = []
    for mol in mols:
        fp = morgan_fp(mol, radius, n_bits)
        if with_descriptors:
            fp = np.concatenate([fp, descriptor_vector(mol)])
        rows.append(fp)
    return np.vstack(rows) if rows else np.empty((0, n_bits))


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union else 0.0


def embed_3d(mol: Chem.Mol, seed: int = 0xF00D) -> Chem.Mol | None:
    """Add hydrogens, embed a single conformer, and MMFF-minimise it."""
    try:
        molh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(molh, params) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(molh, maxIters=500)
        except Exception:  # noqa: BLE001 - fall back to the unminimised pose
            pass
        return molh
    except Exception:  # noqa: BLE001
        return None
