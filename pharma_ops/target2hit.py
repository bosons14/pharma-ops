"""Tool 2 -- target to hit.

Pipeline:
  target name or CHEMBL id
      -> resolve the ChEMBL target  (or read a local actives CSV offline)
      -> pull dose-response bioactivities, collapse per compound
      -> fit a scaffold-split QSAR model, report honest metrics
      -> optionally fetch a structure (RCSB holo preferred, AlphaFold fallback)
      -> score a candidate library, with applicability-domain flags

Offline use: ``--actives-file`` takes any CSV with a SMILES column and an
activity column, so the whole modelling path works with your own data on a
network-restricted machine.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict

import pandas as pd

from .qsar import QSARModel
from .sources import (
    ChEMBL,
    SourceError,
    Target,
    aggregate_activities,
    fetch_alphafold,
    fetch_pdb,
    pdb_structures_for_uniprot,
)

log = logging.getLogger("pharma_ops.target2hit")


@dataclass
class TargetReport:
    target: dict
    n_activities: int
    n_compounds: int
    activity_summary: dict
    model_metrics: dict
    structure: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ----------------------------------------------------------------------------
# Actives acquisition
# ----------------------------------------------------------------------------


def actives_from_chembl(
    target_query: str,
    organism: str | None = "Homo sapiens",
    activity_types: tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50"),
    max_records: int = 5000,
) -> tuple[Target, pd.DataFrame, pd.DataFrame]:
    client = ChEMBL()
    target = client.resolve_target(target_query, organism=organism)
    log.info("resolved target: %s", target)
    raw = client.activities(target.chembl_id, activity_types, max_records=max_records)
    agg = aggregate_activities(raw)
    return target, raw, agg


def actives_from_file(
    path: str, smiles_col: str | None = None, activity_col: str | None = None
) -> pd.DataFrame:
    """Read a local actives table. Accepts pIC50/pChEMBL directly, or raw
    nM IC50/Ki values which are converted to pChEMBL units."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    scol = smiles_col or next((cols[c] for c in ("smiles", "canonical_smiles", "structure") if c in cols), None)
    if scol is None:
        raise ValueError(f"no SMILES column in {path}; columns are {list(df.columns)}")

    acol = activity_col
    if acol is None:
        for cand in ("pchembl", "pchembl_value", "pic50", "pki", "activity", "ic50_nm", "ic50", "ki_nm", "ki"):
            if cand in cols:
                acol = cols[cand]
                break
    if acol is None:
        raise ValueError(f"no activity column in {path}; columns are {list(df.columns)}")

    out = pd.DataFrame({"smiles": df[scol].astype(str), "pchembl": pd.to_numeric(df[acol], errors="coerce")})
    out = out.dropna(subset=["pchembl"])

    # Heuristic: values far above 14 are raw nM concentrations, not p-units
    if not out.empty and out["pchembl"].median() > 14:
        log.info("activity column looks like raw nM; converting to pChEMBL units")
        import numpy as np

        out["pchembl"] = 9.0 - np.log10(out["pchembl"].clip(lower=1e-6))

    out["molecule_chembl_id"] = df[cols.get("molecule_chembl_id", scol)].astype(str) if "molecule_chembl_id" in cols else [
        f"local_{i+1}" for i in range(len(out))
    ]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------


def get_structure(target: Target, out_dir: str) -> dict:
    """Prefer an experimental PDB entry; fall back to AlphaFold."""
    os.makedirs(out_dir, exist_ok=True)
    info: dict = {"uniprot": target.uniprot}

    if not target.uniprot:
        info["status"] = "no UniProt accession on the ChEMBL target; skipped"
        return info

    try:
        hits = pdb_structures_for_uniprot(target.uniprot, limit=10)
    except SourceError as exc:
        hits = []
        info["pdb_search_error"] = str(exc)

    if hits:
        pdb_id = hits[0]["pdb_id"]
        path = os.path.join(out_dir, f"{pdb_id}.pdb")
        try:
            fetch_pdb(pdb_id, path)
            info.update(
                source="RCSB PDB",
                pdb_id=pdb_id,
                path=path,
                alternatives=[h["pdb_id"] for h in hits[1:6]],
                status="ok",
                note="experimental structure; check for a bound ligand to define the pocket",
            )
            return info
        except SourceError as exc:
            info["pdb_download_error"] = str(exc)

    try:
        path = os.path.join(out_dir, f"AF-{target.uniprot}.pdb")
        fetch_alphafold(target.uniprot, path)
        info.update(
            source="AlphaFold DB",
            path=path,
            status="ok",
            note=(
                "predicted model: no bound ligand, pocket may be collapsed, and "
                "B-factor column holds pLDDT confidence. Prefer an experimental "
                "holo structure for docking where one exists."
            ),
        )
    except SourceError as exc:
        info["status"] = f"no structure retrieved: {exc}"
    return info


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


def run(
    target_query: str | None = None,
    actives_file: str | None = None,
    library: str | None = None,
    out_dir: str = "t2h_out",
    organism: str | None = "Homo sapiens",
    task: str = "regression",
    activity_threshold: float | None = None,
    activity_types: tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50"),
    max_records: int = 5000,
    test_frac: float = 0.2,
    split: str = "scaffold",
    with_structure: bool = True,
    seed: int = 0,
) -> tuple[TargetReport, pd.DataFrame | None]:
    os.makedirs(out_dir, exist_ok=True)
    target = Target(chembl_id="LOCAL", pref_name=target_query or "local dataset")
    raw: pd.DataFrame | None = None

    if actives_file:
        agg = actives_from_file(actives_file)
        n_raw = len(agg)
    elif target_query:
        target, raw, agg = actives_from_chembl(target_query, organism, activity_types, max_records)
        n_raw = len(raw)
        raw.to_csv(os.path.join(out_dir, "bioactivities_raw.csv"), index=False)
    else:
        raise ValueError("give either a target name/CHEMBL id or --actives-file")

    agg.to_csv(os.path.join(out_dir, "actives.csv"), index=False)

    act_summary = {
        "n_compounds": len(agg),
        "pchembl_min": round(float(agg["pchembl"].min()), 2),
        "pchembl_median": round(float(agg["pchembl"].median()), 2),
        "pchembl_max": round(float(agg["pchembl"].max()), 2),
        "n_sub_micromolar": int((agg["pchembl"] >= 6).sum()),
        "n_sub_nanomolar": int((agg["pchembl"] >= 9).sum()),
    }

    model = QSARModel(task=task, seed=seed)
    if activity_threshold is not None:
        model.activity_threshold = activity_threshold
    metrics = model.fit(agg["smiles"].tolist(), agg["pchembl"].tolist(), test_frac=test_frac, split=split)

    structure = get_structure(target, os.path.join(out_dir, "structures")) if (with_structure and not actives_file) else None

    report = TargetReport(
        target=asdict(target) if hasattr(target, "__dataclass_fields__") else {"name": str(target)},
        n_activities=n_raw,
        n_compounds=len(agg),
        activity_summary=act_summary,
        model_metrics=metrics,
        structure=structure,
    )
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        fh.write(report.to_json())

    scored = None
    if library:
        from .chem import load_molecules, canonical_smiles

        recs = load_molecules(library)
        smis = [canonical_smiles(r.mol) for r in recs if r.ok]
        ids = [r.mol_id for r in recs if r.ok]
        scored = model.predict(smis)
        id_map = dict(zip(smis, ids))
        scored.insert(0, "mol_id", scored["smiles"].map(id_map))
        scored.to_csv(os.path.join(out_dir, "scored_library.csv"), index=False)

    return report, scored
