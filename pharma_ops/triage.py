"""Tool 1 -- compound triage and ADMET-proxy screening.

Given a compound library, compute physicochemical descriptors, apply the
standard published drug-likeness rule sets, flag problematic substructures,
estimate synthetic accessibility, and rank the library.

Every number produced here is either (a) a computed physicochemical property
or (b) an empirical rule or regression published in the medicinal-chemistry
literature. Nothing here is a prediction of clinical safety or efficacy; the
"ADMET" columns are the usual cheap *proxies* used to deprioritise obviously
unsuitable compounds early. A flagged compound is a compound to look at, not
a compound to discard, and an unflagged compound is not a safe compound.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .chem import MolRecord, canonical_smiles, inchikey, load_molecules

log = logging.getLogger("pharma_ops.triage")

# ----------------------------------------------------------------------------
# Synthetic accessibility (Ertl & Schuffenhauer 2009), shipped with RDKit
# ----------------------------------------------------------------------------


def _load_sascorer():
    try:
        import rdkit

        contrib = os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore

        return sascorer
    except Exception as exc:  # noqa: BLE001
        log.warning("SA score unavailable (%s); sa_score will be NaN", exc)
        return None


_SASCORER = _load_sascorer()

# ----------------------------------------------------------------------------
# Structural alert catalogues
# ----------------------------------------------------------------------------

_ALERT_SETS = {
    "PAINS": FilterCatalogParams.FilterCatalogs.PAINS,
    "BRENK": FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalogParams.FilterCatalogs.NIH,
}


def _build_catalogs() -> dict[str, FilterCatalog]:
    cats = {}
    for name, flag in _ALERT_SETS.items():
        params = FilterCatalogParams()
        params.AddCatalog(flag)
        cats[name] = FilterCatalog(params)
    return cats


_CATALOGS = _build_catalogs()


def structural_alerts(mol: Chem.Mol) -> dict[str, list[str]]:
    """Return matched alert descriptions per catalogue.

    PAINS  -- frequent-hitter / assay-interference substructures (Baell 2010)
    BRENK  -- substructures Brenk et al. 2008 advise against in lead series
    NIH    -- NIH/NCATS annotated reactive and undesirable groups
    """
    out: dict[str, list[str]] = {}
    for name, cat in _CATALOGS.items():
        matches = cat.GetMatches(mol)
        out[name] = [m.GetDescription() for m in matches]
    return out


# ----------------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------------


def descriptors(mol: Chem.Mol) -> dict:
    """Physicochemical descriptor block."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    d = {
        "mw": round(mw, 2),
        "clogp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotb,
        "heavy_atoms": heavy,
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": arom_rings,
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "stereocentres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "molar_refractivity": round(Crippen.MolMR(mol), 2),
        "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
    }

    # Derived / composite metrics
    d["qed"] = round(QED.qed(mol), 3)
    d["sa_score"] = round(_SASCORER.calculateScore(mol), 2) if _SASCORER else float("nan")
    # Ligand efficiency-style normalisations (useful once you have potency)
    d["heavy_atom_logp"] = round(logp / heavy, 3) if heavy else float("nan")
    # Lipophilicity-corrected: higher is generally better behaved
    d["logp_tpsa_ratio"] = round(logp / tpsa, 3) if tpsa else float("nan")
    return d


# ----------------------------------------------------------------------------
# Published rule sets
# ----------------------------------------------------------------------------


def rule_violations(d: dict) -> dict:
    """Apply the standard filters. Values are the count of violations."""
    mw, logp, tpsa = d["mw"], d["clogp"], d["tpsa"]
    hbd, hba, rotb = d["hbd"], d["hba"], d["rotatable_bonds"]
    mr, heavy = d["molar_refractivity"], d["heavy_atoms"]

    # Lipinski rule of five (oral bioavailability heuristic)
    lipinski = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    # Veber (oral bioavailability, rodent)
    veber = sum([rotb > 10, tpsa > 140])

    # Ghose
    ghose = sum([not (160 <= mw <= 480), not (-0.4 <= logp <= 5.6),
                 not (40 <= mr <= 130), not (20 <= heavy <= 70)])

    # Egan (absorption)
    egan = sum([tpsa > 131.6, logp > 5.88])

    # Lead-likeness (Teague): keeps room to grow during optimisation
    lead = sum([not (250 <= mw <= 350), logp > 3.5, rotb > 7])

    # Rule of three (fragment screening)
    ro3 = sum([mw > 300, logp > 3, hbd > 3, hba > 3, rotb > 3])

    return {
        "lipinski_violations": lipinski,
        "veber_violations": veber,
        "ghose_violations": ghose,
        "egan_violations": egan,
        "leadlike_violations": lead,
        "ro3_violations": ro3,
        "passes_lipinski": lipinski <= 1,
        "passes_veber": veber == 0,
    }


# ----------------------------------------------------------------------------
# ADMET proxies -- published empirical regressions, not ML predictions
# ----------------------------------------------------------------------------


def admet_proxies(mol: Chem.Mol, d: dict) -> dict:
    """Cheap, transparent absorption/distribution proxies.

    ``esol_logs``  Delaney (2004) ESOL aqueous solubility estimate, log mol/L.
    ``bbb_likely`` Clark/Kelder TPSA heuristic for CNS penetration.
    ``gi_absorption`` Egan "egg" region membership.
    ``pgp_risk``   crude MW/TPSA efflux-liability heuristic.

    These are rules of thumb that correlate loosely with measured values.
    Treat them as a first pass; measured or properly modelled values should
    replace them as soon as you have any.
    """
    logp, mw, tpsa = d["clogp"], d["mw"], d["tpsa"]
    rotb, arom = d["rotatable_bonds"], d["aromatic_rings"]
    heavy = d["heavy_atoms"]

    # Delaney ESOL: logS = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP
    arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = arom_atoms / heavy if heavy else 0.0
    esol = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap

    # TPSA-based CNS heuristic (Clark 1999; Kelder 1999)
    bbb = bool(tpsa < 90 and 1 < logp < 4 and mw < 400 and d["hbd"] <= 3)

    # Egan absorption "egg"
    gi = bool(tpsa <= 131.6 and logp <= 5.88)

    # Crude P-gp substrate liability
    pgp = bool(mw > 400 and tpsa > 75 and rotb > 7)

    return {
        "esol_logs": round(esol, 2),
        "esol_mg_per_ml": round((10 ** esol) * mw * 1000 / 1000, 4),
        "solubility_class": _sol_class(esol),
        "bbb_permeant_likely": bbb,
        "gi_absorption_likely": gi,
        "pgp_efflux_risk": pgp,
        "aromatic_atom_fraction": round(ap, 3),
        "aromatic_ring_overload": arom > 4,
    }


def _sol_class(logs: float) -> str:
    if logs > -2:
        return "soluble"
    if logs > -4:
        return "moderate"
    if logs > -6:
        return "poor"
    return "insoluble"


# ----------------------------------------------------------------------------
# Composite score
# ----------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "qed": 0.35,
    "sa": 0.20,
    "alerts": 0.25,
    "solubility": 0.10,
    "rules": 0.10,
}


@dataclass(frozen=True)
class WeightSet:
    """The weighting a triage run actually used, and where it came from.

    The score is documented as 0-100 and ``--min-score`` filters on that range,
    so the weights have to sum to 1. Rather than trusting callers to arrange
    that, ``resolve_weights`` normalises and records that it did: the range
    then holds by construction, and the report can say which blend produced
    the number instead of leaving a default 73.4 and a custom 73.4
    indistinguishable.
    """

    weights: dict
    is_default: bool
    normalised: bool
    raw_sum: float

    def describe(self) -> str:
        if self.is_default:
            return "default weights: " + self._fmt()
        note = self._fmt()
        if self.normalised:
            note += (
                f" (rescaled from a total of {self.raw_sum:.3g} so the score stays "
                "on 0-100; relative emphasis is unchanged)"
            )
        return "custom weights: " + note

    def _fmt(self) -> str:
        return ", ".join(f"{k} {v:.3g}" for k, v in self.weights.items())


def resolve_weights(weights: dict | None = None) -> WeightSet:
    """Validate and normalise a weighting, refusing anything unusable.

    Unknown keys are rejected rather than ignored. They used to be merged in
    and never read, so a typo such as ``{"qedd": 0.9}`` produced exactly the
    default score and the caller had no way to tell their retuning had done
    nothing.
    """
    if not weights:
        return WeightSet(dict(DEFAULT_WEIGHTS), True, False, sum(DEFAULT_WEIGHTS.values()))

    unknown = sorted(set(weights) - set(DEFAULT_WEIGHTS))
    if unknown:
        raise ValueError(
            f"unknown triage weight(s) {unknown}; valid keys are "
            f"{sorted(DEFAULT_WEIGHTS)}. Nothing was applied."
        )
    bad = sorted(k for k, v in weights.items() if not isinstance(v, (int, float)) or v < 0)
    if bad:
        raise ValueError(f"triage weight(s) {bad} must be non-negative numbers")

    merged = {k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("triage weights sum to zero; at least one must be positive")

    normalised = abs(total - 1.0) > 1e-9
    if normalised:
        merged = {k: v / total for k, v in merged.items()}
    return WeightSet(merged, False, normalised, total)


def composite_score(row: dict, weights: dict | WeightSet | None = None) -> float:
    """A single 0-100 triage score, deliberately simple and auditable.

    It is a weighted blend of QED, synthetic accessibility, structural-alert
    burden, solubility class and rule-set violations. The weights are exposed
    so you can re-tune them for your own programme -- the default blend has no
    special authority, it is just a sane starting point for ranking.

    Weights are resolved through ``resolve_weights``, so the 0-100 range holds
    whatever is passed in.
    """
    ws = weights if isinstance(weights, WeightSet) else resolve_weights(weights)
    w = ws.weights

    qed = float(row.get("qed") or 0.0)  # already 0-1

    sa = row.get("sa_score")
    # SA score runs 1 (easy) to 10 (hard); invert to 0-1
    sa_n = 0.5 if sa is None or np.isnan(sa) else max(0.0, min(1.0, (10.0 - float(sa)) / 9.0))

    n_alerts = int(row.get("n_alerts") or 0)
    alerts_n = max(0.0, 1.0 - 0.25 * n_alerts)  # 4+ alerts -> 0

    esol = float(row.get("esol_logs") or -6.0)
    sol_n = max(0.0, min(1.0, (esol + 6.0) / 6.0))  # -6 -> 0, 0 -> 1

    viol = int(row.get("lipinski_violations") or 0) + int(row.get("veber_violations") or 0)
    rules_n = max(0.0, 1.0 - 0.25 * viol)

    score = (
        w["qed"] * qed
        + w["sa"] * sa_n
        + w["alerts"] * alerts_n
        + w["solubility"] * sol_n
        + w["rules"] * rules_n
    )
    return round(100.0 * score, 1)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


@dataclass
class TriageConfig:
    weights: dict | None = None
    drop_failed: bool = False
    alert_sets: Sequence[str] = ("PAINS", "BRENK", "NIH")

    def __post_init__(self) -> None:
        # Resolve once, at the boundary, so a bad weighting fails before any
        # compound is scored rather than silently producing default-looking
        # numbers.
        self.weight_set = resolve_weights(self.weights)


def screen_record(rec: MolRecord, cfg: TriageConfig | None = None) -> dict:
    cfg = cfg or TriageConfig()
    row: dict = {"mol_id": rec.mol_id, "input_smiles": rec.smiles}

    if not rec.ok:
        row["status"] = "error"
        row["error"] = "; ".join(rec.errors) or "unparseable"
        return row

    mol = rec.mol
    row["status"] = "ok"
    row["smiles"] = canonical_smiles(mol)
    row["inchikey"] = inchikey(mol)
    row["formula"] = rdMolDescriptors.CalcMolFormula(mol)

    d = descriptors(mol)
    row.update(d)
    row.update(rule_violations(d))
    row.update(admet_proxies(mol, d))

    alerts = structural_alerts(mol)
    total = 0
    for name in cfg.alert_sets:
        hits = alerts.get(name, [])
        row[f"{name.lower()}_alerts"] = len(hits)
        row[f"{name.lower()}_detail"] = "; ".join(hits[:5])
        total += len(hits)
    row["n_alerts"] = total

    row["triage_score"] = composite_score(row, cfg.weight_set)
    return row


def screen(
    source: str,
    smiles_col: str | None = None,
    id_col: str | None = None,
    cfg: TriageConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Screen a library and return a ranked DataFrame."""
    cfg = cfg or TriageConfig()
    records = load_molecules(source, smiles_col=smiles_col, id_col=id_col, limit=limit)
    rows = [screen_record(r, cfg) for r in records]
    df = pd.DataFrame(rows)

    if cfg.drop_failed and "status" in df:
        df = df[df["status"] == "ok"].copy()

    if "triage_score" in df:
        df = df.sort_values("triage_score", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def summarise(df: pd.DataFrame, cfg: TriageConfig | None = None) -> dict:
    """Summary statistics, including which weighting produced the scores.

    ``cfg`` is optional for backwards compatibility; pass it so the summary
    records the blend, since a score is not interpretable without it.
    """
    ws = (cfg or TriageConfig()).weight_set
    provenance = {
        "weights": {k: round(v, 4) for k, v in ws.weights.items()},
        "weights_are_default": ws.is_default,
        "weights_note": ws.describe(),
    }
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    if ok.empty:
        return {"n_input": len(df), "n_valid": 0, **provenance}
    return {
        **provenance,
        "n_input": len(df),
        "n_valid": len(ok),
        "n_failed": len(df) - len(ok),
        "median_mw": round(float(ok["mw"].median()), 1),
        "median_clogp": round(float(ok["clogp"].median()), 2),
        "median_qed": round(float(ok["qed"].median()), 3),
        "median_sa_score": round(float(ok["sa_score"].median()), 2),
        "pct_pass_lipinski": round(100.0 * float(ok["passes_lipinski"].mean()), 1),
        "pct_pass_veber": round(100.0 * float(ok["passes_veber"].mean()), 1),
        "pct_alert_free": round(100.0 * float((ok["n_alerts"] == 0).mean()), 1),
        "pct_gi_absorption_likely": round(100.0 * float(ok["gi_absorption_likely"].mean()), 1),
        "top_triage_score": float(ok["triage_score"].max()),
    }
