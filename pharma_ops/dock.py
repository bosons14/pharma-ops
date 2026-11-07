"""Tool 3 -- structure-based virtual screening with AutoDock Vina.

Pipeline:
  receptor PDB + ligand library
      -> clean receptor, define the box          (pharma_ops.receptor)
      -> 3D-embed and protonate each ligand, write PDBQT via Meeko
      -> run Vina per ligand, in parallel
      -> parse affinities, rerank, join with triage descriptors
      -> optional redocking of the reference ligand as a positive control

What docking scores are and are not
-----------------------------------
A Vina affinity in kcal/mol is an *empirical* score, not a free energy. In
retrospective benchmarks Vina typically enriches actives into the top few
percent of a library, but the correlation between score and measured potency
within a congeneric series is weak -- commonly r ~ 0.3-0.5, often worse.

So: use the score to *rank and cull*, never to predict potency. Two
compounds 0.5 kcal/mol apart are not meaningfully different. A pose that
looks right can still score badly, and a top-scoring pose can be nonsense.
Always look at the poses, and always run the redocking control -- if Vina
cannot reproduce the crystal pose of the reference ligand to within ~2 A
RMSD, the scores for everything else in that box are not trustworthy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rdkit import Chem

from .chem import canonical_smiles, embed_3d, load_molecules
from .receptor import Box, prepare

log = logging.getLogger("pharma_ops.dock")

VINA_INSTALL_HINT = (
    "AutoDock Vina was not found on PATH.\n"
    "  conda:  conda install -c conda-forge vina\n"
    "  binary: https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
    "  smina:  conda install -c conda-forge smina   (then pass --vina-exe smina)\n"
    "Then re-run, or pass --vina-exe /path/to/vina."
)


# ----------------------------------------------------------------------------
# Ligand preparation
# ----------------------------------------------------------------------------


def ligand_pdbqt(mol: Chem.Mol, path: str, seed: int = 0xF00D) -> tuple[bool, str]:
    """3D-embed, minimise and write a flexible-ligand PDBQT via Meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        return False, "meeko is not installed (pip install meeko gemmi)"

    molh = embed_3d(mol, seed=seed)
    if molh is None:
        return False, "3D embedding failed"

    try:
        prep = MoleculePreparation()
        setups = prep.prepare(molh)
        if not setups:
            return False, "Meeko produced no setup"
        result = PDBQTWriterLegacy.write_string(setups[0])
        # Meeko returns (pdbqt_string, success, error_message)
        if isinstance(result, tuple):
            text, ok = result[0], bool(result[1])
            if not ok:
                return False, f"Meeko: {result[2] if len(result) > 2 else 'write failed'}"
        else:
            text = result
        if not text or "ATOM" not in text:
            return False, "Meeko produced an empty PDBQT"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"Meeko failed: {exc}"


# ----------------------------------------------------------------------------
# Vina invocation
# ----------------------------------------------------------------------------


def find_vina(exe: str | None = None) -> str | None:
    for cand in filter(None, [exe, "vina", "vina_1.2.5_linux_x86_64", "smina", "qvina2"]):
        found = shutil.which(cand)
        if found:
            return found
    return None


_AFFINITY_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")


def parse_vina_output(stdout: str, out_pdbqt: str | None = None) -> list[dict]:
    """Parse the Vina result table into mode records.

    Vina prints a table of modes with affinity and RMSD columns. We prefer
    the output PDBQT's REMARK VINA RESULT lines when available, since they
    survive log truncation.
    """
    modes: list[dict] = []

    if out_pdbqt and os.path.exists(out_pdbqt):
        with open(out_pdbqt, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("REMARK VINA RESULT"):
                    parts = line.split(":")[-1].split()
                    if len(parts) >= 3:
                        modes.append(
                            {
                                "mode": len(modes) + 1,
                                "affinity": float(parts[0]),
                                "rmsd_lb": float(parts[1]),
                                "rmsd_ub": float(parts[2]),
                            }
                        )
        if modes:
            return modes

    in_table = False
    for line in stdout.splitlines():
        if re.match(r"^\s*mode\s*\|", line) or "affinity" in line.lower() and "|" in line:
            in_table = True
            continue
        if in_table:
            m = _AFFINITY_RE.match(line)
            if m:
                modes.append(
                    {
                        "mode": int(m.group(1)),
                        "affinity": float(m.group(2)),
                        "rmsd_lb": float(m.group(3)),
                        "rmsd_ub": float(m.group(4)),
                    }
                )
            elif modes:
                break
    return modes


def run_vina(
    vina_exe: str,
    receptor_pdbqt: str,
    ligand_pdbqt_path: str,
    box: Box,
    out_pdbqt: str,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    cpu: int = 1,
    timeout: int = 900,
    scoring: str | None = None,
) -> dict:
    cmd = [
        vina_exe,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt_path,
        "--center_x", f"{box.center[0]:.3f}",
        "--center_y", f"{box.center[1]:.3f}",
        "--center_z", f"{box.center[2]:.3f}",
        "--size_x", f"{box.size[0]:.2f}",
        "--size_y", f"{box.size[1]:.2f}",
        "--size_z", f"{box.size[2]:.2f}",
        "--out", out_pdbqt,
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--cpu", str(cpu),
    ]
    if scoring:
        cmd += ["--scoring", scoring]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"vina exceeded {timeout}s"}
    except OSError as exc:
        return {"status": "error", "error": f"could not execute vina: {exc}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"status": "error", "error": "; ".join(tail[-3:]) or f"exit {proc.returncode}"}

    modes = parse_vina_output(proc.stdout, out_pdbqt)
    if not modes:
        return {"status": "error", "error": "vina produced no scored poses"}

    best = modes[0]
    return {
        "status": "ok",
        "affinity": best["affinity"],
        "n_modes": len(modes),
        "best_mode_rmsd_lb": best["rmsd_lb"],
        "mode_spread": round(modes[-1]["affinity"] - modes[0]["affinity"], 2),
        "out_pdbqt": out_pdbqt,
        "all_modes": modes,
    }


# ----------------------------------------------------------------------------
# Pose geometry / controls
# ----------------------------------------------------------------------------


def pdbqt_coords(path: str, model: int = 1) -> np.ndarray:
    """Heavy-atom coordinates of one model in a PDBQT file."""
    coords, cur = [], 1
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                cur = int(line.split()[1]) if len(line.split()) > 1 else cur
            elif line.startswith("ENDMDL"):
                if cur >= model:
                    break
            elif line.startswith(("ATOM", "HETATM")) and cur == model:
                el = line[77:79].strip().upper()
                if el.startswith("H") or el == "HD":
                    continue
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords)


def symmetry_blind_rmsd(a: np.ndarray, b: np.ndarray) -> float | None:
    """Nearest-neighbour RMSD between two heavy-atom point sets.

    This ignores atom identity and molecular symmetry, so it is a *lower
    bound* on the true RMSD -- adequate for a pass/fail redocking control,
    not for publication-grade pose comparison.
    """
    if len(a) == 0 or len(b) == 0:
        return None
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(np.sqrt((d.min(axis=1) ** 2).mean()))


def redock_control(
    vina_exe: str,
    receptor_pdbqt: str,
    reference_ligand_pdb: str,
    box: Box,
    work_dir: str,
    exhaustiveness: int = 8,
) -> dict:
    """Redock the co-crystallised ligand and compare to its crystal pose.

    This is the single most informative check in structure-based screening,
    and the one most often skipped. If it fails, nothing else in the run
    means much.
    """
    os.makedirs(work_dir, exist_ok=True)
    try:
        ref = Chem.MolFromPDBFile(reference_ligand_pdb, removeHs=False, sanitize=False)
        if ref is None:
            return {"status": "skipped", "reason": "could not read reference ligand"}
        ref_coords = np.array([list(ref.GetConformer().GetAtomPosition(i))
                               for i in range(ref.GetNumAtoms())
                               if ref.GetAtomWithIdx(i).GetSymbol() != "H"])

        smi = Chem.MolToSmiles(ref)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {"status": "skipped",
                    "reason": "reference ligand perceived from PDB has no valid SMILES; "
                              "supply it explicitly for a redocking control"}

        lp = os.path.join(work_dir, "ref_lig.pdbqt")
        ok, err = ligand_pdbqt(mol, lp)
        if not ok:
            return {"status": "skipped", "reason": err}

        out = os.path.join(work_dir, "ref_redock.pdbqt")
        res = run_vina(vina_exe, receptor_pdbqt, lp, box, out, exhaustiveness=exhaustiveness)
        if res["status"] != "ok":
            return {"status": "failed", "reason": res.get("error")}

        rmsd = symmetry_blind_rmsd(pdbqt_coords(out), ref_coords)
        verdict = (
            "pass -- Vina reproduces the crystal pose; box and receptor prep look sound"
            if rmsd is not None and rmsd <= 2.0
            else "FAIL -- Vina does not reproduce the crystal pose. Treat all scores in "
                 "this run as unreliable until the box/protonation/receptor prep is fixed"
        )
        return {
            "status": "ok",
            "reference_smiles": smi,
            "redock_affinity": res["affinity"],
            "rmsd_to_crystal": round(rmsd, 2) if rmsd is not None else None,
            "rmsd_note": "nearest-neighbour RMSD, symmetry-blind: a lower bound on true RMSD",
            "verdict": verdict,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


@dataclass
class DockConfig:
    exhaustiveness: int = 8
    num_modes: int = 9
    seed: int = 42
    workers: int = 4
    cpu_per_job: int = 1
    timeout: int = 900
    keep_poses: bool = True
    vina_exe: str | None = None
    scoring: str | None = None
    max_heavy_atoms: int = 60
    run_control: bool = True


def screen(
    receptor_pdb: str,
    library: str,
    out_dir: str = "dock_out",
    chain: str | None = None,
    ligand_resname: str | None = None,
    site_residues: list[int] | None = None,
    box_center: tuple[float, float, float] | None = None,
    box_size: float = 22.0,
    cfg: DockConfig | None = None,
    with_triage: bool = True,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a full virtual screen. Returns (ranked results, run report)."""
    cfg = cfg or DockConfig()
    os.makedirs(out_dir, exist_ok=True)

    vina = find_vina(cfg.vina_exe)
    if vina is None:
        raise RuntimeError(VINA_INSTALL_HINT)

    # --- receptor ---------------------------------------------------------
    rec_dir = os.path.join(out_dir, "receptor")
    rec = prepare(
        receptor_pdb, rec_dir, chain=chain, ligand_resname=ligand_resname,
        site_residues=site_residues, box_center=box_center, box_size=box_size,
    )
    b = rec["box"]
    box = Box(
        center=(b["center_x"], b["center_y"], b["center_z"]),
        size=(b["size_x"], b["size_y"], b["size_z"]),
        source=b["source"],
    )

    # --- ligands ----------------------------------------------------------
    records = load_molecules(library, limit=limit)
    lig_dir = os.path.join(out_dir, "ligands")
    pose_dir = os.path.join(out_dir, "poses")
    os.makedirs(lig_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    jobs: list[tuple[str, str, str, str]] = []
    rows: list[dict] = []

    for rec_i in records:
        if not rec_i.ok:
            rows.append({"mol_id": rec_i.mol_id, "smiles": rec_i.smiles,
                         "status": "error", "error": "; ".join(rec_i.errors)})
            continue
        if rec_i.mol.GetNumHeavyAtoms() > cfg.max_heavy_atoms:
            rows.append({"mol_id": rec_i.mol_id, "smiles": canonical_smiles(rec_i.mol),
                         "status": "skipped",
                         "error": f"{rec_i.mol.GetNumHeavyAtoms()} heavy atoms exceeds "
                                  f"--max-heavy-atoms {cfg.max_heavy_atoms}; large flexible "
                                  "ligands need much higher exhaustiveness to dock meaningfully"})
            continue

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec_i.mol_id)[:60]
        lp = os.path.join(lig_dir, f"{safe}.pdbqt")
        ok, err = ligand_pdbqt(rec_i.mol, lp, seed=cfg.seed)
        if not ok:
            rows.append({"mol_id": rec_i.mol_id, "smiles": canonical_smiles(rec_i.mol),
                         "status": "error", "error": err})
            continue
        jobs.append((rec_i.mol_id, canonical_smiles(rec_i.mol), lp,
                     os.path.join(pose_dir, f"{safe}_out.pdbqt")))

    log.info("docking %d ligands (exhaustiveness=%d, %d workers)",
             len(jobs), cfg.exhaustiveness, cfg.workers)

    # --- dock -------------------------------------------------------------
    def _one(job):
        mol_id, smi, lp, op = job
        res = run_vina(
            vina, rec["receptor_pdbqt"], lp, box, op,
            exhaustiveness=cfg.exhaustiveness, num_modes=cfg.num_modes,
            seed=cfg.seed, cpu=cfg.cpu_per_job, timeout=cfg.timeout, scoring=cfg.scoring,
        )
        row = {"mol_id": mol_id, "smiles": smi, "status": res["status"]}
        if res["status"] == "ok":
            row.update(
                affinity=res["affinity"],
                n_modes=res["n_modes"],
                mode_spread=res["mode_spread"],
                pose_file=res["out_pdbqt"] if cfg.keep_poses else None,
            )
        else:
            row["error"] = res.get("error")
        return row

    if jobs:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(_one, j): j for j in jobs}
            for fut in as_completed(futures):
                rows.append(fut.result())

    df = pd.DataFrame(rows)

    # --- ligand efficiency + triage join ----------------------------------
    if "affinity" in df.columns:
        heavy = []
        for smi in df["smiles"]:
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
            heavy.append(m.GetNumHeavyAtoms() if m else np.nan)
        df["heavy_atoms"] = heavy
        # Ligand efficiency: binding score per heavy atom. Guards against the
        # classic docking artefact of large greasy molecules scoring well.
        df["ligand_efficiency"] = (-df["affinity"] / df["heavy_atoms"]).round(3)

        if with_triage:
            from .triage import screen_record
            from .chem import MolRecord, mol_from_smiles

            tri_rows = []
            for mid, smi in zip(df["mol_id"], df["smiles"]):
                if not isinstance(smi, str) or not smi:
                    continue
                m = mol_from_smiles(smi)
                if m is None:
                    continue
                t = screen_record(MolRecord(mol_id=mid, smiles=smi, mol=m))
                tri_rows.append({k: t.get(k) for k in
                                 ("mol_id", "qed", "sa_score", "n_alerts",
                                  "clogp", "mw", "esol_logs", "triage_score")})
            if tri_rows:
                df = df.merge(pd.DataFrame(tri_rows), on="mol_id", how="left",
                              suffixes=("", "_triage"))

        df = df.sort_values("affinity", ascending=True, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

    # --- positive control -------------------------------------------------
    control = {"status": "not run"}
    ref_pdb = os.path.join(rec_dir, "reference_ligand.pdb")
    if cfg.run_control and os.path.exists(ref_pdb):
        control = redock_control(vina, rec["receptor_pdbqt"], ref_pdb, box,
                                 os.path.join(out_dir, "control"),
                                 exhaustiveness=cfg.exhaustiveness)
    elif cfg.run_control:
        control = {"status": "skipped",
                   "reason": "no co-crystallised reference ligand, so no redocking control "
                             "is possible. Scores from an apo or predicted structure carry "
                             "substantially more uncertainty."}

    ok = df[df["status"] == "ok"] if "status" in df else df
    report = {
        "receptor": rec,
        "box": box.as_dict(),
        "vina": vina,
        "settings": {
            "exhaustiveness": cfg.exhaustiveness, "num_modes": cfg.num_modes,
            "seed": cfg.seed, "workers": cfg.workers, "scoring": cfg.scoring or "vina",
        },
        "n_input": len(records),
        "n_docked": int(len(ok)),
        "n_failed": int(len(df) - len(ok)),
        "positive_control": control,
    }
    if not ok.empty and "affinity" in ok:
        report["affinity_best"] = float(ok["affinity"].min())
        report["affinity_median"] = round(float(ok["affinity"].median()), 2)
        report["n_better_than_minus8"] = int((ok["affinity"] <= -8.0).sum())

    with open(os.path.join(out_dir, "run_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    df.to_csv(os.path.join(out_dir, "docking_results.csv"), index=False)

    return df, report
