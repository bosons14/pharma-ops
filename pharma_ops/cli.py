"""Command-line interface: ``pharma-ops {triage,target2hit,dock,info}``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from . import __version__

LOG_FMT = "%(levelname)-7s %(name)s: %(message)s"

TRIAGE_COLS = [
    "rank", "mol_id", "triage_score", "mw", "clogp", "tpsa", "qed", "sa_score",
    "n_alerts", "esol_logs", "solubility_class", "lipinski_violations",
    "veber_violations", "gi_absorption_likely", "bbb_permeant_likely", "smiles",
]

DOCK_COLS = [
    "rank", "mol_id", "affinity", "ligand_efficiency", "heavy_atoms",
    "mode_spread", "qed", "sa_score", "n_alerts", "triage_score", "status", "smiles",
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=LOG_FMT)


def _write_outputs(df: pd.DataFrame, summary: dict, out_prefix: str,
                   title: str, subtitle: str, cards, cols, extra=None, notes=None) -> list[str]:
    from .report import render

    written = []
    csv_path = f"{out_prefix}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    written.append(csv_path)

    html_path = f"{out_prefix}.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render(title, subtitle, df, summary, cards, cols, extra, notes))
    written.append(html_path)

    json_path = f"{out_prefix}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "detail": extra or {}}, fh, indent=2, default=str)
    written.append(json_path)
    return written


# ----------------------------------------------------------------------------
# triage
# ----------------------------------------------------------------------------


def cmd_triage(args: argparse.Namespace) -> int:
    from .triage import TriageConfig, screen, summarise

    cfg = TriageConfig(drop_failed=args.drop_failed)
    df = screen(args.library, smiles_col=args.smiles_col, id_col=args.id_col,
                cfg=cfg, limit=args.limit)
    summary = summarise(df)

    if args.min_score is not None and "triage_score" in df:
        before = len(df)
        df = df[df["triage_score"].fillna(-1) >= args.min_score].reset_index(drop=True)
        summary["filtered_out_below_min_score"] = before - len(df)
    if args.alert_free and "n_alerts" in df:
        before = len(df)
        df = df[df["n_alerts"].fillna(99) == 0].reset_index(drop=True)
        summary["filtered_out_with_alerts"] = before - len(df)
    if args.top:
        df = df.head(args.top).reset_index(drop=True)

    cards = [("n_valid", "Compounds"), ("median_mw", "Median MW"),
             ("median_clogp", "Median cLogP"), ("median_qed", "Median QED"),
             ("pct_pass_lipinski", "% pass Ro5"), ("pct_alert_free", "% alert-free"),
             ("pct_gi_absorption_likely", "% GI absorbed")]
    notes = [
        "<b>ADMET columns are rules of thumb.</b> <code>esol_logs</code> is the Delaney ESOL "
        "regression; the absorption and BBB columns are TPSA/logP heuristics (Egan, Clark, "
        "Kelder). They correlate loosely with measured values and should be replaced by "
        "measured or properly modelled data as soon as any exists."
    ]
    written = _write_outputs(df, summary, args.out, "Compound triage",
                             f"{os.path.basename(args.library)}", cards, TRIAGE_COLS,
                             notes=notes)

    print(json.dumps(summary, indent=2))
    cols = [c for c in TRIAGE_COLS if c in df.columns][:11]
    print()
    print(df[cols].head(args.show).to_string(index=False))
    print()
    for p in written:
        print(f"wrote {p}")
    return 0


# ----------------------------------------------------------------------------
# target2hit
# ----------------------------------------------------------------------------


def cmd_target2hit(args: argparse.Namespace) -> int:
    from .target2hit import run

    report, scored = run(
        target_query=args.target,
        actives_file=args.actives_file,
        library=args.library,
        out_dir=args.out_dir,
        organism=None if args.any_organism else args.organism,
        task=args.task,
        activity_threshold=args.activity_threshold,
        activity_types=tuple(args.activity_types),
        max_records=args.max_records,
        test_frac=args.test_frac,
        split=args.split,
        with_structure=not args.no_structure,
        seed=args.seed,
    )

    print(report.to_json())

    if scored is not None and not scored.empty:
        score_col = "p_active" if args.task == "classification" else "predicted_pchembl"
        cols = [c for c in ("mol_id", score_col, "nn_similarity",
                            "in_applicability_domain", "confidence") if c in scored.columns]
        print()
        print(scored[cols].head(args.show).to_string(index=False))

        n_out = int((~scored["in_applicability_domain"]).sum()) if "in_applicability_domain" in scored else 0
        if n_out:
            print(f"\nNOTE: {n_out}/{len(scored)} compounds fall outside the model's "
                  "applicability domain. Those scores are extrapolations.")

        from .report import render

        summary = {**report.activity_summary, **report.model_metrics}
        cards = [("n_compounds", "Training compounds"), ("n_train_scaffolds", "Scaffolds"),
                 ("r2", "Test R²"), ("spearman", "Test Spearman"),
                 ("roc_auc", "Test ROC-AUC"), ("rmse", "Test RMSE")]
        notes = [f"<b>Model validation.</b> {report.model_metrics.get('interpretation','')}",
                 "<b>Applicability domain.</b> Scores for compounds with low nearest-neighbour "
                 "similarity to the training set are extrapolations, not predictions."]
        html_path = os.path.join(args.out_dir, "scored_library.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render("Target-to-hit scoring",
                            report.target.get("pref_name") or args.target or "local dataset",
                            scored, summary, cards,
                            cols + ["smiles"], report.model_metrics, notes))
        print(f"\nwrote {html_path}")

    print(f"outputs in {args.out_dir}/")
    return 0


# ----------------------------------------------------------------------------
# dock
# ----------------------------------------------------------------------------


def cmd_dock(args: argparse.Namespace) -> int:
    from .dock import DockConfig, screen

    cfg = DockConfig(
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        seed=args.seed,
        workers=args.workers,
        cpu_per_job=args.cpu_per_job,
        timeout=args.timeout,
        vina_exe=args.vina_exe,
        scoring=args.scoring,
        max_heavy_atoms=args.max_heavy_atoms,
        run_control=not args.no_control,
    )
    center = tuple(args.box_center) if args.box_center else None

    try:
        df, report = screen(
            args.receptor, args.library, out_dir=args.out_dir,
            chain=args.chain, ligand_resname=args.ligand_resname,
            site_residues=args.site_residues, box_center=center,
            box_size=args.box_size, cfg=cfg, limit=args.limit,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({k: v for k, v in report.items() if k != "receptor"}, indent=2, default=str))
    cols = [c for c in DOCK_COLS if c in df.columns][:10]
    print()
    print(df[cols].head(args.show).to_string(index=False))

    ctrl = report.get("positive_control", {})
    if ctrl.get("status") == "ok":
        print(f"\nRedocking control: {ctrl.get('verdict')}")
    elif ctrl.get("status") in ("skipped", "failed"):
        print(f"\nRedocking control not available: {ctrl.get('reason')}")

    from .report import render

    summary = {k: v for k, v in report.items() if isinstance(v, (int, float, str))}
    cards = [("n_docked", "Docked"), ("n_failed", "Failed"),
             ("affinity_best", "Best score"), ("affinity_median", "Median score"),
             ("n_better_than_minus8", "Score ≤ -8")]
    notes = [
        "<b>Docking scores rank, they do not predict potency.</b> Vina affinities are an "
        "empirical score, not a free energy; within a congeneric series the correlation with "
        "measured potency is weak. Differences under ~1 kcal/mol are not meaningful. "
        "Check <code>ligand_efficiency</code> too, since raw scores favour large greasy "
        "molecules, and inspect the poses before trusting any ranking.",
    ]
    if ctrl.get("status") == "ok":
        notes.append(f"<b>Redocking control.</b> {ctrl.get('verdict')} "
                     f"(RMSD {ctrl.get('rmsd_to_crystal')} Å)")
    else:
        notes.append("<b>No redocking control was run</b>, so there is no evidence that this "
                     "box and receptor preparation can reproduce a known binding mode.")

    html_path = os.path.join(args.out_dir, "docking_report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render("Virtual screening", os.path.basename(args.receptor),
                        df, summary, cards, DOCK_COLS, report.get("box"), notes))
    print(f"\nwrote {html_path}")
    print(f"outputs in {args.out_dir}/")
    return 0


# ----------------------------------------------------------------------------
# info
# ----------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    import importlib

    print(f"pharma-ops {__version__}\n")
    print("dependencies:")
    for mod in ("rdkit", "numpy", "pandas", "sklearn", "scipy", "requests", "meeko", "gemmi"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<10} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            need = " (needed for docking: pip install meeko gemmi)" if mod in ("meeko", "gemmi") else ""
            print(f"  {mod:<10} NOT INSTALLED{need}")

    from .dock import find_vina

    vina = find_vina(args.vina_exe)
    print(f"\nAutoDock Vina: {vina or 'NOT FOUND on PATH'}")
    if not vina:
        from .dock import VINA_INSTALL_HINT
        print("\n" + VINA_INSTALL_HINT)

    print("\nnetwork sources (needed only by target2hit):")
    import requests

    for name, url in (
        ("ChEMBL", "https://www.ebi.ac.uk/chembl/api/data/status.json"),
        ("RCSB PDB", "https://files.rcsb.org/download/1CRN.pdb"),
        ("AlphaFold", "https://alphafold.ebi.ac.uk/api/prediction/P00533"),
    ):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "pharma-ops/0.1"})
            print(f"  {name:<10} HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<10} unreachable ({type(exc).__name__}) -- use --actives-file offline")
    return 0


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pharma-ops",
        description="Early-stage drug discovery toolkit: compound triage, "
                    "target-to-hit modelling, and structure-based virtual screening.",
        epilog="Outputs prioritise compounds for experimental follow-up. They are not "
               "predictions of potency, safety or efficacy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"pharma-ops {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # -- triage -----------------------------------------------------------
    t = sub.add_parser("triage", help="screen a library for drug-likeness, alerts and ADMET proxies",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("library", help="SMILES/CSV/SDF file, or a literal SMILES string")
    t.add_argument("-o", "--out", default="triage", help="output prefix (.csv/.html/_summary.json)")
    t.add_argument("--smiles-col", help="SMILES column name for CSV input")
    t.add_argument("--id-col", help="identifier column name for CSV input")
    t.add_argument("--min-score", type=float, help="keep only compounds at or above this triage score")
    t.add_argument("--alert-free", action="store_true", help="keep only compounds with zero structural alerts")
    t.add_argument("--top", type=int, help="keep only the top N after ranking")
    t.add_argument("--limit", type=int, help="read at most N compounds from the input")
    t.add_argument("--drop-failed", action="store_true", help="omit unparseable compounds from the output")
    t.add_argument("--show", type=int, default=20, help="rows to print to the terminal")
    t.set_defaults(func=cmd_triage)

    # -- target2hit -------------------------------------------------------
    h = sub.add_parser("target2hit", help="pull actives for a target, fit a QSAR model, score a library",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = h.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", help="target name or CHEMBL id, e.g. 'EGFR' or CHEMBL203")
    src.add_argument("--actives-file", help="local CSV of known actives (works offline)")
    h.add_argument("--library", help="candidate library to score")
    h.add_argument("-o", "--out-dir", default="t2h_out")
    h.add_argument("--organism", default="Homo sapiens")
    h.add_argument("--any-organism", action="store_true", help="do not filter targets by organism")
    h.add_argument("--task", choices=["regression", "classification"], default="regression")
    h.add_argument("--activity-threshold", type=float,
                   help="pChEMBL cut for the active class (default 6.0 == 1 uM)")
    h.add_argument("--activity-types", nargs="+", default=["IC50", "Ki", "Kd", "EC50"])
    h.add_argument("--max-records", type=int, default=5000)
    h.add_argument("--test-frac", type=float, default=0.2)
    h.add_argument("--split", choices=["scaffold", "random"], default="scaffold",
                   help="scaffold is harder and far more honest; random splits leak analogues")
    h.add_argument("--no-structure", action="store_true", help="skip PDB/AlphaFold download")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--show", type=int, default=20)
    h.set_defaults(func=cmd_target2hit)

    # -- dock -------------------------------------------------------------
    d = sub.add_parser("dock", help="structure-based virtual screening with AutoDock Vina",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d.add_argument("receptor", help="receptor PDB file")
    d.add_argument("library", help="ligand library (SMILES/CSV/SDF)")
    d.add_argument("-o", "--out-dir", default="dock_out")
    d.add_argument("--chain", help="chain to keep (default: the largest)")
    d.add_argument("--ligand-resname", help="residue name of the co-crystallised ligand defining the box")
    d.add_argument("--site-residues", type=int, nargs="+", help="residue numbers defining the box")
    d.add_argument("--box-center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    d.add_argument("--box-size", type=float, default=22.0, help="cubic box edge in angstrom")
    d.add_argument("--exhaustiveness", type=int, default=8,
                   help="Vina search effort; 8 for screening, 16-32 for final poses")
    d.add_argument("--num-modes", type=int, default=9)
    d.add_argument("--workers", type=int, default=4, help="ligands docked in parallel")
    d.add_argument("--cpu-per-job", type=int, default=1)
    d.add_argument("--timeout", type=int, default=900, help="per-ligand timeout in seconds")
    d.add_argument("--max-heavy-atoms", type=int, default=60)
    d.add_argument("--vina-exe", help="path to the vina or smina executable")
    d.add_argument("--scoring", help="Vina scoring function, e.g. vinardo")
    d.add_argument("--no-control", action="store_true", help="skip the redocking positive control")
    d.add_argument("--limit", type=int)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--show", type=int, default=20)
    d.set_defaults(func=cmd_dock)

    # -- info -------------------------------------------------------------
    i = sub.add_parser("info", help="check dependencies, Vina, and data-source reachability")
    i.add_argument("--vina-exe")
    i.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
