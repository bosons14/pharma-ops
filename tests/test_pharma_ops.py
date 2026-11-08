"""Test suite. Runs fully offline: no network, no Vina binary required.

Docking is exercised against ``fake_vina.py``, a stub that emits real-format
Vina output. That tests the orchestration, parsing, ranking and control
logic; it does not test Vina's physics, which is Vina's job.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")

from pharma_ops import chem, dock, qsar, receptor, triage  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
CATECHOL = "Oc1ccccc1O"  # PAINS
LIGNOCERIC = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)O"  # greasy, insoluble


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    """Generate the sample fixtures once per session."""
    for script in ("make_sample_data.py", "make_sample_pdb.py"):
        path = os.path.join(EXAMPLES, script)
        marker = {
            "make_sample_data.py": "sample_actives.csv",
            "make_sample_pdb.py": "sample_receptor.pdb",
        }[script]
        if not os.path.exists(os.path.join(EXAMPLES, marker)):
            subprocess.run([sys.executable, path], check=True, capture_output=True)
    return EXAMPLES


@pytest.fixture(scope="session")
def fake_vina(tmp_path_factory):
    """Install the Vina stub on PATH and yield its path."""
    d = tmp_path_factory.mktemp("fakebin")
    target = d / "vina"
    shutil.copy(os.path.join(HERE, "fake_vina.py"), target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"
    return str(target)


# ------------------------------------------------------------------- chem


class TestChem:
    def test_parses_valid_smiles(self):
        assert chem.mol_from_smiles(ASPIRIN) is not None

    def test_rejects_invalid_smiles(self):
        assert chem.mol_from_smiles("not_a_smiles") is None
        assert chem.mol_from_smiles("C(C(C(") is None

    def test_canonicalisation_is_stable(self):
        a = chem.canonical_smiles(chem.mol_from_smiles(ASPIRIN))
        b = chem.canonical_smiles(chem.mol_from_smiles(a))
        assert a == b

    def test_salt_is_stripped(self):
        mol = chem.mol_from_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert mol is not None
        assert "Na" not in chem.canonical_smiles(mol)

    def test_largest_fragment_is_kept(self):
        mol = chem.mol_from_smiles(f"{ASPIRIN}.C")
        assert mol.GetNumHeavyAtoms() == chem.mol_from_smiles(ASPIRIN).GetNumHeavyAtoms()

    def test_loads_literal_smiles(self):
        recs = chem.load_molecules(f"{ASPIRIN} {CAFFEINE}")
        assert len(recs) == 2
        assert all(r.ok for r in recs)

    def test_records_parse_errors_without_raising(self):
        recs = chem.load_molecules(f"{ASPIRIN} junk_smiles")
        assert len(recs) == 2
        assert sum(r.ok for r in recs) == 1
        assert recs[1].errors

    def test_loads_smi_file(self, sample_data):
        recs = chem.load_molecules(os.path.join(sample_data, "sample_triage_library.smi"))
        assert len(recs) == 15
        assert recs[0].mol_id == "aspirin"

    def test_loads_csv_with_header(self, sample_data):
        recs = chem.load_molecules(os.path.join(sample_data, "sample_actives.csv"))
        assert len(recs) > 100
        assert all(r.ok for r in recs)

    def test_limit_is_respected(self, sample_data):
        recs = chem.load_molecules(os.path.join(sample_data, "sample_actives.csv"), limit=7)
        assert len(recs) == 7

    def test_missing_file_and_bad_smiles_raises(self):
        with pytest.raises(FileNotFoundError):
            chem.load_molecules("/nonexistent/path.smi")

    def test_fingerprint_shape_and_binary(self):
        fp = chem.morgan_fp(chem.mol_from_smiles(ASPIRIN), n_bits=1024)
        assert fp.shape == (1024,)
        assert set(np.unique(fp)) <= {0.0, 1.0}

    def test_featurise_appends_descriptors(self):
        X = chem.featurise([chem.mol_from_smiles(ASPIRIN)], n_bits=512)
        assert X.shape == (1, 512 + len(chem.DEFAULT_DESCRIPTORS))

    def test_tanimoto_bounds(self):
        a = chem.morgan_fp(chem.mol_from_smiles(ASPIRIN))
        b = chem.morgan_fp(chem.mol_from_smiles(CAFFEINE))
        assert chem.tanimoto(a, a) == pytest.approx(1.0)
        assert 0.0 <= chem.tanimoto(a, b) < 1.0

    def test_3d_embedding(self):
        m = chem.embed_3d(chem.mol_from_smiles(ASPIRIN))
        assert m is not None and m.GetNumConformers() == 1


# ----------------------------------------------------------------- triage


class TestTriage:
    def test_descriptors_match_known_values(self):
        d = triage.descriptors(chem.mol_from_smiles(ASPIRIN))
        assert d["mw"] == pytest.approx(180.16, abs=0.1)
        assert d["hba"] == 3
        assert 0.0 < d["qed"] < 1.0
        assert 1.0 <= d["sa_score"] <= 10.0

    def test_lipinski_passes_a_drug(self):
        d = triage.descriptors(chem.mol_from_smiles(ASPIRIN))
        assert triage.rule_violations(d)["lipinski_violations"] == 0

    def test_lipinski_flags_a_greasy_acid(self):
        d = triage.descriptors(chem.mol_from_smiles(LIGNOCERIC))
        assert triage.rule_violations(d)["lipinski_violations"] >= 1

    def test_pains_catches_catechol(self):
        alerts = triage.structural_alerts(chem.mol_from_smiles(CATECHOL))
        assert len(alerts["PAINS"]) > 0

    def test_clean_drug_has_no_pains(self):
        assert triage.structural_alerts(chem.mol_from_smiles(CAFFEINE))["PAINS"] == []

    def test_insoluble_compound_is_classified(self):
        mol = chem.mol_from_smiles(LIGNOCERIC)
        a = triage.admet_proxies(mol, triage.descriptors(mol))
        assert a["esol_logs"] < -4
        assert a["solubility_class"] in ("poor", "insoluble")

    def test_composite_score_is_bounded(self):
        for smi in (ASPIRIN, CAFFEINE, CATECHOL, LIGNOCERIC):
            row = triage.screen_record(
                chem.MolRecord("x", smi, chem.mol_from_smiles(smi))
            )
            assert 0.0 <= row["triage_score"] <= 100.0

    def test_greasy_scores_below_drug(self):
        def s(smi):
            return triage.screen_record(chem.MolRecord("x", smi, chem.mol_from_smiles(smi)))["triage_score"]
        assert s(LIGNOCERIC) < s(ASPIRIN)

    def test_screen_ranks_and_handles_errors(self):
        df = triage.screen(f"{ASPIRIN} {CATECHOL} {LIGNOCERIC} bad_smiles")
        assert len(df) == 4
        assert (df["status"] == "error").sum() == 1
        ok = df[df["status"] == "ok"]
        assert ok["triage_score"].is_monotonic_decreasing
        assert list(df["rank"]) == [1, 2, 3, 4]

    def test_summarise_reports_percentages(self, sample_data):
        df = triage.screen(os.path.join(sample_data, "sample_triage_library.smi"))
        s = triage.summarise(df)
        assert s["n_valid"] == 15
        assert 0 <= s["pct_pass_lipinski"] <= 100
        assert 0 <= s["pct_alert_free"] <= 100

    def test_empty_summary_does_not_crash(self):
        assert triage.summarise(pd.DataFrame({"status": []}))["n_valid"] == 0

    # -- weighting provenance and validation -----------------------------

    def test_default_weights_sum_to_one(self):
        assert sum(triage.DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_score_stays_in_range_whatever_weights_are_passed(self):
        """The 0-100 range is documented and --min-score filters on it."""
        row = dict(qed=0.9, sa_score=2.0, n_alerts=0, esol_logs=-1.0,
                   lipinski_violations=0, veber_violations=0)
        for mult in (0.01, 0.5, 1.0, 2.0, 100.0):
            w = {k: v * mult for k, v in triage.DEFAULT_WEIGHTS.items()}
            assert 0.0 <= triage.composite_score(row, w) <= 100.0

    def test_rescaling_preserves_relative_emphasis(self):
        row = dict(qed=0.9, sa_score=2.0, n_alerts=1, esol_logs=-3.0,
                   lipinski_violations=1, veber_violations=0)
        base = triage.composite_score(row)
        doubled = triage.composite_score(
            row, {k: v * 2 for k, v in triage.DEFAULT_WEIGHTS.items()}
        )
        assert doubled == pytest.approx(base)

    def test_unknown_weight_key_is_refused_not_ignored(self):
        """The bug: a typo merged in, was never read, and scored as default."""
        with pytest.raises(ValueError, match="unknown triage weight"):
            triage.resolve_weights({"qedd": 0.9})

    def test_negative_and_zero_weights_are_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            triage.resolve_weights({"qed": -0.5})
        with pytest.raises(ValueError, match="sum to zero"):
            triage.resolve_weights({k: 0.0 for k in triage.DEFAULT_WEIGHTS})

    def test_bad_weights_fail_before_any_compound_is_scored(self):
        with pytest.raises(ValueError, match="unknown triage weight"):
            triage.TriageConfig(weights={"nonsense": 1.0})

    def test_weight_set_records_its_provenance(self):
        d = triage.resolve_weights(None)
        assert d.is_default and not d.normalised
        assert "default weights" in d.describe()

        c = triage.resolve_weights({"qed": 0.6, "alerts": 0.4})
        assert not c.is_default and c.normalised
        assert "custom weights" in c.describe()
        assert "rescaled" in c.describe()
        assert sum(c.weights.values()) == pytest.approx(1.0)

    def test_summary_says_which_blend_produced_the_scores(self):
        cfg = triage.TriageConfig(weights={"qed": 0.6, "alerts": 0.4})
        df = triage.screen(ASPIRIN, cfg=cfg)
        s = triage.summarise(df, cfg)
        assert s["weights_are_default"] is False
        assert s["weights"]["qed"] == pytest.approx(0.4286, abs=1e-3)
        assert "custom weights" in s["weights_note"]

        dflt = triage.TriageConfig()
        s2 = triage.summarise(triage.screen(ASPIRIN, cfg=dflt), dflt)
        assert s2["weights_are_default"] is True

    def test_custom_weights_actually_change_the_score(self):
        """Guards the silent-no-op: retuning must be observable in the output."""
        a = triage.screen(ASPIRIN, cfg=triage.TriageConfig())
        b = triage.screen(ASPIRIN, cfg=triage.TriageConfig(weights={"sa": 0.9}))
        assert a["triage_score"].iloc[0] != b["triage_score"].iloc[0]


# ------------------------------------------------------------------- qsar


class TestQSAR:
    def test_murcko_scaffold(self):
        assert qsar.murcko_scaffold("CC(C)c1ccc2[nH]ncc2c1") != ""

    def test_scaffold_split_is_disjoint_and_covers_all(self):
        smis = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))["smiles"].tolist()
        tr, te = qsar.scaffold_split(smis, test_frac=0.2)
        assert set(tr).isdisjoint(set(te))
        assert len(tr) + len(te) == len(smis)
        assert len(te) > 0 and len(tr) > 0

    def test_scaffold_split_separates_scaffolds(self):
        smis = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))["smiles"].tolist()
        tr, te = qsar.scaffold_split(smis, test_frac=0.3)
        tr_s = {qsar.murcko_scaffold(smis[i]) for i in tr}
        te_s = {qsar.murcko_scaffold(smis[i]) for i in te}
        assert tr_s.isdisjoint(te_s), "scaffold split leaked a scaffold across the boundary"

    def test_scaffold_split_is_deterministic(self):
        smis = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))["smiles"].tolist()
        a = qsar.scaffold_split(smis, 0.2)
        b = qsar.scaffold_split(smis, 0.2)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_regression_fit_reports_metrics(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        m = qsar.QSARModel(task="regression", n_estimators=60)
        met = m.fit(df["smiles"].tolist(), df["pchembl"].tolist())
        for key in ("r2", "rmse", "mae", "spearman", "n_train", "n_test", "interpretation"):
            assert key in met
        assert met["split"] == "scaffold"
        assert met["n_train"] + met["n_test"] == met["n_total"]

    def test_classification_fit_reports_auc(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        m = qsar.QSARModel(task="classification", n_estimators=60)
        m.activity_threshold = 7.3
        met = m.fit(df["smiles"].tolist(), df["pchembl"].tolist())
        assert met["roc_auc"] is None or 0.0 <= met["roc_auc"] <= 1.0

    def test_too_few_compounds_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            qsar.QSARModel().fit([ASPIRIN, CAFFEINE], [6.0, 7.0])

    def test_predict_flags_applicability_domain(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        m = qsar.QSARModel(n_estimators=60)
        m.fit(df["smiles"].tolist(), df["pchembl"].tolist())

        # A training compound should be in-domain; an unrelated drug should not
        out = m.predict([df["smiles"].iloc[0], LIGNOCERIC])
        row_known = out[out["smiles"] == df["smiles"].iloc[0]].iloc[0]
        row_alien = out[out["smiles"] == LIGNOCERIC].iloc[0]
        assert row_known["nn_similarity"] > row_alien["nn_similarity"]
        assert not row_alien["in_applicability_domain"]
        assert "out of domain" in row_alien["confidence"]

    def test_predict_handles_bad_smiles(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        m = qsar.QSARModel(n_estimators=40)
        m.fit(df["smiles"].tolist(), df["pchembl"].tolist())
        out = m.predict([ASPIRIN, "junk"])
        assert (out["status"] == "error").sum() == 1

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            qsar.QSARModel().predict([ASPIRIN])

    def test_negative_r2_with_good_rank_is_explained(self):
        msg = qsar.QSARModel._interpret(
            {"task": "regression", "split": "scaffold", "r2": -0.7, "spearman": 0.65}
        )
        assert "not calibrated" in msg.lower()
        assert "rank" in msg.lower()

    def test_random_split_verdict_warns_of_inflation(self):
        msg = qsar.QSARModel._interpret(
            {"task": "regression", "split": "random", "r2": 0.73, "spearman": 0.86}
        )
        assert "RANDOM SPLIT" in msg
        assert "upper bound" in msg.lower()
        assert "not an estimate" in msg.lower()

    def test_random_split_never_claims_held_out_scaffolds(self):
        """The bug this guards: a random-split run described as a scaffold one."""
        for task, extra in (("regression", {"r2": 0.73, "spearman": 0.86}),
                            ("classification", {"roc_auc": 0.91})):
            msg = qsar.QSARModel._interpret({"task": task, "split": "random", **extra})
            assert "held-out scaffolds" not in msg
            assert "RANDOM SPLIT" in msg

    def test_scaffold_split_verdict_has_no_inflation_warning(self):
        msg = qsar.QSARModel._interpret(
            {"task": "regression", "split": "scaffold", "r2": 0.63, "spearman": 0.80}
        )
        assert "RANDOM SPLIT" not in msg
        assert "held-out scaffolds" in msg

    def test_unknown_split_is_refused_at_fit(self):
        """A protocol with no SplitSpec must not run: it has no agreed wording."""
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        with pytest.raises(ValueError, match="unknown split protocol"):
            qsar.QSARModel(n_estimators=20).fit(
                df["smiles"].tolist(), df["pchembl"].tolist(), split="time"
            )

    def test_every_split_declares_its_own_wording(self):
        for name, spec in qsar.SPLITS.items():
            assert spec.name == name
            assert spec.held_out and spec.description
            # An optimistic protocol must carry the warning that says so.
            assert bool(spec.caveat) == spec.optimistic

    def test_no_split_can_be_described_as_another(self):
        """The original bug, generalised: wording must track the protocol."""
        for name, spec in qsar.SPLITS.items():
            msg = qsar.QSARModel._interpret(
                {"task": "regression", "split": name, "r2": 0.63, "spearman": 0.80}
            )
            assert spec.held_out in msg
            for other in qsar.SPLITS.values():
                if other.held_out != spec.held_out:
                    assert other.held_out not in msg

    def test_unrecognised_split_metrics_describe_nothing(self):
        msg = qsar.QSARModel._interpret(
            {"task": "regression", "split": "mystery", "r2": 0.9, "spearman": 0.9}
        )
        assert "not recognised" in msg
        assert "unvalidated" in msg
        assert "held-out scaffolds" not in msg

    def test_confidence_never_contradicts_the_domain_flag(self):
        """At any threshold, 'out of domain' and a confidence grade must agree."""
        for thr in (0.20, 0.30, 0.50, 0.70, 0.90):
            for sim in [i / 100 for i in range(0, 101, 5)]:
                conf = qsar._conf(sim, thr)
                in_domain = sim >= thr
                if in_domain:
                    assert "out of domain" not in conf, (thr, sim, conf)
                else:
                    assert "out of domain" in conf, (thr, sim, conf)

    def test_confidence_bands_scale_with_threshold(self):
        # The old hardcoded bands put 0.55 at "medium" even when it was below
        # a 0.70 threshold and therefore out of domain.
        assert "out of domain" in qsar._conf(0.55, 0.70)
        assert qsar._conf(0.55, 0.30) == "medium"
        assert "high" in qsar._conf(0.95, 0.70)

    def test_metrics_carry_the_provenance_the_wording_depends_on(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        m = qsar.QSARModel(n_estimators=40, ad_threshold=0.42).fit(
            df["smiles"].tolist(), df["pchembl"].tolist(), split="random"
        )
        assert m["split"] == "random"
        assert m["split_is_optimistic"] is True
        assert m["ad_threshold"] == 0.42
        assert m["split_description"]

    def test_fitted_model_verdict_matches_the_split_that_ran(self):
        df = pd.read_csv(os.path.join(EXAMPLES, "sample_actives.csv"))
        smis, ys = df["smiles"].tolist(), df["pchembl"].tolist()
        scaf = qsar.QSARModel(n_estimators=40).fit(smis, ys, split="scaffold")
        rand = qsar.QSARModel(n_estimators=40).fit(smis, ys, split="random")
        assert "RANDOM SPLIT" not in scaf["interpretation"]
        assert "RANDOM SPLIT" in rand["interpretation"]


# --------------------------------------------------------------- receptor


class TestReceptor:
    def test_parses_pdb(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        assert len(atoms) > 100
        assert {a.chain for a in atoms} >= {"A", "B"}

    def test_empty_pdb_raises(self, tmp_path):
        p = tmp_path / "empty.pdb"
        p.write_text("REMARK nothing here\nEND\n")
        with pytest.raises(ValueError, match="no ATOM"):
            receptor.parse_pdb(str(p))

    def test_finds_the_ligand(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        ligs = receptor.find_ligands(atoms)
        assert len(ligs) == 1
        assert ligs[0].resname == "LIG"
        assert ligs[0].n_heavy == 13

    def test_solvent_is_not_mistaken_for_a_ligand(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        names = {lg.resname for lg in receptor.find_ligands(atoms)}
        assert names.isdisjoint({"HOH", "SO4", "GOL", "ZN"})

    def test_cleaning_strips_solvent_and_other_chains(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        clean, rep = receptor.clean_receptor(atoms)
        assert rep["solvent_dropped"] > 0
        assert rep["other_chain_dropped"] > 0
        assert rep["chain_kept"] in ("A", "B")
        assert all(a.resname in receptor.AMINO for a in clean)

    def test_cleaning_can_keep_the_ligand(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        clean, rep = receptor.clean_receptor(atoms, keep_ligand="LIG")
        assert rep["ligand_kept"] == 13
        assert any(a.resname == "LIG" for a in clean)

    def test_non_protein_raises(self, tmp_path):
        p = tmp_path / "junk.pdb"
        p.write_text(
            "HETATM    1  O   HOH A 001       0.000   0.000   0.000  1.00  0.00           O\n"
        )
        atoms = receptor.parse_pdb(str(p))
        with pytest.raises(ValueError, match="no standard amino-acid"):
            receptor.clean_receptor(atoms)

    def test_box_from_ligand_contains_the_ligand(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        lig = receptor.find_ligands(atoms)[0]
        box = receptor.box_from_ligand(lig, padding=5.0)
        c, s = np.array(box.center), np.array(box.size)
        lo, hi = c - s / 2, c + s / 2
        for a in lig.atoms:
            assert np.all(a.xyz >= lo) and np.all(a.xyz <= hi)

    def test_box_has_a_minimum_size(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        box = receptor.box_from_ligand(receptor.find_ligands(atoms)[0], min_size=20.0)
        assert min(box.size) >= 20.0

    def test_box_from_unknown_residues_raises(self, sample_data):
        atoms = receptor.parse_pdb(os.path.join(sample_data, "sample_receptor.pdb"))
        with pytest.raises(ValueError, match="no atoms found"):
            receptor.box_from_residues(atoms, [99999])

    def test_prepare_writes_receptor_and_pdbqt(self, sample_data, tmp_path):
        info = receptor.prepare(
            os.path.join(sample_data, "sample_receptor.pdb"), str(tmp_path)
        )
        assert os.path.exists(info["receptor_pdbqt"])
        assert os.path.exists(info["receptor_pdb"])
        assert "co-crystallised ligand" in info["box"]["source"]
        txt = open(info["receptor_pdbqt"], encoding="utf-8").read()
        assert "ATOM" in txt
        assert " OA" in txt, "polar oxygens should get the OA AutoDock type"

    def test_prepare_box_is_restricted_to_the_kept_chain(self, sample_data, tmp_path):
        # Residues 5-8 exist in both chains; the box must not span both.
        info = receptor.prepare(
            os.path.join(sample_data, "sample_receptor.pdb"),
            str(tmp_path), site_residues=[5, 6, 7, 8],
        )
        assert info["box"]["size_x"] < 40.0

    def test_prepare_warns_on_an_oversized_box(self, sample_data, tmp_path):
        info = receptor.prepare(
            os.path.join(sample_data, "sample_receptor.pdb"),
            str(tmp_path), box_center=(0.0, 0.0, 0.0), box_size=60.0,
        )
        assert any("very large" in w for w in info["warnings"])

    def test_apo_falls_back_to_pocket_detection(self, sample_data, tmp_path):
        src = open(os.path.join(sample_data, "sample_receptor.pdb"), encoding="utf-8").read()
        apo = tmp_path / "apo.pdb"
        apo.write_text("\n".join(ln for ln in src.splitlines() if "LIG A" not in ln) + "\n")
        info = receptor.prepare(str(apo), str(tmp_path / "out"))
        assert "pocket detection" in info["box"]["source"]
        assert any("pocket detection" in w for w in info["warnings"])

    def test_explicit_centre_is_honoured(self, sample_data, tmp_path):
        info = receptor.prepare(
            os.path.join(sample_data, "sample_receptor.pdb"),
            str(tmp_path), box_center=(1.5, 2.5, 3.5), box_size=20.0,
        )
        assert (info["box"]["center_x"], info["box"]["center_y"], info["box"]["center_z"]) == (1.5, 2.5, 3.5)


# ------------------------------------------------------------------- dock


class TestDock:
    def test_vina_absent_is_detected(self):
        assert dock.find_vina("definitely_not_a_real_binary_xyz") is None

    def test_vina_stub_is_found(self, fake_vina):
        assert dock.find_vina() is not None

    def test_ligand_pdbqt_is_written(self, tmp_path):
        p = tmp_path / "lig.pdbqt"
        ok, err = dock.ligand_pdbqt(chem.mol_from_smiles(ASPIRIN), str(p))
        assert ok, err
        assert "ATOM" in p.read_text()

    def test_parses_remark_vina_result(self, tmp_path):
        p = tmp_path / "out.pdbqt"
        p.write_text(
            "MODEL 1\nREMARK VINA RESULT:      -9.10     0.000     0.000\nENDMDL\n"
            "MODEL 2\nREMARK VINA RESULT:      -8.40     1.200     2.300\nENDMDL\n"
        )
        modes = dock.parse_vina_output("", str(p))
        assert [m["affinity"] for m in modes] == [-9.10, -8.40]

    def test_parses_stdout_table(self):
        out = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "    1       -9.10      0.000      0.000\n"
            "    2       -8.40      1.200      2.300\n"
        )
        modes = dock.parse_vina_output(out)
        assert len(modes) == 2
        assert modes[0]["affinity"] == -9.10

    def test_rmsd_of_identical_sets_is_zero(self):
        a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert dock.symmetry_blind_rmsd(a, a) == pytest.approx(0.0)

    def test_rmsd_of_shifted_set(self):
        # Two points shifted by (2,2,2). Nearest-neighbour distances are
        # 3.464 and 3.000, so the RMSD is sqrt((12+9)/2) = 3.240.
        a = np.array([[0.0, 0, 0], [1, 0, 0]])
        assert dock.symmetry_blind_rmsd(a, a + 2.0) == pytest.approx(3.240, abs=0.01)

    def test_rmsd_grows_with_displacement(self):
        a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        near = dock.symmetry_blind_rmsd(a, a + 0.5)
        far = dock.symmetry_blind_rmsd(a, a + 5.0)
        assert near < far

    def test_rmsd_of_empty_is_none(self):
        assert dock.symmetry_blind_rmsd(np.empty((0, 3)), np.ones((2, 3))) is None

    def test_missing_vina_raises_with_a_hint(self, sample_data, tmp_path):
        cfg = dock.DockConfig(vina_exe="nope_not_real_xyz")
        # find_vina still searches PATH, so only assert when nothing is there
        if dock.find_vina(None) is None:
            with pytest.raises(RuntimeError, match="not found"):
                dock.screen(os.path.join(sample_data, "sample_receptor.pdb"),
                            ASPIRIN, out_dir=str(tmp_path), cfg=cfg)

    def test_full_screen_ranks_results(self, sample_data, fake_vina, tmp_path):
        df, rep = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"),
            os.path.join(sample_data, "sample_triage_library.smi"),
            out_dir=str(tmp_path),
            cfg=dock.DockConfig(workers=2, exhaustiveness=2, vina_exe=fake_vina),
        )
        ok = df[df["status"] == "ok"]
        assert len(ok) > 10
        assert ok["affinity"].is_monotonic_increasing, "best (most negative) must come first"
        assert rep["n_docked"] == len(ok)
        assert os.path.exists(os.path.join(tmp_path, "docking_results.csv"))
        assert os.path.exists(os.path.join(tmp_path, "run_report.json"))

    def test_screen_computes_ligand_efficiency(self, sample_data, fake_vina, tmp_path):
        df, _ = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"),
            f"{ASPIRIN} {CAFFEINE}",
            out_dir=str(tmp_path),
            cfg=dock.DockConfig(workers=2, exhaustiveness=2, vina_exe=fake_vina),
        )
        ok = df[df["status"] == "ok"]
        assert "ligand_efficiency" in ok
        expected = -ok["affinity"] / ok["heavy_atoms"]
        assert np.allclose(ok["ligand_efficiency"], expected.round(3))

    def test_screen_joins_triage_columns(self, sample_data, fake_vina, tmp_path):
        df, _ = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"),
            f"{ASPIRIN} {CATECHOL}",
            out_dir=str(tmp_path), with_triage=True,
            cfg=dock.DockConfig(workers=2, exhaustiveness=2, vina_exe=fake_vina),
        )
        for col in ("qed", "sa_score", "n_alerts", "triage_score"):
            assert col in df.columns

    def test_oversized_ligands_are_skipped_not_scored(self, sample_data, fake_vina, tmp_path):
        df, _ = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"),
            f"{ASPIRIN} {LIGNOCERIC}",
            out_dir=str(tmp_path),
            cfg=dock.DockConfig(workers=1, exhaustiveness=2,
                                vina_exe=fake_vina, max_heavy_atoms=20),
        )
        skipped = df[df["status"] == "skipped"]
        assert len(skipped) == 1
        assert "heavy atoms" in skipped.iloc[0]["error"]

    def test_bad_smiles_recorded_not_fatal(self, sample_data, fake_vina, tmp_path):
        df, rep = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"),
            f"{ASPIRIN} totally_invalid",
            out_dir=str(tmp_path),
            cfg=dock.DockConfig(workers=1, exhaustiveness=2, vina_exe=fake_vina),
        )
        assert rep["n_failed"] >= 1
        assert rep["n_docked"] >= 1

    def test_redocking_control_runs(self, sample_data, fake_vina, tmp_path):
        _, rep = dock.screen(
            os.path.join(sample_data, "sample_receptor.pdb"), ASPIRIN,
            out_dir=str(tmp_path),
            cfg=dock.DockConfig(workers=1, exhaustiveness=2, vina_exe=fake_vina),
        )
        ctrl = rep["positive_control"]
        assert ctrl["status"] in ("ok", "skipped", "failed")
        if ctrl["status"] == "ok":
            assert "rmsd_to_crystal" in ctrl
            assert ("pass" in ctrl["verdict"]) or ("FAIL" in ctrl["verdict"])

    def test_control_absent_is_explained_for_apo(self, sample_data, fake_vina, tmp_path):
        src = open(os.path.join(sample_data, "sample_receptor.pdb"), encoding="utf-8").read()
        apo = tmp_path / "apo.pdb"
        apo.write_text("\n".join(ln for ln in src.splitlines() if "LIG A" not in ln) + "\n")
        _, rep = dock.screen(str(apo), ASPIRIN, out_dir=str(tmp_path / "o"),
                             cfg=dock.DockConfig(workers=1, exhaustiveness=2, vina_exe=fake_vina))
        ctrl = rep["positive_control"]
        assert ctrl["status"] == "skipped"
        assert "uncertainty" in ctrl["reason"] or "no redocking control" in ctrl["reason"]


# -------------------------------------------------------------- end to end


class TestCLI:
    def _run(self, *args, env=None):
        return subprocess.run(
            [sys.executable, "-m", "pharma_ops.cli", *args],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, **(env or {})},
        )

    def test_help(self):
        r = self._run("--help")
        assert r.returncode == 0
        assert "triage" in r.stdout and "target2hit" in r.stdout and "dock" in r.stdout

    def test_info(self):
        r = self._run("info")
        assert r.returncode == 0
        assert "rdkit" in r.stdout

    def test_triage_end_to_end(self, sample_data, tmp_path):
        out = str(tmp_path / "t")
        r = self._run("triage", os.path.join(sample_data, "sample_triage_library.smi"),
                      "-o", out, "--show", "5")
        assert r.returncode == 0, r.stderr
        for ext in (".csv", ".html", "_summary.json"):
            assert os.path.exists(out + ext)
        df = pd.read_csv(out + ".csv")
        assert len(df) == 15 and "triage_score" in df.columns

    def test_triage_filters_apply(self, sample_data, tmp_path):
        out = str(tmp_path / "f")
        r = self._run("triage", os.path.join(sample_data, "sample_triage_library.smi"),
                      "-o", out, "--alert-free")
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out + ".csv")
        assert (df["n_alerts"] == 0).all()

    def test_triage_weights_flag_end_to_end(self, sample_data, tmp_path):
        out = str(tmp_path / "w")
        r = self._run("triage", os.path.join(sample_data, "sample_triage_library.smi"),
                      "-o", out, "--weights", "qed=0.6", "alerts=0.4", "--show", "3")
        assert r.returncode == 0, r.stderr
        summary = json.loads(open(out + "_summary.json").read())["summary"]
        assert summary["weights_are_default"] is False
        df = pd.read_csv(out + ".csv")
        assert df["triage_score"].between(0, 100).all()
        assert "custom weights" in open(out + ".html").read()

    def test_triage_rejects_a_bad_weight_key(self, sample_data, tmp_path):
        r = self._run("triage", os.path.join(sample_data, "sample_triage_library.smi"),
                      "-o", str(tmp_path / "bad"), "--weights", "qedd=0.9")
        assert r.returncode == 1
        assert "unknown triage weight" in r.stderr

    def test_triage_bad_input_fails_cleanly(self, tmp_path):
        r = self._run("triage", "/nonexistent/file.smi", "-o", str(tmp_path / "x"))
        assert r.returncode == 1
        assert "error" in r.stderr.lower()

    def test_target2hit_offline(self, sample_data, tmp_path):
        out = str(tmp_path / "t2h")
        r = self._run("target2hit",
                      "--actives-file", os.path.join(sample_data, "sample_actives.csv"),
                      "--library", os.path.join(sample_data, "sample_library.smi"),
                      "-o", out, "--no-structure", "--show", "3")
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(out, "report.json"))
        scored = pd.read_csv(os.path.join(out, "scored_library.csv"))
        assert "in_applicability_domain" in scored.columns
        assert "applicability domain" in r.stdout

    def test_dock_without_vina_exits_2(self, sample_data, tmp_path):
        if dock.find_vina(None) is not None:
            pytest.skip("a vina executable is on PATH")
        r = self._run("dock", os.path.join(sample_data, "sample_receptor.pdb"),
                      ASPIRIN, "-o", str(tmp_path / "d"))
        assert r.returncode == 2
        assert "not found" in r.stderr

    def test_dock_end_to_end(self, sample_data, fake_vina, tmp_path):
        out = str(tmp_path / "d")
        r = self._run("dock", os.path.join(sample_data, "sample_receptor.pdb"),
                      os.path.join(sample_data, "sample_triage_library.smi"),
                      "-o", out, "--show", "5", "--workers", "2",
                      "--exhaustiveness", "2", "--vina-exe", fake_vina)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(out, "docking_results.csv"))
        assert os.path.exists(os.path.join(out, "docking_report.html"))
        assert "ligand_efficiency" in r.stdout


class TestReportHTML:
    def test_render_is_self_contained(self):
        from pharma_ops.report import render

        df = pd.DataFrame({"rank": [1], "mol_id": ["x"], "triage_score": [80.0]})
        html = render("T", "sub", df, {"n_valid": 1}, [("n_valid", "N")], ["rank", "mol_id"])
        assert html.startswith("<!DOCTYPE html>")
        assert "<style>" in html and "http" not in html.split("<footer>")[0].replace("https://", "")
        assert "not predictions of potency" in html

    def test_render_escapes_input(self):
        from pharma_ops.report import render

        df = pd.DataFrame({"mol_id": ["<script>alert(1)</script>"]})
        html = render("T", "s", df, {}, [], ["mol_id"])
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
