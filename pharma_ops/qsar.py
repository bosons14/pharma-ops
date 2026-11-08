"""QSAR modelling with honest validation.

Two things make most published QSAR numbers optimistic, and both are handled
here by default:

1. **Random splits leak.** Analogue series share scaffolds, so a random
   train/test split puts near-duplicates on both sides and the reported R²
   flatters the model. We default to a Bemis-Murcko *scaffold split*, which
   is the harder and more realistic setting.

2. **No applicability domain.** A model will happily score a compound
   nothing like its training set. Every prediction here carries the Tanimoto
   similarity to its nearest training neighbour, and predictions below a
   threshold are marked out-of-domain. An out-of-domain score is not a
   prediction, it is an extrapolation.

The model is a random forest on Morgan fingerprints plus a descriptor block.
That is a strong, boring baseline. It is not state of the art, and it should
not be the last model you ever run -- but it will not quietly mislead you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .chem import featurise, mol_from_smiles

log = logging.getLogger("pharma_ops.qsar")


# ----------------------------------------------------------------------------
# Split protocols
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """What a split protocol does, and therefore what may be claimed about it.

    Verdict wording is derived from this record rather than written by hand at
    the point of reporting. A split that does not declare how it should be
    described cannot run (see ``resolve_split``), so a result can never be
    labelled with a protocol other than the one that produced it.

    This exists because the alternative failed in practice: the wording was
    once hardcoded, and a random-split run was reported as a scaffold split --
    which put the reassuring label on the inflated number, the precise
    inversion this module is written to prevent.
    """

    name: str
    held_out: str          # noun phrase for what the test set withholds
    optimistic: bool       # does it leave analogues of training data in test?
    description: str
    caveat: str = ""       # prepended to the verdict when optimistic


SPLITS: dict[str, SplitSpec] = {
    "scaffold": SplitSpec(
        name="scaffold",
        held_out="held-out scaffolds",
        optimistic=False,
        description="whole Bemis-Murcko scaffold groups assigned to one side only",
    ),
    "random": SplitSpec(
        name="random",
        held_out="held-out compounds",
        optimistic=True,
        description="compounds assigned independently of scaffold",
        caveat=(
            "RANDOM SPLIT -- these numbers are optimistic and are NOT an estimate of "
            "performance on new chemistry: analogues of the training compounds sit in "
            "the test set, so the model is partly being rewarded for recognising "
            "scaffolds it has already seen. Read them as an upper bound."
        ),
    ),
}


def resolve_split(split: str | None) -> SplitSpec:
    """Look up a split protocol, refusing anything undeclared.

    Failing here is deliberate. Previously an unrecognised name fell through to
    random partitioning while the report still described a scaffold split, so
    the run and its label disagreed silently. A protocol with no SplitSpec has
    no agreed wording, so it does not run.
    """
    spec = SPLITS.get(split or "")
    if spec is None:
        raise ValueError(
            f"unknown split protocol {split!r}; valid options are "
            f"{sorted(SPLITS)}. Add a SplitSpec for a new protocol so its "
            "results can be described accurately."
        )
    return spec


def murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:  # noqa: BLE001
        return ""


def scaffold_split(
    smiles: Sequence[str], test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic scaffold split: the largest scaffold groups go to train.

    Returns index arrays (train_idx, test_idx).
    """
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(murcko_scaffold(smi), []).append(i)

    # Big groups first, so the test set is made of many small/rare scaffolds
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_test_target = int(round(test_frac * len(smiles)))

    train: list[int] = []
    test: list[int] = []
    for grp in ordered:
        if len(test) < n_test_target:
            test.extend(grp)
        else:
            train.extend(grp)

    if not train or not test:  # degenerate tiny datasets
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(smiles))
        cut = max(1, int(round(test_frac * len(smiles))))
        return idx[cut:], idx[:cut]

    return np.asarray(sorted(train)), np.asarray(sorted(test))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@dataclass
class QSARModel:
    task: str = "regression"  # or "classification"
    radius: int = 2
    n_bits: int = 2048
    n_estimators: int = 400
    seed: int = 0
    ad_threshold: float = 0.30  # min Tanimoto to nearest training neighbour

    model: object | None = None
    train_fp: np.ndarray | None = None
    train_smiles: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    activity_threshold: float | None = None

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        smiles: Sequence[str],
        y: Sequence[float],
        test_frac: float = 0.2,
        split: str = "scaffold",
    ) -> dict:
        mols, keep = [], []
        for i, smi in enumerate(smiles):
            m = mol_from_smiles(smi)
            if m is not None:
                mols.append(m)
                keep.append(i)
        if len(mols) < 10:
            raise ValueError(
                f"only {len(mols)} parseable compounds; need at least 10 "
                "(and realistically 100+) to fit anything meaningful"
            )

        # Validate the protocol before anything runs, so the metadata can never
        # describe a split that did not happen.
        spec = resolve_split(split)

        smiles_k = [smiles[i] for i in keep]
        y_arr = np.asarray([y[i] for i in keep], dtype=float)
        X = featurise(mols, self.radius, self.n_bits)

        if spec.name == "scaffold":
            tr, te = scaffold_split(smiles_k, test_frac, self.seed)
        else:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(smiles_k))
            cut = max(1, int(round(test_frac * len(smiles_k))))
            te, tr = idx[:cut], idx[cut:]

        if self.task == "classification":
            if self.activity_threshold is None:
                self.activity_threshold = 6.0  # pChEMBL 6 == 1 uM, the usual cut
            y_arr = (y_arr >= self.activity_threshold).astype(int)
            if len(np.unique(y_arr[tr])) < 2:
                raise ValueError(
                    "training set is single-class after thresholding at "
                    f"pChEMBL {self.activity_threshold}; pick a different --activity-threshold"
                )
            est = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.seed,
                n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2,
            )
        else:
            est = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.seed,
                n_jobs=-1, min_samples_leaf=2,
            )

        est.fit(X[tr], y_arr[tr])
        self.model = est
        self.train_fp = X[tr]
        self.train_smiles = [smiles_k[i] for i in tr]

        pred = est.predict(X[te])
        m: dict = {
            "task": self.task,
            "split": spec.name,
            "split_description": spec.description,
            "split_is_optimistic": spec.optimistic,
            "ad_threshold": self.ad_threshold,
            "n_total": len(smiles_k),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_unparseable": len(smiles) - len(smiles_k),
            "n_train_scaffolds": len({murcko_scaffold(s) for s in self.train_smiles}),
        }
        if self.task == "classification":
            proba = est.predict_proba(X[te])[:, 1]
            m["roc_auc"] = _safe(roc_auc_score, y_arr[te], proba)
            m["balanced_accuracy"] = _safe(balanced_accuracy_score, y_arr[te], pred)
            m["pct_active_train"] = round(100.0 * float(y_arr[tr].mean()), 1)
            m["activity_threshold"] = self.activity_threshold
        else:
            m["r2"] = _safe(r2_score, y_arr[te], pred)
            m["rmse"] = _safe(lambda a, b: float(np.sqrt(mean_squared_error(a, b))), y_arr[te], pred)
            m["mae"] = _safe(mean_absolute_error, y_arr[te], pred)
            m["spearman"] = _spearman(y_arr[te], pred)
            m["y_train_range"] = [round(float(y_arr[tr].min()), 2), round(float(y_arr[tr].max()), 2)]

        m["interpretation"] = self._interpret(m)
        self.metrics = m
        return m

    @staticmethod
    def _interpret(m: dict) -> str:
        """Plain-English verdict on the metrics, honest about which split ran.

        A random split puts analogues of training compounds in the test set, so
        its numbers measure recognition as much as generalisation. They are an
        upper bound, not an estimate of performance on new chemistry, and this
        says so rather than letting a flattering number pass unlabelled.
        """
        try:
            spec = resolve_split(m.get("split"))
        except ValueError:
            # Metrics from an unrecognised protocol: describe nothing.
            return (
                f"Split protocol {m.get('split')!r} is not recognised, so these metrics "
                "cannot be characterised. Treat the model as unvalidated."
            )
        core = QSARModel._verdict(m, spec.held_out)
        if not spec.optimistic:
            return core
        return (
            f"{spec.caveat} {core} "
            "Re-run with the default --split scaffold for the honest figure."
        )

    @staticmethod
    def _verdict(m: dict, unseen: str) -> str:
        if m["task"] == "classification":
            auc = m.get("roc_auc")
            if auc is None:
                return "Test set too small or single-class to score; treat the model as unvalidated."
            if auc >= 0.85:
                return f"Strong ranking on {unseen}; usable for prioritisation."
            if auc >= 0.70:
                return "Moderate; useful for coarse triage, not for go/no-go decisions."
            return f"Weak; the model is close to guessing on {unseen} -- do not rely on it."
        r2, rho = m.get("r2"), m.get("spearman")
        if r2 is None:
            return "Test set too small to score; treat the model as unvalidated."
        if r2 >= 0.5:
            return f"Reasonable performance on {unseen}; usable for ranking within ~1 log unit."
        if r2 >= 0.2:
            return "Weak but non-random; use rank order only, never the absolute values."
        # Negative R2 with decent rank correlation is common on scaffold splits:
        # the model has the ordering roughly right but the absolute scale wrong.
        if rho is not None and rho >= 0.4:
            return (
                f"R2 is poor ({r2}) so the predicted values are NOT calibrated -- do not "
                f"quote them as potencies. Rank correlation is usable (Spearman {rho}), so "
                "the model can order a library, which is all you need for picking what to assay."
            )
        return f"No useful signal on {unseen} -- more or more diverse data needed."

    # -- prediction -------------------------------------------------------

    def predict(self, smiles: Sequence[str]) -> pd.DataFrame:
        if self.model is None or self.train_fp is None:
            raise RuntimeError("model is not fitted")

        rows = []
        mols, valid_idx = [], []
        for i, smi in enumerate(smiles):
            m = mol_from_smiles(smi)
            if m is None:
                rows.append({"smiles": smi, "status": "error", "error": "unparseable SMILES"})
            else:
                mols.append(m)
                valid_idx.append(i)

        if mols:
            X = featurise(mols, self.radius, self.n_bits)
            if self.task == "classification":
                scores = self.model.predict_proba(X)[:, 1]
                col = "p_active"
            else:
                scores = self.model.predict(X)
                col = "predicted_pchembl"

            sims, nn = self._nearest_neighbour(X)
            for j, i in enumerate(valid_idx):
                in_domain = bool(sims[j] >= self.ad_threshold)
                rows.append(
                    {
                        "smiles": smiles[i],
                        "status": "ok",
                        col: round(float(scores[j]), 3),
                        "nn_similarity": round(float(sims[j]), 3),
                        "nn_train_smiles": nn[j],
                        "in_applicability_domain": in_domain,
                        "confidence": _conf(sims[j], self.ad_threshold),
                    }
                )

        df = pd.DataFrame(rows)
        sort_col = "p_active" if self.task == "classification" else "predicted_pchembl"
        if sort_col in df:
            df = df.sort_values([sort_col], ascending=False, na_position="last").reset_index(drop=True)
        return df

    def _nearest_neighbour(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Max Tanimoto to the training set, on the fingerprint block only."""
        nb = self.n_bits
        A = (X[:, :nb] > 0).astype(np.float32)
        B = (self.train_fp[:, :nb] > 0).astype(np.float32)  # type: ignore[index]
        inter = A @ B.T
        a_sum = A.sum(axis=1, keepdims=True)
        b_sum = B.sum(axis=1, keepdims=True).T
        union = a_sum + b_sum - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        best = sim.argmax(axis=1)
        return sim.max(axis=1), [self.train_smiles[i] for i in best]


def _conf(sim: float, threshold: float = 0.30) -> str:
    """Confidence wording, with its bands derived from the domain threshold.

    The bands used to be fixed at 0.70/0.45/0.30 while ``in_applicability_domain``
    read the configurable ``ad_threshold``. At a threshold of 0.70 a similarity
    of 0.55 was therefore reported as out of domain AND "medium" confidence, in
    the same row. Deriving the bands from the threshold makes that
    contradiction unrepresentable.
    """
    if sim < threshold:
        return "out of domain (extrapolation, not a prediction)"
    span = 1.0 - threshold
    if span <= 0:
        return "high (close analogue in training set)"
    over = (sim - threshold) / span
    if over >= 0.66:
        return "high (close analogue in training set)"
    if over >= 0.33:
        return "medium"
    return "low"


def _safe(fn, a, b):
    try:
        if len(np.unique(a)) < 2:
            return None
        return round(float(fn(a, b)), 3)
    except Exception:  # noqa: BLE001
        return None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr

        rho = spearmanr(a, b).statistic
        return None if np.isnan(rho) else round(float(rho), 3)
    except Exception:  # noqa: BLE001
        return None
