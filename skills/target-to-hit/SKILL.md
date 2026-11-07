---
name: target-to-hit
description: This skill should be used when the user names a protein target and wants known actives, bioactivity data or a predictive model — "find inhibitors of EGFR", "pull ChEMBL actives for this target", "build a QSAR model", "score my library against JAK2", "what's known to bind this protein", "get me a structure for docking" — or runs /target-to-hit. Covers ChEMBL target resolution and bioactivity retrieval, scaffold-split QSAR with applicability domain, and PDB/AlphaFold structure retrieval.
---

# Target to hit

Turn a target name into known actives, a validated QSAR model, and scored
candidates.

## When to use this

- The user names a protein and wants compounds known to act on it.
- They want a model trained on public bioactivity data.
- They want to score their own library against a target.
- They need a structure to dock into (this fetches it; `virtual-screening`
  does the docking).

## First run: make sure the CLI exists

Installing the plugin ships these instructions and the Python source, but it
does not install the `pharma-ops` command. Before the first call in a
session, check:

```bash
pharma-ops info || pip install -e "${CLAUDE_PLUGIN_ROOT}"
```

`pharma-ops info` also reports whether ChEMBL, RCSB and AlphaFold are
reachable. If they are not, go straight to the `--actives-file` route rather
than letting a network call fail mid-run.

## Running it

```bash
# Full online path: resolve target, pull actives, fit, fetch structure
pharma-ops target2hit --target EGFR -o egfr_out

# Score a candidate library at the same time
pharma-ops target2hit --target CHEMBL203 --library candidates.smi -o egfr_out

# Classification instead of regression, 100 nM active/inactive cut
pharma-ops target2hit --target JAK2 --task classification --activity-threshold 7.0

# Offline, or with proprietary data
pharma-ops target2hit --actives-file my_actives.csv --library candidates.smi --no-structure
```

`--actives-file` takes any CSV with a SMILES column and an activity column.
pIC50/pKi/pChEMBL values are used directly; raw nM IC50/Ki values are
detected and converted.

Outputs in `--out-dir`: `actives.csv`, `bioactivities_raw.csv`,
`report.json`, `scored_library.csv`, `scored_library.html`, and
`structures/` when a structure was retrieved.

## Reading the model metrics

**Always report the metrics before the predictions.** A model's scores mean
nothing without them, and this is the step people skip.

| Metric | Read it as |
|---|---|
| `split: scaffold` | Default and correct. Test compounds have scaffolds unseen in training. |
| `r2` | Variance explained on held-out scaffolds. **Often negative, and that is normal** on a genuine scaffold split. Negative R² means the predicted *values* are uncalibrated. |
| `spearman` | Rank correlation. This is the one that matters for prioritisation — ≥0.4 means the model can order a library even when R² is poor. |
| `rmse` / `mae` | Error in log units. RMSE 0.7 means roughly 5-fold uncertainty in potency. |
| `roc_auc` | Classification only. ≥0.85 strong, 0.70-0.85 moderate, <0.70 near guessing. |
| `n_train_scaffolds` | Chemical diversity of the training set. A model trained on 5 scaffolds will not generalise. |
| `interpretation` | Written verdict — quote it. |

## Applicability domain

Every prediction carries `nn_similarity` (Tanimoto to the nearest training
compound) and `in_applicability_domain`. Compounds below 0.30 similarity are
flagged out-of-domain.

**An out-of-domain score is an extrapolation, not a prediction.** Say so.
If most of the user's library is out of domain, the honest answer is that
this model cannot assess their compounds, and that more diverse training
data or a different approach is needed — not a ranked list with a caveat
buried at the bottom.

## Structures

Prefers an experimental RCSB PDB entry, sorted by resolution; falls back to
AlphaFold DB. The report says which, and flags the caveats:

- An **experimental holo structure** is best — the bound ligand defines the
  pocket, which `pharma-ops dock` then uses automatically.
- An **apo structure** has no ligand, so the box must come from known
  site residues or geometric detection.
- An **AlphaFold model** has no ligand, may have a collapsed or occluded
  pocket, and carries pLDDT confidence in the B-factor column. Docking into
  one is substantially less reliable. Say this rather than quietly docking.

## What to tell the user

- ChEMBL aggregates literature data of mixed quality. Assay conditions
  differ between papers, so pChEMBL values for the same compound can vary by
  a log unit or more. `pchembl_std` and `n_measurements` show where that
  happened.
- Only exact (`=`) dose-response measurements are used. Censored values
  (`>10000 nM`) are dropped, which biases the set toward actives: ChEMBL
  holds what people published, and inactives are published far less often.
- A model trained only on actives cannot tell you what is *inactive*. For
  genuine virtual screening, decoys (DUD-E, DEKOIS) or measured inactives
  are needed.
- Public bioactivity data reflects where the field has looked, not where the
  chemistry is good.

## Rate limits

ChEMBL is a free public service. Keep `--max-records` modest for
exploration. For repeated large pulls, point the user at the ChEMBL SQLite
dump rather than the REST API.
