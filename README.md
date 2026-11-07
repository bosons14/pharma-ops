# pharma-ops

Early-stage drug discovery tooling: compound triage, target-to-hit modelling,
and structure-based virtual screening. CLI plus a Claude Code plugin.

Built on public data (ChEMBL, RCSB PDB, AlphaFold DB) and open-source
software (RDKit, scikit-learn, AutoDock Vina, Meeko).

---

## What this is, and what it is not

This toolkit computes physicochemical properties, applies published
empirical rules, fits statistical models to public bioactivity data, and
runs an empirical docking score. Those are real, useful, standard
techniques, and together they do one job: **decide which compounds to look
at first.**

They do not predict whether a compound will work, or whether it is safe.
Nothing here models toxicity, hERG, CYP inhibition, reactive metabolites,
pharmacokinetics in a living animal, or clinical outcome. A compound that
scores well here is a compound worth an assay. A compound that scores badly
is worth a second look before discarding, because every filter here has
known false positives.

Real drug discovery is assays, medicinal chemists, animal models, and years.
This is the cheap pass that happens before any of that, and it is only
worth running if its output is read with that in mind. Specific caveats are
in each tool's section below, and in the generated HTML reports.

---

## Install

```bash
git clone https://github.com/bosons14/pharma-ops
cd pharma-ops
pip install -e .
```

RDKit, NumPy, pandas, scikit-learn, requests and Jinja2 come with the
install. Docking needs two more things:

```bash
pip install meeko gemmi                  # ligand PDBQT preparation
conda install -c conda-forge vina        # the Vina binary itself
```

Vina alternatives: `smina`, `qvina2` — pass `--vina-exe`.

Check what is available:

```bash
pharma-ops info
```

This reports dependency versions, whether Vina is on PATH, and whether
ChEMBL/PDB/AlphaFold are reachable. On a restricted network the data
sources will show as unreachable; tools 1 and 3 work entirely offline, and
tool 2 works offline via `--actives-file`.

---

## Tool 1: compound triage

```bash
pharma-ops triage library.smi -o triage
pharma-ops triage "CC(=O)Oc1ccccc1C(=O)O"
pharma-ops triage compounds.csv --smiles-col canonical_smiles --alert-free --min-score 60
```

Input: `.smi`, `.csv`, `.tsv`, `.sdf`, `.sdf.gz`, or a literal SMILES.
Columns are auto-detected. Output: CSV, a self-contained HTML report, and a
JSON summary.

Computes:

- **Descriptors** — MW, cLogP, TPSA, HBD/HBA, rotatable bonds, rings,
  Fsp³, stereocentres, molar refractivity
- **Drug-likeness** — Lipinski Ro5, Veber, Ghose, Egan, lead-likeness,
  rule of three, and QED
- **Synthetic accessibility** — SA score, 1 (easy) to 10 (hard)
- **Structural alerts** — PAINS, Brenk, and NIH catalogues, with the
  matching filter named
- **ADMET proxies** — Delaney ESOL solubility, Egan GI absorption,
  TPSA/logP BBB heuristic, a crude P-gp efflux flag
- **A 0-100 composite score** blending the above, with documented and
  retunable weights

### Caveats

Lipinski's rule of five came from a 1997 analysis of compounds in
development. It is a heuristic about oral bioavailability, not a law, and it
does not apply to natural products, macrocycles, PROTACs, injectables or
covalent inhibitors.

PAINS filters flag substructures that are frequent hitters in certain assay
formats. A hit means *check for assay interference*, not *discard*. Several
approved drugs match a PAINS filter.

The ADMET columns are published regressions and cutoffs on one or two
descriptors. They correlate loosely with measured values. Replace them with
measured data as soon as you have any.

The composite weights are a starting point with no special authority. Retune
them for your programme.

---

## Tool 2: target to hit

```bash
# Online: resolve target, pull actives, fit a model, fetch a structure
pharma-ops target2hit --target EGFR -o egfr_out

# Score your own library against it
pharma-ops target2hit --target CHEMBL203 --library candidates.smi -o egfr_out

# Classification at a 100 nM cut
pharma-ops target2hit --target JAK2 --task classification --activity-threshold 7.0

# Offline, or with proprietary data
pharma-ops target2hit --actives-file my_actives.csv --library candidates.smi --no-structure
```

Resolves a free-text target name or CHEMBL id against ChEMBL, pulls
dose-response bioactivities, collapses repeat measurements per compound,
fits a random forest on Morgan fingerprints plus descriptors, and scores a
candidate library. Fetches an experimental PDB entry where one exists,
falling back to AlphaFold DB.

`--actives-file` accepts any CSV with a SMILES column and an activity
column. pIC50/pKi/pChEMBL values are used as-is; raw nM values are detected
and converted.

### Two things this does that most QSAR code does not

**Scaffold splitting by default.** Analogue series share scaffolds, so a
random train/test split puts near-duplicates on both sides and the reported
R² is inflated. The default split assigns whole Bemis-Murcko scaffold groups
to one side or the other. This is harder and much more honest. A consequence
worth knowing: **a negative R² on a scaffold split is common and is not a
bug.** It means the predicted values are uncalibrated. Check `spearman` —
if rank correlation holds up, the model can still order a library, which is
all prioritisation needs.

**An applicability domain on every prediction.** Each scored compound
carries `nn_similarity` (Tanimoto to its nearest training neighbour) and
`in_applicability_domain`. Below 0.30 similarity the score is an
extrapolation, not a prediction, and is labelled as such. If most of a
library falls out of domain, the honest conclusion is that the model cannot
assess those compounds.

### Caveats

ChEMBL aggregates literature data of mixed quality. Assay conditions differ
between papers, so the same compound can vary by more than a log unit;
`pchembl_std` and `n_measurements` show where that happened.

Only exact (`=`) measurements are kept. Censored values (`>10000 nM`) cannot
serve as regression targets. This biases the set toward actives, because
inactives are published far less often. **A model trained only on actives
cannot tell you what is inactive** — for genuine virtual screening you need
decoys (DUD-E, DEKOIS) or measured inactives.

Public bioactivity data reflects where the field has looked, not where the
chemistry is good.

For repeated large pulls, use the ChEMBL SQLite dump rather than the REST
API.

---

## Tool 3: virtual screening

```bash
# Holo structure: the box comes from the co-crystallised ligand automatically
pharma-ops dock receptor.pdb library.smi -o dock_out

# Choose the reference ligand and chain
pharma-ops dock 1M17.pdb library.smi --ligand-resname AQ4 --chain A

# Apo structure: define the site from known residues
pharma-ops dock apo.pdb library.smi --site-residues 745 790 797 855

# Explicit box, more search effort, more parallelism
pharma-ops dock rec.pdb lib.smi --box-center 12.5 -3.2 44.1 --box-size 24 \
  --exhaustiveness 16 --workers 8
```

Pipeline: clean the receptor (strip waters, buffers, ions, extra chains and
altlocs), define a box, 3D-embed and protonate each ligand via Meeko, dock
in parallel with Vina, rank by affinity, compute ligand efficiency, join the
triage descriptors, and run a redocking positive control.

### Box definition, most to least reliable

1. **Co-crystallised ligand** — automatic when the PDB has one. Best option.
2. **Known site residues** (`--site-residues`) — reliable when documented.
3. **Explicit centre** (`--box-center`).
4. **Geometric pocket detection** — automatic fallback for apo structures.
   Finds the largest buried cavity, which is **not necessarily the
   functional site**. Flagged in the report. Verify it before trusting the
   run.

Prep warns when a box exceeds ~40 Å on an edge or ~27000 Å³, because Vina's
search degrades in large boxes.

### The redocking control

When the receptor has a co-crystallised ligand, it is redocked and compared
to its crystal pose.

- **RMSD ≤ 2 Å** — the box, protonation and receptor prep can reproduce a
  known binding mode.
- **RMSD > 2 Å** — every score in the run is suspect. Fix the prep first.
- **No control possible** (apo or AlphaFold) — there is no evidence the
  setup works at all.

This is the most informative check in structure-based screening and the one
most often skipped. Read it before the hit list.

### Caveats

A Vina score is **not a binding free energy**. It is an empirical function
fitted to a set of known complexes; the kcal/mol units are convention, not
derivation.

**Score does not predict potency.** Vina's job is enrichment — getting
actives into the top few percent of a library — and it does that reasonably
well. Within a congeneric series, correlation with measured potency is
typically r ≈ 0.3-0.5 and often worse. Never convert a score to a predicted
IC50.

**Differences under ~1 kcal/mol are noise.** Do not read meaning into the
gaps among your top 10.

**Raw scores favour large, greasy molecules**, which make more contacts.
This is why `ligand_efficiency` (`-affinity / heavy_atoms`) is reported
alongside, and why you should check whether your top hits are simply the
biggest compounds. The sample run in `examples/` demonstrates exactly this
artifact.

**Look at the poses.** A good score with a nonsense pose is a failure, not a
hit. Pose files are in `poses/`.

**The receptor is rigid and solvation is implicit.** No induced fit, no
bridging waters, no entropy. Sites that reorganise on binding are modelled
badly.

---

## Chaining the tools

Cull cheaply before docking expensively:

```bash
pharma-ops triage big_library.smi -o clean --alert-free --min-score 55 --top 2000
pharma-ops dock receptor.pdb clean.csv --smiles-col smiles -o dock_out
```

A QSAR score and a docking score are independent lines of evidence.
Compounds ranked well by both are the most defensible picks, and
disagreement between them is informative.

---

## Claude Code plugin

The repo is also a Claude Code plugin. Add it as a marketplace:

```
/plugin marketplace add bosons14/pharma-ops
/plugin install pharma-ops
```

Provides three skills (`compound-triage`, `target-to-hit`,
`virtual-screening`) and three slash commands (`/triage`, `/target2hit`,
`/dock`). The skills carry the interpretation guidance, so Claude reports
model metrics and the redocking control before showing any ranking.

---

## Development

```bash
pip install -e ".[dev]"
python examples/make_sample_data.py     # synthetic library + actives
python examples/make_sample_pdb.py      # synthetic receptor + ligand
python -m pytest tests/ -q
```

78 tests, all offline. Docking is tested against `tests/fake_vina.py`, a
stub that emits real-format Vina output, so the orchestration, parsing,
ranking and control logic are covered without the binary. Vina's physics is
Vina's business.

The sample data is synthetic — a combinatorial library over eight scaffolds
with a made-up SAR, and a receptor built from a short peptide. It exists to
exercise the code, not to represent chemistry.

### Layout

```
pharma_ops/
  chem.py         I/O, standardisation, fingerprints
  triage.py       tool 1: descriptors, rules, alerts, ADMET proxies
  sources.py      ChEMBL, UniProt, RCSB, AlphaFold clients
  qsar.py         scaffold split, random forest, applicability domain
  target2hit.py   tool 2 orchestration
  receptor.py     PDB parsing, cleaning, box definition, pocket detection
  dock.py         tool 3: Meeko prep, Vina runs, controls, ranking
  report.py       self-contained HTML reports
  cli.py          command line interface
skills/           Claude Code skills
commands/         Claude Code slash commands
```

---

## Licence and citations

MIT.

If you publish work using this, cite the underlying methods, not this
wrapper: RDKit; Lipinski *et al.* 1997; Veber *et al.* 2002; Bickerton
*et al.* 2012 (QED); Ertl & Schuffenhauer 2009 (SA score); Baell & Holloway
2010 (PAINS); Brenk *et al.* 2008; Delaney 2004 (ESOL); Trott & Olson 2010
and Eberhardt *et al.* 2021 (AutoDock Vina); Forli *et al.* 2016 (Meeko);
Zdrazil *et al.* 2024 (ChEMBL); Varadi *et al.* 2024 (AlphaFold DB).
