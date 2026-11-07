---
name: virtual-screening
description: This skill should be used when the user wants structure-based docking — "dock these compounds", "run a virtual screen", "will this bind", "what's the binding affinity", "prepare this PDB for docking", "find the binding site", "where's the pocket", "set up a Vina run", "redock the crystal ligand" — or runs /virtual-screening. Covers receptor cleaning, box definition from a co-crystallised ligand or site residues, apo pocket detection, Meeko ligand prep, parallel AutoDock Vina docking, ligand efficiency and the redocking control.
---

# Structure-based virtual screening

Dock a ligand library into a receptor and rank the results, with the
controls that make the ranking worth anything.

## Prerequisites

Two things are needed and neither comes with the plugin: the `pharma-ops`
CLI itself, and the Vina binary. Check both before the first call:

```bash
pharma-ops info || pip install -e "${CLAUDE_PLUGIN_ROOT}"
```

`pharma-ops info` prints the Vina path, or says it is not found. If it is
missing:

```bash
conda install -c conda-forge vina      # or: smina, then --vina-exe smina
pip install meeko gemmi                # ligand PDBQT preparation
```

Docking cannot run without Vina. Say so and stop rather than attempting it.

## Running it

```bash
# Simplest case: holo structure, box from the co-crystallised ligand
pharma-ops dock receptor.pdb library.smi -o dock_out

# Pick which ligand defines the box, and which chain
pharma-ops dock 1M17.pdb library.smi --ligand-resname AQ4 --chain A

# Apo structure: define the site from known residues
pharma-ops dock apo.pdb library.smi --site-residues 745 790 797 855

# Explicit box, higher search effort, more parallelism
pharma-ops dock rec.pdb lib.smi --box-center 12.5 -3.2 44.1 --box-size 24 \
  --exhaustiveness 16 --workers 8
```

Outputs in `--out-dir`: `docking_results.csv`, `docking_report.html`,
`run_report.json`, `receptor/`, `ligands/`, `poses/`, `control/`.

## Box definition, in order of reliability

1. **Co-crystallised ligand** — used automatically when the PDB has one.
   Best option: the box is centred on a site known to bind something.
2. **Known site residues** (`--site-residues`) — reliable when the catalytic
   or binding residues are documented.
3. **Explicit centre** (`--box-center`) — when coordinates are known.
4. **Geometric pocket detection** — automatic fallback for apo structures.
   Finds the largest buried cavity, which is **not necessarily the
   functional site**. The report flags this. Always ask the user to verify
   it against the literature before trusting the run.

Keep boxes small. Vina's search degrades in large boxes; prep warns above
~40 Å on an edge or ~27000 Å³.

## The redocking control — do not skip it

When the receptor has a co-crystallised ligand, the run redocks it and
compares to the crystal pose.

- **RMSD ≤ 2 Å: pass.** The box, protonation and receptor prep can reproduce
  a known binding mode.
- **RMSD > 2 Å: fail.** Every other score in the run is suspect. Fix the
  prep before interpreting anything — check protonation, the box, whether
  the right chain was kept, whether a cofactor or metal was wrongly stripped.
- **No control possible** (apo or predicted structure): there is *no
  evidence* the setup works. Say this explicitly.

Report the control before the hit list. A ranked table from an unvalidated
setup is worse than no table, because it looks like a result.

## Reading the results

| Column | Meaning |
|---|---|
| `affinity` | Best-mode Vina score, kcal/mol. More negative is better. |
| `ligand_efficiency` | `-affinity / heavy_atoms`. Use this alongside raw score. |
| `mode_spread` | Score range across poses. A wide spread means the pose is ambiguous. |
| `qed`, `sa_score`, `n_alerts`, `triage_score` | Joined from the triage tool. |

## What to tell the user

This matters more here than anywhere else in the toolkit, because docking
scores look authoritative and are not:

- **A Vina score is not a binding free energy.** It is an empirical
  function fitted to a training set of complexes. The units are kcal/mol by
  convention, not by derivation.
- **Score does not predict potency.** In retrospective benchmarks Vina
  enriches actives into the top few percent of a library, which is what it
  is for. Within a congeneric series, correlation with measured potency is
  typically r ≈ 0.3-0.5, often worse. Never convert a score to a predicted
  IC50.
- **Differences under ~1 kcal/mol are noise.** Do not rank-order the top 10
  as though the gaps mean anything.
- **Raw scores favour large, greasy molecules.** This is the classic
  docking artefact: more atoms make more contacts. Always show
  `ligand_efficiency` next to `affinity`, and flag it when the top hits are
  simply the biggest compounds.
- **Poses must be looked at.** A good score with a nonsense pose — a polar
  group buried in a hydrophobic pocket, a ring clashing through a
  sidechain — is a failure, not a hit. Pose files are in `poses/`.
- **The receptor is rigid and the water is implicit.** No induced fit, no
  explicit bridging waters, no entropy. Sites that reorganise on binding
  are modelled badly.

The honest summary of a docking run is: *these compounds are worth testing
first*, never *these compounds will bind*.

## Tuning

| Setting | Guidance |
|---|---|
| `--exhaustiveness` | 8 for screening thousands; 16-32 for final poses on a handful. Roughly linear in time. |
| `--workers` | One ligand per worker. Set near the core count. |
| `--max-heavy-atoms` | Default 60. Larger flexible ligands need much higher exhaustiveness to dock meaningfully; they are skipped with a note rather than scored badly. |
| `--scoring vinardo` | Alternative function; sometimes better enrichment. Worth trying if the control passes with both. |

## Chaining

Cull with triage first on large libraries — docking is far more expensive
than descriptors:

```bash
pharma-ops triage big.smi -o clean --alert-free --min-score 55 --top 2000
pharma-ops dock receptor.pdb clean.csv --smiles-col smiles -o dock_out
```

Scoring with a QSAR model (`target2hit`) and docking are independent lines
of evidence. Compounds ranked well by both are the most defensible picks;
disagreement between them is informative and worth showing the user.
