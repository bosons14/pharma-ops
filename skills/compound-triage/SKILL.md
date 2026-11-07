---
name: compound-triage
description: This skill should be used when the user asks to "triage these compounds", "check drug-likeness", "is this molecule drug-like", "run Lipinski / rule of five", "flag PAINS", "check for structural alerts", "how synthesisable is this", "estimate solubility / logP / TPSA", "filter this library", or runs /compound-triage. Covers physicochemical descriptors, drug-likeness rule sets, assay-interference and reactive-group alerts, synthetic accessibility and ADMET proxies for SMILES, CSV or SDF input.
---

# Compound triage and ADMET-proxy screening

Rank a compound library by cheap, transparent computed properties so the
expensive work goes to the compounds most likely to survive it.

## When to use this

- The user hands over SMILES, a CSV, or an SDF and wants to know which
  compounds are worth pursuing.
- They ask about any single descriptor (MW, cLogP, TPSA, QED, solubility).
- They want PAINS or reactive-group flags.
- Output from `target2hit` or `dock` needs a drug-likeness sanity pass.

## First run: make sure the CLI exists

Installing the plugin ships these instructions and the Python source, but it
does not install the `pharma-ops` command. Before the first call in a
session, check:

```bash
pharma-ops info || pip install -e "${CLAUDE_PLUGIN_ROOT}"
```

`pharma-ops info` also reports RDKit's version, so a success means the
cheminformatics stack is ready.

## Running it

```bash
# A single molecule
pharma-ops triage "CC(=O)Oc1ccccc1C(=O)O"

# A library, keeping only clean, decent-scoring compounds
pharma-ops triage library.smi -o triage --alert-free --min-score 60

# CSV with named columns, top 200 only
pharma-ops triage compounds.csv --smiles-col canonical_smiles --id-col cmpd_id --top 200
```

Writes `<prefix>.csv`, `<prefix>.html` and `<prefix>_summary.json`.

Input formats: `.smi`, `.csv`, `.tsv`, `.sdf`, `.sdf.gz`, or a literal SMILES.
Columns are auto-detected when not named.

## Reading the output

| Column | Meaning |
|---|---|
| `triage_score` | 0-100 composite: QED, synthetic accessibility, alert burden, solubility, rule violations. Weights are arbitrary but documented — retune them per programme. |
| `qed` | Quantitative Estimate of Drug-likeness, 0-1 (Bickerton 2012). |
| `sa_score` | Synthetic accessibility, 1 easy to 10 hard (Ertl 2009). |
| `n_alerts` | Total PAINS + Brenk + NIH substructure matches. |
| `pains_detail` | Which PAINS filter matched, if any. |
| `esol_logs` | Delaney ESOL aqueous solubility estimate, log mol/L. |
| `lipinski_violations` | Rule-of-five violations; ≤1 is the usual pass. |
| `gi_absorption_likely` | Egan absorption region membership. |
| `bbb_permeant_likely` | TPSA/logP CNS-penetration heuristic. |

## What to tell the user

Be straight about what these numbers are. They are computed properties and
published empirical rules, not predictions of safety or efficacy:

- **A PAINS hit is a warning, not a verdict.** PAINS filters flag
  substructures that are *frequent hitters* in certain assay formats. Plenty
  of approved drugs match one. The right response is to check for
  assay interference, not to delete the compound.
- **Lipinski is a heuristic about oral bioavailability**, derived from a
  1997 analysis of compounds in development. It does not apply to
  natural products, macrocycles, PROTACs, injectables or covalent
  inhibitors, and violating it is not disqualifying.
- **The ADMET columns are rules of thumb.** ESOL is a 1-descriptor
  regression; the absorption and BBB flags are TPSA/logP cutoffs. They
  correlate loosely with measured values. Replace them with measured or
  properly modelled data as soon as any exists.
- **An unflagged compound is not a safe compound.** Nothing here models
  toxicity, hERG, CYP inhibition, genotoxicity or reactive metabolites.

If the user seems to be treating the output as a go/no-go on a real
compound, say plainly that this is a prioritisation aid and that a
medicinal chemist and real assays decide.

## Chaining

`triage` is the natural filter before and after the other two tools:

```bash
pharma-ops triage big_library.smi -o filtered --alert-free --min-score 55
# then dock only what survived
python -c "import pandas as pd; d=pd.read_csv('filtered.csv'); \
  d[['smiles','mol_id']].to_csv('to_dock.smi',sep=' ',index=False,header=False)"
pharma-ops dock receptor.pdb to_dock.smi -o dock_out
```

`pharma-ops dock` already joins triage columns onto its results, so a
separate pass is only needed when culling *before* docking.
