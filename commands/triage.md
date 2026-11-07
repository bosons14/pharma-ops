---
description: Screen compounds for drug-likeness, structural alerts and ADMET proxies
argument-hint: <SMILES | library file> [--alert-free] [--min-score N] [--top N]
---

Run compound triage on: `$ARGUMENTS`

Use the `compound-triage` skill. Steps:

1. If `$ARGUMENTS` is empty, ask what to screen.
2. Run `pharma-ops triage` with the given input and flags.
3. Report the summary statistics and the top compounds.
4. Call out any PAINS or reactive-group hits by name, and explain that an
   alert is a warning to investigate rather than a verdict.
5. State plainly that these are computed properties and published rules,
   not predictions of safety or efficacy.
