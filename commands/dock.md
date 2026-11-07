---
description: Prepare a receptor and run an AutoDock Vina virtual screen
argument-hint: <receptor.pdb> <library> [--site-residues N N] [--exhaustiveness N]
---

Run structure-based virtual screening on: `$ARGUMENTS`

Use the `virtual-screening` skill. Steps:

1. Run `pharma-ops info` first if Vina availability is unknown. If it is
   missing, give the install options and stop.
2. Run `pharma-ops dock`. Let the box come from the co-crystallised ligand
   when the PDB has one; otherwise ask for site residues before falling
   back to geometric pocket detection.
3. **Report the redocking control first.** If it failed, or no control was
   possible, say that the scores are unvalidated before showing any
   ranking.
4. Show `affinity` and `ligand_efficiency` together. If the top hits are
   simply the largest molecules, say so.
5. State that the scores rank compounds for testing and do not predict
   potency, and that differences under ~1 kcal/mol are noise.
6. Point to the pose files and recommend inspecting them.
