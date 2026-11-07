---
description: Pull known actives for a target, fit a scaffold-split QSAR model, score a library
argument-hint: <target name or CHEMBL id> [--library file] [--task classification]
---

Run the target-to-hit pipeline for: `$ARGUMENTS`

Use the `target-to-hit` skill. Steps:

1. If no target is given, ask for one. If the user has local data instead,
   use `--actives-file`.
2. Run `pharma-ops target2hit`. If the network is unavailable, say so and
   offer the `--actives-file` route.
3. **Report the model metrics before any predictions**, and quote the
   `interpretation` field. A negative R² on a scaffold split is normal and
   means the values are uncalibrated while the ranking may still be useful —
   check `spearman`.
4. If a structure was retrieved, say whether it is experimental or an
   AlphaFold model, and give the caveats for whichever it is.
5. When scoring a library, report how many compounds fall outside the
   applicability domain, and treat those scores as extrapolations.
