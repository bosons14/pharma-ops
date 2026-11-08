"""pharma-ops: an open-source toolkit for early-stage drug discovery.

Three stages, usable alone or chained:

  1. ``triage``      -- physicochemical / drug-likeness / ADMET-proxy screening
  2. ``target2hit``  -- pull known actives for a target, build a QSAR model,
                        score new compounds
  3. ``dock``        -- structure-based virtual screening with AutoDock Vina

Everything here is built on public data (ChEMBL, RCSB PDB, AlphaFold DB) and
open-source software (RDKit, scikit-learn, AutoDock Vina). The outputs are
*prioritisation aids* for a trained medicinal chemist, not predictions of
safety or efficacy in humans. See README for the stated limitations.
"""

__version__ = "0.1.3"

__all__ = ["__version__"]
