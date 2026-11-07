"""Generate reproducible sample data so the pipeline can be tested offline.

Builds a combinatorial library over several distinct scaffolds with a
*synthetic* structure-activity relationship, so scaffold splitting, QSAR
fitting and applicability-domain logic are all genuinely exercised without
touching the network.

The activities are made up. This is a software fixture, not chemistry data.
"""

from __future__ import annotations

import os
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))

# Distinct cores with an attachment point, loosely kinase-inhibitor shaped
SCAFFOLDS = {
    "quinazoline":   "Nc1ncnc2cc({R})ccc12",
    "pyrimidine":    "Nc1ncc({R})cn1",
    "indazole":      "c1ccc2[nH]nc({R})c2c1",
    "benzimidazole": "c1ccc2[nH]c({R})nc2c1",
    "pyrazolopyr":   "c1cc2cc({R})n[nH]c2cn1",
    "thienopyr":     "Nc1ncnc2sc({R})cc12",
    "naphthalene":   "c1ccc2cc({R})ccc2c1",
    "quinoline":     "c1ccc2nc({R})ccc2c1",
}

SUBSTITUENTS = [
    "C", "CC", "CCC", "C(C)C", "OC", "OCC", "F", "Cl", "Br",
    "C#N", "C(=O)N", "C(=O)OC", "S(=O)(=O)N", "N(C)C", "NC(=O)C",
    "c1ccccc1", "c1ccncc1", "c1ccc(F)cc1", "c1ccc(OC)cc1",
    "N1CCOCC1", "N1CCNCC1", "N1CCCC1", "C1CCCCC1",
    "CCN(C)C", "COC", "CF", "C(F)(F)F", "CO", "CCO",
]


def build(seed: int = 7) -> list[tuple[str, str, float]]:
    rng = np.random.default_rng(seed)
    rows = []

    # Each scaffold gets its own intrinsic contribution
    scaffold_effect = {name: rng.uniform(-1.2, 1.2) for name in SCAFFOLDS}

    for sname, template in SCAFFOLDS.items():
        for sub in SUBSTITUENTS:
            smi = template.replace("{R}", sub)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)

            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)

            # A deliberately learnable but noisy SAR
            pchembl = (
                6.2
                + scaffold_effect[sname]
                + 0.42 * logp
                - 0.011 * tpsa
                + 0.004 * mw
                + 0.25 * hbd
                + rng.normal(0, 0.45)
            )
            pchembl = float(np.clip(pchembl, 4.0, 10.5))
            rows.append((f"{sname}_{sub}", smi, round(pchembl, 2)))

    return rows


def main() -> None:
    rows = build()
    # Hold out two whole scaffolds as an unseen "candidate library"
    held = {"quinoline", "thienopyr"}
    train = [r for r in rows if r[0].split("_")[0] not in held]
    cand = [r for r in rows if r[0].split("_")[0] in held]

    actives = os.path.join(HERE, "sample_actives.csv")
    with open(actives, "w", encoding="utf-8") as fh:
        fh.write("molecule_chembl_id,smiles,pchembl\n")
        for mid, smi, p in train:
            fh.write(f"{mid},{smi},{p}\n")

    lib = os.path.join(HERE, "sample_library.smi")
    with open(lib, "w", encoding="utf-8") as fh:
        fh.write("# candidate library: two scaffolds never seen in training\n")
        for mid, smi, _ in cand:
            fh.write(f"{smi} {mid}\n")

    # A small, chemically real library for the triage tool
    real = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("metformin", "CN(C)C(=N)NC(=N)N"),
        ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
        ("diazepam", "CN1c2ccc(Cl)cc2C(=Nc2ccccc21)c1ccccc1"),
        ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
        ("catechol_pains", "Oc1ccccc1O"),
        ("rhodanine_pains", "O=C1CSC(=S)N1"),
        ("quinone_pains", "O=C1C=CC(=O)C=C1"),
        ("lignoceric_acid", "CCCCCCCCCCCCCCCCCCCCCCCC(=O)O"),
        ("cyclosporin_like", "CC[C@H]1NC(=O)[C@H](C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](C)NC1=O"),
        ("fragment_small", "c1ccc(C(=O)N)cc1"),
        ("nitro_alert", "O=[N+]([O-])c1ccc(CBr)cc1"),
    ]
    tri = os.path.join(HERE, "sample_triage_library.smi")
    with open(tri, "w", encoding="utf-8") as fh:
        for name, smi in real:
            fh.write(f"{smi} {name}\n")

    print(f"actives: {len(train)} -> {actives}")
    print(f"candidates: {len(cand)} -> {lib}")
    print(f"triage library: {len(real)} -> {tri}")


if __name__ == "__main__":
    main()
