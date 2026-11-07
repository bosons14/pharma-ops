"""Public data-source clients: ChEMBL, UniProt, RCSB PDB, AlphaFold DB.

All four are free, public, and documented. Each client is deliberately thin
and returns plain dicts/DataFrames so the modelling code stays testable
without a network connection.

Rate limiting: a polite default delay between paged requests. ChEMBL asks
clients to identify themselves, so we send a descriptive User-Agent. If you
run large pulls, consider downloading the ChEMBL SQLite dump instead of
hammering the REST API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd
import requests

log = logging.getLogger("pharma_ops.sources")

USER_AGENT = "pharma-ops/0.1 (open-source cheminformatics toolkit)"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILES = "https://files.rcsb.org/download"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api/prediction"


class SourceError(RuntimeError):
    """Raised when a public data source cannot be reached or returns garbage."""


def _session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    s.request_timeout = timeout  # type: ignore[attr-defined]
    return s


def _get(session: requests.Session, url: str, params: dict | None = None, timeout: int = 30) -> Any:
    try:
        r = session.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise SourceError(
            f"could not reach {url}: {exc}. "
            "If you are behind a restricted network, use --actives-file to supply "
            "a local CSV of known actives instead."
        ) from exc
    if r.status_code == 404:
        raise SourceError(f"{url} returned 404 (not found)")
    if not r.ok:
        raise SourceError(f"{url} returned HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as exc:
        raise SourceError(f"{url} did not return JSON") from exc


# ----------------------------------------------------------------------------
# ChEMBL
# ----------------------------------------------------------------------------


@dataclass
class Target:
    chembl_id: str
    pref_name: str
    organism: str | None = None
    target_type: str | None = None
    uniprot: str | None = None

    def __str__(self) -> str:
        return f"{self.chembl_id} | {self.pref_name} | {self.organism or '?'}"


class ChEMBL:
    """Minimal ChEMBL REST client for the target -> bioactivity path."""

    def __init__(self, session: requests.Session | None = None, delay: float = 0.2):
        self.s = session or _session()
        self.delay = delay

    # -- targets ----------------------------------------------------------

    def search_targets(self, query: str, organism: str | None = "Homo sapiens", limit: int = 10) -> list[Target]:
        data = _get(self.s, f"{CHEMBL_BASE}/target/search.json", {"q": query, "limit": limit * 3})
        out: list[Target] = []
        for t in data.get("targets", []):
            org = t.get("organism")
            if organism and org and organism.lower() not in org.lower():
                continue
            uni = None
            for comp in t.get("target_components") or []:
                if comp.get("accession"):
                    uni = comp["accession"]
                    break
            out.append(
                Target(
                    chembl_id=t.get("target_chembl_id", ""),
                    pref_name=t.get("pref_name", ""),
                    organism=org,
                    target_type=t.get("target_type"),
                    uniprot=uni,
                )
            )
            if len(out) >= limit:
                break
        if not out:
            raise SourceError(f"no ChEMBL target matched {query!r} (organism={organism})")
        return out

    def resolve_target(self, query: str, organism: str | None = "Homo sapiens") -> Target:
        """Pick the best single target for a free-text name or a CHEMBL id."""
        if query.upper().startswith("CHEMBL"):
            data = _get(self.s, f"{CHEMBL_BASE}/target/{query.upper()}.json")
            uni = None
            for comp in data.get("target_components") or []:
                if comp.get("accession"):
                    uni = comp["accession"]
                    break
            return Target(
                chembl_id=data.get("target_chembl_id", query.upper()),
                pref_name=data.get("pref_name", ""),
                organism=data.get("organism"),
                target_type=data.get("target_type"),
                uniprot=uni,
            )

        candidates = self.search_targets(query, organism=organism, limit=10)
        # Prefer SINGLE PROTEIN targets -- cleanest bioactivity data
        for t in candidates:
            if (t.target_type or "").upper() == "SINGLE PROTEIN":
                return t
        return candidates[0]

    # -- bioactivities ----------------------------------------------------

    def activities(
        self,
        target_chembl_id: str,
        standard_types: tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50"),
        max_records: int = 5000,
        page_size: int = 1000,
    ) -> pd.DataFrame:
        """Pull dose-response bioactivities for a target.

        Only records with a usable ``pchembl_value`` and an exact ``=``
        relation are kept -- censored values ('>', '<') are unusable as
        regression targets and are a common source of silently bad models.
        """
        rows: list[dict] = []
        for stype in standard_types:
            offset = 0
            while len(rows) < max_records:
                params = {
                    "target_chembl_id": target_chembl_id,
                    "standard_type": stype,
                    "limit": page_size,
                    "offset": offset,
                }
                data = _get(self.s, f"{CHEMBL_BASE}/activity.json", params)
                acts = data.get("activities", [])
                if not acts:
                    break
                for a in acts:
                    smi = a.get("canonical_smiles")
                    pchembl = a.get("pchembl_value")
                    if not smi or pchembl in (None, ""):
                        continue
                    if a.get("standard_relation") not in ("=", None):
                        continue
                    rows.append(
                        {
                            "molecule_chembl_id": a.get("molecule_chembl_id"),
                            "smiles": smi,
                            "standard_type": a.get("standard_type"),
                            "standard_value": a.get("standard_value"),
                            "standard_units": a.get("standard_units"),
                            "pchembl_value": float(pchembl),
                            "assay_chembl_id": a.get("assay_chembl_id"),
                            "assay_type": a.get("assay_type"),
                            "target_chembl_id": a.get("target_chembl_id"),
                            "document_year": a.get("document_year"),
                        }
                    )
                page_meta = data.get("page_meta", {})
                if not page_meta.get("next"):
                    break
                offset += page_size
                time.sleep(self.delay)

        df = pd.DataFrame(rows)
        if df.empty:
            raise SourceError(
                f"no usable bioactivities for {target_chembl_id}. "
                "Try widening --activity-types, or check the target has dose-response data."
            )
        return df


def aggregate_activities(df: pd.DataFrame, prefer: tuple[str, ...] = ("Ki", "Kd", "IC50", "EC50")) -> pd.DataFrame:
    """Collapse multiple measurements per compound into one value.

    Uses the median pChEMBL per compound within the most-preferred assay type
    available for that compound, and records the spread so you can see which
    compounds have inconsistent literature values.
    """
    if df.empty:
        return df
    rank = {t: i for i, t in enumerate(prefer)}
    df = df.copy()
    df["_rank"] = df["standard_type"].map(lambda t: rank.get(t, len(prefer)))
    best = df.sort_values("_rank").groupby("molecule_chembl_id")["_rank"].transform("min")
    df = df[df["_rank"] == best]

    agg = (
        df.groupby(["molecule_chembl_id", "smiles"], as_index=False)
        .agg(
            pchembl=("pchembl_value", "median"),
            pchembl_std=("pchembl_value", "std"),
            n_measurements=("pchembl_value", "size"),
            standard_type=("standard_type", "first"),
        )
        .fillna({"pchembl_std": 0.0})
    )
    return agg.sort_values("pchembl", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# UniProt / structures
# ----------------------------------------------------------------------------


def uniprot_for_gene(gene: str, organism_id: int = 9606, session: requests.Session | None = None) -> str | None:
    s = session or _session()
    data = _get(
        s,
        f"{UNIPROT_BASE}/search",
        {"query": f"gene:{gene} AND organism_id:{organism_id} AND reviewed:true", "size": 1, "format": "json"},
    )
    results = data.get("results") or []
    return results[0].get("primaryAccession") if results else None


def pdb_structures_for_uniprot(accession: str, session: requests.Session | None = None, limit: int = 10) -> list[dict]:
    """Find experimental structures for a UniProt accession via RCSB search."""
    s = session or _session()
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": accession,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit},
                            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]},
    }
    try:
        r = s.post(RCSB_SEARCH, json=query, timeout=30)
    except requests.RequestException as exc:
        raise SourceError(f"RCSB search failed: {exc}") from exc
    if r.status_code == 204:
        return []
    if not r.ok:
        raise SourceError(f"RCSB search returned HTTP {r.status_code}")
    return [{"pdb_id": hit["identifier"], "score": hit.get("score")} for hit in r.json().get("result_set", [])]


def fetch_pdb(pdb_id: str, out_path: str, session: requests.Session | None = None) -> str:
    s = session or _session()
    url = f"{RCSB_FILES}/{pdb_id.upper()}.pdb"
    try:
        r = s.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise SourceError(f"could not download {pdb_id} from RCSB: {exc}") from exc
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(r.text)
    return out_path


def fetch_alphafold(accession: str, out_path: str, session: requests.Session | None = None) -> str:
    """Download the AlphaFold DB model for a UniProt accession.

    Note the per-residue confidence (pLDDT) is in the B-factor column. Low
    pLDDT regions -- loops especially -- are not reliable for docking, and an
    AlphaFold model has no bound ligand and often an occluded or collapsed
    pocket. Use an experimental holo structure when one exists.
    """
    s = session or _session()
    meta = _get(s, f"{ALPHAFOLD_BASE}/{accession}")
    if not meta:
        raise SourceError(f"no AlphaFold model for {accession}")
    pdb_url = meta[0].get("pdbUrl")
    if not pdb_url:
        raise SourceError(f"AlphaFold entry for {accession} has no PDB file")
    try:
        r = s.get(pdb_url, timeout=120)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise SourceError(f"could not download AlphaFold model: {exc}") from exc
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(r.text)
    return out_path
