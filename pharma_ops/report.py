"""HTML report rendering, with no external template files needed."""

from __future__ import annotations

import datetime as _dt
import html
import json

import pandas as pd

_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--accent:#2d5f8a;
--good:#2f6f4f;--warn:#8a5a2d;--bad:#8a2d2d;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#eceae4;--mut:#9a9a92;
--line:#2e2e2a;--accent:#7fb0d8;--good:#7fc3a0;--warn:#d8ab7f;--bad:#d88f8f;--card:#1e1e1b}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .25rem;letter-spacing:-.01em}
h2{font-size:1.05rem;margin:2rem 0 .75rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:.875rem;margin:0 0 1.5rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem;margin:1rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem .9rem}
.card .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.card .v{font-size:1.4rem;font-weight:600;margin-top:.15rem;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:.82rem;background:var(--card);
border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:.45rem .6rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:color-mix(in srgb,var(--card) 92%,var(--fg));font-weight:600;font-size:.72rem;
text-transform:uppercase;letter-spacing:.04em;color:var(--mut);position:sticky;top:0}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:color-mix(in srgb,var(--card) 94%,var(--accent))}
.scroll{overflow-x:auto;border-radius:8px}
.tag{display:inline-block;padding:.1rem .4rem;border-radius:4px;font-size:.72rem;font-weight:600}
.t-good{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.t-warn{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.t-bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.note{background:color-mix(in srgb,var(--warn) 10%,var(--card));border:1px solid
color-mix(in srgb,var(--warn) 35%,var(--line));border-left:3px solid var(--warn);
border-radius:6px;padding:.75rem .9rem;margin:1rem 0;font-size:.85rem}
.note b{color:var(--warn)}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem;
overflow-x:auto;font-size:.76rem;line-height:1.5;margin:0}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);
color:var(--mut);font-size:.78rem}
"""

_DISCLAIMER = (
    "<b>What this is.</b> Computed physicochemical properties, published empirical rules, "
    "and statistical models fitted to public data. These outputs rank and cull compounds "
    "for follow-up. They are not predictions of potency, safety, or efficacy in any "
    "organism, and no compound should advance on the strength of these numbers alone. "
    "Flagged compounds are compounds to look at; unflagged compounds are not safe compounds."
)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "&mdash;"
    if isinstance(v, bool):
        cls = "t-good" if v else "t-bad"
        return f'<span class="tag {cls}">{"yes" if v else "no"}</span>'
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 300) -> str:
    if df.empty:
        return "<p class='sub'>No rows.</p>"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    sub = df[cols].head(max_rows)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r[c])}</td>" for c in cols) + "</tr>"
        for _, r in sub.iterrows()
    )
    more = (f"<p class='sub'>Showing {max_rows} of {len(df)} rows; "
            "the full set is in the CSV.</p>") if len(df) > max_rows else ""
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _cards(summary: dict, keys: list[tuple[str, str]]) -> str:
    out = []
    for key, label in keys:
        if key in summary and summary[key] is not None:
            out.append(f"<div class='card'><div class='k'>{html.escape(label)}</div>"
                       f"<div class='v'>{_fmt(summary[key])}</div></div>")
    return f"<div class='cards'>{''.join(out)}</div>" if out else ""


def render(
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    summary: dict,
    cards: list[tuple[str, str]],
    table_cols: list[str] | None = None,
    extra_json: dict | None = None,
    extra_notes: list[str] | None = None,
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    notes = "".join(f"<div class='note'>{n}</div>" for n in (extra_notes or []))
    extra = ""
    if extra_json:
        extra = ("<h2>Run detail</h2><pre>"
                 + html.escape(json.dumps(extra_json, indent=2, default=str))
                 + "</pre>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)} &middot; generated {now} by pharma-ops</p>
<div class="note">{_DISCLAIMER}</div>
{notes}
{_cards(summary, cards)}
<h2>Results</h2>
{_table(df, table_cols)}
{extra}
<footer>pharma-ops &middot; RDKit, scikit-learn, AutoDock Vina &middot;
data from ChEMBL, RCSB PDB and AlphaFold DB where applicable.</footer>
</div></body></html>"""
