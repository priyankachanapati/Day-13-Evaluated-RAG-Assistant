"""Phase 7 charts (SVG) from results/phase7/retrieval.json - no retrieval or LLM call.

    .venv/bin/python scripts/phase7_charts.py   ->  results/phase7/charts/{coverage-at-3,mrr-at-3,retrieval-latency}.svg

One measure per chart (no dual axis), single series = one hue (validated reference palette slot 1, light #257de0 /
dark #338ced), zero-based horizontal bars, direct value labels, native hover tooltip per bar (<title>), dark mode via
prefers-color-scheme. The data tables are in results/phase7/phase7-report.md.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "phase7" / "charts"
BASE = "v1_paragraph_800_150+dense"
STYLE = """<style>
  .surface{fill:#ffffff} .t1{fill:#191919} .t2{fill:#666666} .grid{stroke:#e6e6e6} .axis{stroke:#bdbdbd}
  .bar{fill:#257de0} .bar:hover{opacity:.8}
  @media (prefers-color-scheme: dark){
    .surface{fill:#191919} .t1{fill:#efefef} .t2{fill:#9c9c9c} .grid{stroke:#2e2e2e} .axis{stroke:#555555} .bar{fill:#338ced}
  }
  text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}
</style>"""


def bar_chart(
    title: str, subtitle: str, rows: list[tuple[str, float]], vmax: float, ticks: list[float], fmt
) -> str:
    left, right, top, row_h, bar_h = 300, 70, 70, 24, 14
    plot_w = 420
    width, height = left + plot_w + right, top + row_h * len(rows) + 40
    x = lambda v: left + plot_w * v / vmax  # noqa: E731
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        STYLE,
        f'<rect class="surface" width="{width}" height="{height}"/>',
        f'<text class="t1" x="16" y="26" font-size="15" font-weight="600">{escape(title)}</text>',
        f'<text class="t2" x="16" y="46" font-size="12">{escape(subtitle)}</text>',
    ]
    bottom = top + row_h * len(rows)
    for t in ticks:
        s.append(
            f'<line class="grid" x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{bottom}" stroke-width="1"/>'
        )
        s.append(
            f'<text class="t2" x="{x(t):.1f}" y="{bottom + 16}" font-size="11" text-anchor="middle">{fmt(t)}</text>'
        )
    s.append(f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{bottom}" stroke-width="1"/>')
    for i, (label, v) in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        name = label + ("  (v1 baseline)" if label == BASE else "")
        weight = ' font-weight="600"' if label == BASE else ""
        s.append(
            f'<text class="t1" x="{left - 8}" y="{y + bar_h - 3:.1f}" font-size="11.5" text-anchor="end"{weight}>'
            f"{escape(name)}</text>"
        )
        w = max(0.0, x(v) - left)
        s.append(
            f"<g><title>{escape(label)}: {fmt(v)}</title>"
            f'<rect class="bar" x="{left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" rx="2"/>'
            f'<rect x="{left}" y="{top + i * row_h:.1f}" width="{plot_w}" height="{row_h}" fill="transparent"/></g>'
        )
        s.append(
            f'<text class="t1" x="{left + w + 6:.1f}" y="{y + bar_h - 3:.1f}" font-size="11">{fmt(v)}</text>'
        )
    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    data = json.loads((PROJECT_ROOT / "results" / "phase7" / "retrieval.json").read_text())["configs"]
    OUT.mkdir(parents=True, exist_ok=True)
    src = "Dataset v1.1, 25 answerable queries (macro); Phase 7 Part A; no LLM"
    charts = {
        "coverage-at-3.svg": (
            "Fact coverage@3 by configuration",
            src,
            [(n, c["aggregate"]["3"]["coverage_macro"]) for n, c in data.items()],
            1.0,
            [0, 0.25, 0.5, 0.75, 1.0],
            lambda v: f"{v:.2f}",
        ),
        "mrr-at-3.svg": (
            "MRR@3 by configuration",
            src,
            [(n, c["aggregate"]["3"]["mrr"]) for n, c in data.items()],
            1.0,
            [0, 0.25, 0.5, 0.75, 1.0],
            lambda v: f"{v:.2f}",
        ),
        "retrieval-latency.svg": (
            "Median retrieval latency per query (ms)",
            "30 queries, single pass, Apple M3 CPU; excludes LLM generation (≈1 s)",
            [(n, c["median_ms"]["retrieval_total"]) for n, c in data.items()],
            100.0,
            [0, 25, 50, 75, 100],
            lambda v: f"{v:.0f}",
        ),
    }
    for name, args in charts.items():
        (OUT / name).write_text(bar_chart(*args), encoding="utf-8")
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
