"""Self-contained, print-friendly HTML file for the Commander's daily brief.

Wraps the brief body the Commander page already builds; no number is
recomputed here. Plain CSS only: no external fonts, no links.
"""

from __future__ import annotations

from app.lib import theme

TOP_LINE = "SHIKRA · advisory decision support · synthetic wing + NASA C-MAPSS · not IAF data"
BOTTOM_LINE = "Generated offline by SHIKRA. Every action requires officer sign-off."

_CSS = (
    "body{background:#ffffff;color:#000000;font-family:Arial,Helvetica,sans-serif;"
    "font-size:13px;line-height:1.45;margin:24px auto;max-width:960px;padding:0 16px;}"
    "#shikra_brief,#shikra_brief *{color:#000000 !important;background:transparent !important;"
    "box-shadow:none !important;}"
    "#shikra_brief{border-top:2px solid #000000 !important;padding:12px 0 !important;}"
    "table{width:100%;border-collapse:collapse;margin:4px 0;}"
    "th,td{border:1px solid #999999;padding:4px 6px;text-align:left;}"
    "th{font-weight:bold;}code{font-family:Consolas,Courier,monospace;}"
    ".sk-eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.08em;}"
    ".sk-banner{font-size:11px;border-bottom:1px solid #999999;padding-bottom:6px;margin-bottom:12px;}"
    ".sk-foot{font-size:11px;border-top:1px solid #999999;padding-top:6px;margin-top:16px;}"
    "@media print{body{margin:0;}}"
)


def brief_document(body_html: str, day: int, utilisation_line: str = "") -> str:
    """Return a complete HTML document around the brief body."""
    util = f'<p class="sk-util"><b>Utilisation:</b> {theme.esc(utilisation_line)}</p>' if utilisation_line else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>SHIKRA Commander brief – Day {int(day)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f'<div class="sk-banner">{theme.esc(TOP_LINE)}</div>'
        f"{theme.oneline(body_html)}{util}"
        f'<div class="sk-foot">{theme.esc(BOTTOM_LINE)}</div>'
        "</body></html>"
    )
