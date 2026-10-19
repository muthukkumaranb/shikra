"""SHIKRA visual shell: one stylesheet plus reusable HTML components.

Every page calls ``apply(page_key)`` right after ``st.set_page_config``. It loads
app/assets/theme.css (with the bundled fonts inlined, so nothing is fetched),
draws the sidebar shell and returns a placeholder for the top status strip,
which the page fills with ``status_strip`` once it knows its day and snapshot.

All HTML goes through ``html()``, which collapses it to one line so Markdown
never turns indented markup into a code block.
"""

from __future__ import annotations

import base64
import html as _html
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
ASSETS = ROOT / "app" / "assets"
CSS_PATH = ASSETS / "theme.css"

TAGLINE = "Predict the snag, place the spare, fly the sortie."
EXPANSION = "Smart Health-Integrated Knowledge for Readiness Advisory"
BIRD = "Named after the shikra (Accipiter badius), a small, sharp-eyed Indian hawk that spots its target early."
FORMATION = "Synthetic wing · 54 aircraft · 3 squadrons"
FOOTER = "SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS"
N_AIRCRAFT = 54

# Status colours shared with app/lib/visuals.py
MC_COLOR, MAINT_COLOR, SUPPLY_COLOR = "#22c55e", "#f59e0b", "#ef4444"
STATUS_COLOR = {"MC": MC_COLOR, "NMCM": MAINT_COLOR, "NMCS": SUPPLY_COLOR}
STATUS_TONE = {"MC": "green", "NMCM": "amber", "NMCS": "red"}
STATUS_TEXT = {"MC": "Mission-capable", "NMCM": "In maintenance", "NMCS": "Waiting for spares"}


def esc(value: object) -> str:
    """HTML-escape any value for interpolation."""
    return _html.escape(str(value), quote=True)


def oneline(markup: str) -> str:
    """Collapse markup to a single line (no indentation, no blank lines)."""
    return " ".join(part.strip() for part in markup.splitlines() if part.strip())


def html(markup: str, target=None) -> None:
    """Render raw HTML through st.markdown (never st.html)."""
    (target or st).markdown(oneline(markup), unsafe_allow_html=True)


@lru_cache(maxsize=1)
def _css() -> str:
    css = CSS_PATH.read_text(encoding="utf-8")

    def _inline(match: re.Match) -> str:
        data = (ASSETS / match.group(1)).read_bytes()
        return f'url("data:font/woff2;base64,{base64.b64encode(data).decode()}")'

    css = re.sub(r'url\("(fonts/[^"]+\.woff2)"\)', _inline, css)
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return oneline(css)


# Generic swept-wing fighter outline, top view (no real type).
_JET_PATH = (
    "M24 3 L27 15 L28 27 L25 39 L23 39 L20 27 L21 15 Z"
)


def jet_svg(color: str, size: int = 26) -> str:
    """Inline generic-fighter icon in one status colour."""
    return (
        f'<svg viewBox="0 0 48 48" width="{size}" height="{size}" fill="none">'
        f'<path d="{_JET_PATH}" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
        f'<path d="M22 17 L5 30 L6 33 L21 28 Z" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.2"/>'
        f'<path d="M26 17 L43 30 L42 33 L27 28 Z" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.2"/>'
        f'<path d="M18 31 L12 41 L15 42 L20 37 Z" fill="{color}"/>'
        f'<path d="M30 31 L36 41 L33 42 L28 37 Z" fill="{color}"/>'
        f'<path d="M23.5 12 L24.5 12 L25 18 L23 18 Z" fill="#dae2fd" fill-opacity="0.8"/>'
        "</svg>"
    )


_LOCK = (
    '<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6">'
    '<rect x="3" y="7" width="10" height="7" rx="1.5"/><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"/></svg>'
)


def _sidebar() -> None:
    html(
        f"""
        <div class="sk-side-brand">
          <div class="sk-mark">
            <div class="sk-logo">{jet_svg("#8ed5ff", 22)}</div>
            <div><div class="sk-name" title="{esc(BIRD)}">SHIKRA</div>
            <div class="sk-sub">{esc(EXPANSION)}</div></div>
          </div>
          <div class="sk-tag">{esc(TAGLINE)}</div>
        </div>
        <div class="sk-side-state">
          <div class="sk-row"><span class="sk-k" style="font-family:var(--font-sans);font-size:10px;font-weight:600;letter-spacing:.08em;">SYSTEM STATE</span></div>
          <div class="sk-row"><span class="sk-k">Mode:</span><span class="sk-v">offline · advisory</span></div>
          <div class="sk-row"><span class="sk-k">Data:</span><span class="sk-v">local files</span></div>
          <div class="sk-row"><span class="sk-k">Network:</span><span class="sk-v">none</span></div>
        </div>
        <div class="sk-side-foot">
          <div class="sk-h">{_LOCK}<span>Air-gapped</span></div>
          <p>Air-gapped · advisory · synthetic wing + NASA C-MAPSS</p>
          <p class="sk-bird">{esc(BIRD)}</p>
        </div>
        """,
        target=st.sidebar,
    )


def apply(page_key: str):
    """Load the theme and sidebar shell; return the status-strip placeholder.

    Call immediately after ``st.set_page_config`` on every page.
    """
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
    _sidebar()
    st.session_state["_sk_page"] = page_key
    return st.empty()


def status_strip(slot, day: int | None = None, mc: int | None = None) -> None:
    """Fill the top strip. Timeline and mission-capable fields are omitted when
    the page has no day or no snapshot; no placeholder numbers are shown."""
    fields = [f'<div class="sk-f"><span class="sk-k">Formation</span><span class="sk-v">{esc(FORMATION)}</span></div>']
    if day is not None:
        fields.append('<div class="sk-sep"></div>')
        fields.append(
            f'<div class="sk-f"><span class="sk-k">Timeline</span><span class="sk-v sk-blue">Sim day {int(day)}</span></div>'
        )
    if mc is not None:
        fields.append('<div class="sk-sep"></div>')
        fields.append(
            '<div class="sk-f"><span class="sk-k">Mission-capable</span>'
            f'<span class="sk-v sk-green"><span class="sk-chip sk-green" style="padding:0;background:none;box-shadow:none;"><span class="sk-led"></span></span> {int(mc)} / {N_AIRCRAFT}</span></div>'
        )
    fields.append(
        '<div class="sk-f sk-push"><span class="sk-k">System</span><span class="sk-v">Offline</span></div>'
    )
    html(f'<div class="sk-strip">{"".join(fields)}</div>', target=slot)


def header(eyebrow: str, title: str, subtitle: str, source: str) -> None:
    """Page header: mono eyebrow chip, question title, subtitle, Source line."""
    html(
        f"""
        <div class="sk-head">
          <span class="sk-eyebrow">{esc(eyebrow)}</span>
          <h1 class="sk-title">{esc(title)}</h1>
          <p class="sk-subtitle">{esc(subtitle)}</p>
          <div class="sk-source">Source: <b>{esc(source)}</b></div>
        </div>
        """
    )


def chip(text: str, tone: str = "", led: bool = False) -> str:
    dot = '<span class="sk-led"></span>' if led else ""
    return f'<span class="sk-chip sk-{tone}">{dot}{esc(text)}</span>'


def bar(pct: float, color: str = "var(--secondary)") -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return f'<div class="sk-bar"><div style="width:{pct:.1f}%;background:{color};"></div></div>'


def segbar(parts: list[tuple[float, str]]) -> str:
    total = sum(v for v, _ in parts) or 1.0
    segs = "".join(
        f'<div style="width:{v / total * 100:.2f}%;background:{c};"></div>' for v, c in parts if v > 0
    )
    return f'<div class="sk-segbar">{segs}</div>'


def kpi(
    label: str,
    value: str,
    sub: str = "",
    unit: str = "",
    tone: str = "",
    badge: str = "",
    extra: str = "",
    alert: bool = False,
) -> str:
    """One KPI tile (caps label, big mono number, small sub-label)."""
    tone_cls = f" sk-t-{tone}" if tone else ""
    unit_html = f'<span class="sk-unit">{esc(unit)}</span>' if unit else ""
    return (
        f'<div class="sk-kpi{" sk-alert" if alert else ""}">'
        f'<div class="sk-kl"><span class="sk-label">{esc(label)}</span>{badge}</div>'
        f'<div><div class="sk-num{tone_cls}">{esc(value)}{unit_html}</div>{extra}</div>'
        f'<div class="sk-sub">{sub}</div></div>'
    )


def kpi_row(tiles: list[str]) -> None:
    html(f'<div class="sk-kpis">{"".join(tiles)}</div>')


def panel_title(title: str, meta: str = "", right: str = "") -> str:
    meta_html = f'<div class="sk-pm">{meta}</div>' if meta else ""
    return (
        f'<div class="sk-ptitle"><div><div class="sk-pt">{esc(title)}</div>{meta_html}</div>'
        f"<div>{right}</div></div>"
    )


def section(title: str, meta: str = "", right: str = "") -> None:
    html(panel_title(title, meta, right))


def footer() -> None:
    html(f'<div class="sk-footer"><span>{esc(FOOTER)}</span><span>Advisory recommendation · officer sign-off required</span></div>')


def count_statuses(statuses: list[str]) -> dict[str, int]:
    return {k: sum(1 for s in statuses if s == k) for k in ("MC", "NMCM", "NMCS")}


def flight_line(tail_statuses: list[dict], sqn_mc: list[int]) -> str:
    """Stitch-style flight line: three squadron rows of 18 aircraft slots."""
    counts = count_statuses([t["status"] for t in tail_statuses])
    legend = (
        '<div class="sk-legend">'
        f'<span><span class="sk-dot" style="background:{MC_COLOR}"></span>Mission-capable: <b class="sk-t-green">{counts["MC"]}</b></span>'
        f'<span><span class="sk-dot" style="background:{MAINT_COLOR}"></span>In maintenance: <b class="sk-t-amber">{counts["NMCM"]}</b></span>'
        f'<span><span class="sk-dot" style="background:{SUPPLY_COLOR}"></span>Waiting for spares: <b class="sk-t-red">{counts["NMCS"]}</b></span>'
        "</div>"
    )
    parts = [
        '<div class="sk-panel">',
        panel_title(
            f"Flight line — {N_AIRCRAFT} aircraft",
            "Status per tail from the simulation on the selected day · synthetic wing",
            legend,
        ),
    ]
    for sqn in range(3):
        tails = tail_statuses[sqn * 18 : (sqn + 1) * 18]
        mc = sqn_mc[sqn]
        pct = mc / 18 * 100
        tone = "green" if pct >= 75 else ("amber" if pct >= 60 else "red")
        parts.append(
            f'<div class="sk-sqn"><div class="sk-sqnh"><div>{chip(f"S{sqn + 1}", "blue")}'
            f'<span class="sk-sn">Squadron {sqn + 1}</span></div>'
            f'{chip(f"{mc}/18 MC ({pct:.1f}%)", tone)}</div><div class="sk-grid18">'
        )
        for t in tails:
            color = STATUS_COLOR.get(t["status"], SUPPLY_COLOR)
            parts.append(
                f'<div class="sk-slot" title="{esc(t["tail"])} · {esc(STATUS_TEXT.get(t["status"], t["status"]))}">'
                f'{jet_svg(color)}<div class="sk-id"><i style="background:{color}"></i>{esc(t["tail"])}</div></div>'
            )
        parts.append("</div></div>")
    parts.append("</div>")
    return "".join(parts)


def diverging(value: float, vmax: float) -> tuple[str, str]:
    """Background and text colour for a diverging cell (blue = gain, red = loss)."""
    if vmax <= 0:
        vmax = 1.0
    a = min(1.0, abs(value) / vmax)
    if value >= 0:
        rgb = (56, 189, 248)
    else:
        rgb = (239, 68, 68)
    opacity = 0.12 + 0.68 * a
    text = "#060e20" if opacity > 0.55 else "#dae2fd"
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity:.2f})", text
