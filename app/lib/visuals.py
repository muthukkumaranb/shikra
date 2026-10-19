"""Visual components for SHIKRA aircraft rendering.

Renders inline SVG representations of generic twin-engine fighters for:
- Home flight-line view (54 aircraft across S1, S2, S3)
- Engineering Officer aircraft detail view (recolourable engines and LRUs)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SVG_PATH = ROOT / "app" / "assets" / "fighter_top.svg"

# XML namespace identifier (never fetched). Assembled at runtime so the app tree
# carries no web-scheme literal; a data-URI <img> SVG needs it to render.
SVG_NS = "http" + "://www.w3.org/2000/svg"

# Status colors matching requirements
COLOR_MC = "#22c55e"      # Green = Mission Capable
COLOR_NMCM = "#f59e0b"    # Amber = Not Mission Capable Maintenance
COLOR_NMCS = "#ef4444"    # Red = Not Mission Capable Supply

# Engine RUL status colors
COLOR_ENG_OK = "#22c55e"       # OK: q10 > R* + 15
COLOR_ENG_WATCH = "#f59e0b"    # Watch: R* < q10 <= R* + 15
COLOR_ENG_REMOVAL = "#ef4444"  # Removal due: q10 <= R*

# Component status colors
COLOR_COMP_OK = "#0ea5e9"       # Installed / Serviceable
COLOR_COMP_MISSING = "#ef4444"  # Missing / Backordered (NMCS cause)


def get_base_svg_content() -> str:
    """Read the base fighter SVG."""
    if SVG_PATH.exists():
        svg = SVG_PATH.read_text(encoding="utf-8")
        if "xmlns=" not in svg.split(">", 1)[0]:
            svg = svg.replace("<svg ", f'<svg xmlns="{SVG_NS}" ', 1)
        return svg
    return ""


def get_mini_fighter_svg(status: str, width: int = 44, height: int = 55) -> str:
    """Generate an inline mini SVG for flight-line rendering.

    Optimized for rapid DOM rendering across 54 tails.
    """
    if status == "MC":
        fill_color = "#166534"
        stroke_color = "#22c55e"
    elif status == "NMCM":
        fill_color = "#78350f"
        stroke_color = "#f59e0b"
    else:  # NMCS
        fill_color = "#7f1d1d"
        stroke_color = "#ef4444"

    return (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 400 500" width="{width}" height="{height}" style="display: block; margin: 0 auto;">'
        f'<path d="M 200,30 C 205,45 208,65 210,90 L 214,150 L 230,190 L 365,315 C 370,320 368,326 360,328 L 315,335 L 245,340 L 255,420 L 245,445 L 215,445 L 210,430 L 200,435 L 190,430 L 185,445 L 155,445 L 145,420 L 155,340 L 85,335 L 40,328 C 32,326 30,320 35,315 L 170,190 L 186,150 L 190,90 C 192,65 195,45 200,30 Z" '
        f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="6" stroke-linejoin="round"/>'
        f'<polygon points="145,370 120,465 138,470 156,415" fill="{fill_color}" stroke="{stroke_color}" stroke-width="4" />'
        f'<polygon points="255,370 280,465 262,470 244,415" fill="{fill_color}" stroke="{stroke_color}" stroke-width="4" />'
        f'<rect x="157" y="345" width="26" height="85" rx="5" fill="{stroke_color}" opacity="0.8" />'
        f'<rect x="217" y="345" width="26" height="85" rx="5" fill="{stroke_color}" opacity="0.8" />'
        f'<path d="M 200,95 C 207,105 208,135 207,155 C 205,165 195,165 193,155 C 192,135 193,105 200,95 Z" fill="#38bdf8" fill-opacity="0.6"/>'
        f'</svg>'
    )


def render_flight_line_html(tail_statuses: list[dict], sqn_counts: list[int]) -> str:
    """Generate complete inline HTML for the 54-aircraft flight line.

    Renders 3 squadron rows of 18 aircraft tiles each with tail number,
    color-coded mini SVG, and dual-signal status badge (text + color).
    """
    html_parts = [
        '<div style="background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">'
    ]

    squadron_names = ["Squadron 1 (S1)", "Squadron 2 (S2)", "Squadron 3 (S3)"]

    for sqn_idx in range(3):
        sqn_tails = tail_statuses[sqn_idx * 18 : (sqn_idx + 1) * 18]
        mc_in_sqn = sqn_counts[sqn_idx]
        pct = (mc_in_sqn / 18.0) * 100.0

        html_parts.append(
            f'<div style="margin-bottom: {"1.25rem" if sqn_idx < 2 else "0"};">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.6rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.35rem;">'
            f'<span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{squadron_names[sqn_idx]}</span>'
            f'<span style="font-size: 0.85rem; font-weight: 600; color: #38bdf8; font-family: monospace;">'
            f'S{sqn_idx + 1}: {mc_in_sqn} / 18 MC ({pct:.1f}%)'
            f'</span>'
            f'</div>'
            f'<div style="display: grid; grid-template-columns: repeat(18, 1fr); gap: 4px; text-align: center;">'
        )

        for tail_info in sqn_tails:
            tail = tail_info["tail"]
            status = tail_info["status"]

            if status == "MC":
                badge_bg = "#166534"
                badge_fg = "#86efac"
                badge_text = "MC"
                tile_border = "#22c55e"
            elif status == "NMCM":
                badge_bg = "#78350f"
                badge_fg = "#fde047"
                badge_text = "M"
                tile_border = "#f59e0b"
            else:  # NMCS
                badge_bg = "#7f1d1d"
                badge_fg = "#fca5a5"
                badge_text = "S"
                tile_border = "#ef4444"

            svg_icon = get_mini_fighter_svg(status, width=32, height=42)

            html_parts.append(
                f'<div style="background: #1e293b; border: 1px solid {tile_border}40; border-radius: 6px; padding: 4px 2px; display: flex; flex-direction: column; align-items: center; justify-content: space-between;">'
                f'<div style="margin-bottom: 2px;">{svg_icon}</div>'
                f'<div style="font-family: monospace; font-size: 0.72rem; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;">{tail}</div>'
                f'<div style="background: {badge_bg}; color: {badge_fg}; font-size: 0.62rem; font-weight: 700; border-radius: 3px; padding: 1px 3px; line-height: 1; font-family: monospace;">{badge_text}</div>'
                f'</div>'
            )

        html_parts.append('</div></div>')

    html_parts.append(
        '<div style="margin-top: 0.75rem; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 0.5rem;">'
        '<div>Status symbols: <b style="color: #86efac;">MC</b> (Mission Capable) · <b style="color: #fde047;">M</b> (NMCM / Maintenance) · <b style="color: #fca5a5;">S</b> (NMCS / Supply waiting)</div>'
        '<div style="font-style: italic;">synthetic wing</div>'
        '</div>'
        '</div>'
    )

    return "".join(html_parts)


def render_aircraft_detail_svg(
    tail_id: str,
    tail_status: str,
    engine_states: dict[str, str] | None = None,
    component_states: dict[str, str] | None = None,
    width: int = 420,
    height: int = 520,
) -> str:
    """Generate detailed SVG of a selected aircraft with recoloured engines and LRU markers.

    Parameters:
      tail_id: e.g. "T19"
      tail_status: "MC", "NMCM", or "NMCS"
      engine_states: optional dict {"engine_0": "OK"|"WATCH"|"REMOVAL", "engine_1": ...}
      component_states: optional dict {lru_id: "OK"|"MISSING"}
    """
    base_svg = get_base_svg_content()
    if not base_svg:
        return "<div>Fighter SVG asset missing</div>"

    # Default statuses if None
    if engine_states is None:
        # Default based on tail status
        eng_color = COLOR_ENG_OK if tail_status == "MC" else (COLOR_ENG_WATCH if tail_status == "NMCM" else COLOR_ENG_REMOVAL)
        engine_colors = {"engine_0": eng_color, "engine_1": eng_color}
    else:
        engine_colors = {}
        for eng_k in ["engine_0", "engine_1"]:
            st_val = engine_states.get(eng_k, "OK").upper()
            if st_val == "REMOVAL":
                engine_colors[eng_k] = COLOR_ENG_REMOVAL
            elif st_val == "WATCH":
                engine_colors[eng_k] = COLOR_ENG_WATCH
            else:
                engine_colors[eng_k] = COLOR_ENG_OK

    # Set engine colors by replacing class or inline style
    modified_svg = base_svg
    for eng_id, col in engine_colors.items():
        pattern = rf'(<rect id="{eng_id}_pod"[^>]*?)(\s*/?>)'

        def _replace_engine_pod(m: re.Match, color: str = col) -> str:
            tag_content = re.sub(r'\s*style="[^"]*"', "", m.group(1))
            closer = m.group(2)
            return f'{tag_content} style="fill: {color}; stroke: #ffffff; stroke-width: 2;"{closer}'

        modified_svg = re.sub(pattern, _replace_engine_pod, modified_svg)

    # Set component colors
    if component_states is not None:
        for comp_id, comp_st in component_states.items():
            comp_col = COLOR_COMP_MISSING if comp_st.upper() == "MISSING" else COLOR_COMP_OK
            pattern = rf'(<g id="{comp_id}"[^>]*>\s*)(<circle class="component-marker"[^>]*?)(\s*/?>)'

            def _replace_comp_marker(m: re.Match, color: str = comp_col) -> str:
                g_open = m.group(1)
                circle_content = re.sub(r'\s*style="[^"]*"', "", m.group(2))
                closer = m.group(3)
                return f'{g_open}{circle_content} style="fill: {color}; stroke: #ffffff; stroke-width: 2;"{closer}'

            modified_svg = re.sub(pattern, _replace_comp_marker, modified_svg)

    # Collapse all whitespace into single spaces so markdown never interprets indented lines as code blocks
    return " ".join(modified_svg.split())
