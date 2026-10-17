#!/usr/bin/env python3
"""SHIKRA Live Terminal Visualizer.

A high-tech terminal dashboard that monitors background simulation, training,
and stress-testing processes running in the terminal.

Features:
- Live CPU, Memory, and Thread telemetry with Unicode sparkline graphs
- Auto-detection of running Python workloads (multi_condition, supply_shock, sim.runner, etc.)
- Multi-condition RUL pipeline stage tracker (FD002/FD004 clustering & LightGBM training)
- Supply shock & fleet logistics telemetry tracker
- Live output artifact watcher (splits_multi.json, rul_cmapss_multi.csv, shock_results.json)
- Zero external dependencies (uses standard library and ANSI escape sequences)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Terminal Colors & ANSI Sequences
ESC = "\033["
RESET = f"{ESC}0m"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"
ITALIC = f"{ESC}3m"
UNDERLINE = f"{ESC}4m"

# Foreground Colors
FG_BLACK = f"{ESC}30m"
FG_RED = f"{ESC}31m"
FG_GREEN = f"{ESC}32m"
FG_YELLOW = f"{ESC}33m"
FG_BLUE = f"{ESC}34m"
FG_MAGENTA = f"{ESC}35m"
FG_CYAN = f"{ESC}36m"
FG_WHITE = f"{ESC}37m"

# Bright Foreground Colors
FG_BRIGHT_RED = f"{ESC}91m"
FG_BRIGHT_GREEN = f"{ESC}92m"
FG_BRIGHT_YELLOW = f"{ESC}93m"
FG_BRIGHT_BLUE = f"{ESC}94m"
FG_BRIGHT_MAGENTA = f"{ESC}95m"
FG_BRIGHT_CYAN = f"{ESC}96m"
FG_BRIGHT_WHITE = f"{ESC}97m"
FG_GRAY = f"{ESC}90m"

# Background Colors
BG_DARK_GRAY = f"{ESC}100m"
BG_BLUE = f"{ESC}44m"
BG_CYAN = f"{ESC}46m"
BG_DARK = f"{ESC}48;5;235m"

# Unicode Sparkline Characters
SPARKS = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def render_sparkline(values: list[float], max_val: float = 100.0) -> str:
    """Render a smooth Unicode sparkline from a list of numbers."""
    if not values:
        return ""
    if max_val <= 0:
        max_val = max(values) if max(values) > 0 else 1.0

    res = []
    for v in values:
        norm = max(0.0, min(1.0, v / max_val))
        idx = min(len(SPARKS) - 1, int(norm * len(SPARKS)))
        # Color gradient based on intensity
        if norm > 0.8:
            color = FG_BRIGHT_RED
        elif norm > 0.5:
            color = FG_BRIGHT_YELLOW
        elif norm > 0.2:
            color = FG_BRIGHT_CYAN
        else:
            color = FG_CYAN
        res.append(f"{color}{SPARKS[idx]}{RESET}")
    return "".join(res)


def render_progress_bar(pct: float, width: int = 28, color: str = FG_BRIGHT_CYAN) -> str:
    """Render a high-precision unicode progress bar."""
    pct = max(0.0, min(100.0, pct))
    fill_len = (pct / 100.0) * width
    full_blocks = int(fill_len)
    remainder = fill_len - full_blocks

    sub_blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    sub_char = sub_blocks[int(remainder * 8)] if full_blocks < width else ""

    bar = "█" * full_blocks + sub_char
    empty = "░" * (width - len(bar))
    return f"{color}{bar}{FG_GRAY}{empty}{RESET} {BOLD}{pct:5.1f}%{RESET}"


class ProcessMonitor:
    """Monitors running target processes on Windows or POSIX systems."""

    def __init__(self, target_pid: int | None = None):
        self.target_pid = target_pid
        self.cpu_history = deque(maxlen=32)
        self.mem_history = deque(maxlen=32)
        self.prev_cpu_time = 0.0
        self.prev_timestamp = time.time()
        self.detected_command = ""
        self.job_type = "unknown"

    def scan_for_process(self) -> dict[str, Any] | None:
        """Find the active shikra background process."""
        my_pid = os.getpid()

        # If explicit PID specified
        if self.target_pid:
            p = self._get_pid_info(self.target_pid)
            if p:
                return p

        # Otherwise query system for python processes
        candidates = []
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                    "Select-Object ProcessId, CommandLine, CreationDate, UserModeTime, KernelModeTime | "
                    "ConvertTo-Json -Compress"
                ),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                items = data if isinstance(data, list) else [data]
                for it in items:
                    pid = it.get("ProcessId")
                    if pid == my_pid:
                        continue
                    cmdline = it.get("CommandLine") or ""
                    # Check for shikra relevant keywords
                    for kw in ["multi_condition", "shikra", "supply_shock", "engine", "heatmap", "summary"]:
                        if kw in cmdline.lower():
                            candidates.append({
                                "pid": pid,
                                "cmdline": cmdline,
                                "created": it.get("CreationDate"),
                                "raw_time": (it.get("UserModeTime", 0) + it.get("KernelModeTime", 0)) / 10000000.0,
                            })
                            break
        except Exception:
            pass

        if not candidates:
            return None

        # Prefer the one with highest CPU usage or most specific
        # Sort so that multi_condition or active workers come first
        candidates.sort(key=lambda x: x["raw_time"], reverse=True)
        top = candidates[0]
        self.target_pid = top["pid"]
        return self._get_pid_info(self.target_pid, top["cmdline"])

    def _get_pid_info(self, pid: int, cmdline_hint: str = "") -> dict[str, Any] | None:
        """Fetch current process CPU and memory metrics."""
        try:
            ps_script = f"""
            $p = Get-Process -Id {pid} -ErrorAction SilentlyContinue
            if ($p) {{
                [PSCustomObject]@{{
                    Id = $p.Id
                    Name = $p.ProcessName
                    CPU = $p.CPU
                    WS = $p.WorkingSet64
                    StartTime = $p.StartTime.ToString("o")
                    Responding = $p.Responding
                }} | ConvertTo-Json -Compress
            }}
            """
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                cpu_total = float(data.get("CPU", 0.0) or 0.0)
                ws_bytes = int(data.get("WS", 0) or 0)
                ws_mb = ws_bytes / (1024 * 1024)

                now = time.time()
                dt = now - self.prev_timestamp
                if dt > 0 and self.prev_cpu_time > 0:
                    cpu_delta = cpu_total - self.prev_cpu_time
                    # Multi-core CPU percentage estimate
                    cpu_pct = min(100.0 * os.cpu_count(), max(0.0, (cpu_delta / dt) * 100.0))
                else:
                    cpu_pct = 50.0  # reasonable initial estimate

                self.prev_cpu_time = cpu_total
                self.prev_timestamp = now

                self.cpu_history.append(min(100.0, cpu_pct / (os.cpu_count() or 1)))
                self.mem_history.append(ws_mb)

                if cmdline_hint:
                    self.detected_command = cmdline_hint

                # Classify job type
                cmd_low = self.detected_command.lower()
                if "multi_condition" in cmd_low:
                    self.job_type = "multi_condition"
                elif "supply_shock" in cmd_low:
                    self.job_type = "supply_shock"
                elif "heatmap" in cmd_low:
                    self.job_type = "heatmap"
                elif "summary" in cmd_low:
                    self.job_type = "summary"
                else:
                    self.job_type = "simulation"

                return {
                    "pid": pid,
                    "cpu_time": cpu_total,
                    "cpu_pct": cpu_pct,
                    "mem_mb": ws_mb,
                    "responding": data.get("Responding", True),
                    "start_time": data.get("StartTime", ""),
                    "command": self.detected_command or "python workload",
                    "job_type": self.job_type,
                    "is_running": True,
                }
        except Exception:
            pass

        return None


def get_pipeline_status(job_type: str) -> list[dict[str, str]]:
    """Determine progress through stages based on artifact and result files."""
    stages = []

    if job_type == "multi_condition":
        splits_path = ROOT / "artifacts" / "splits_multi.json"
        results_path = ROOT / "results" / "rul_cmapss_multi.csv"

        has_splits = splits_path.exists()
        has_results = results_path.exists()

        stages.append({
            "name": "Multi-Condition Split Generation (seed=2026)",
            "status": "COMPLETED" if has_splits else "RUNNING",
            "info": "Splits 80/20 train/val for FD002 & FD004",
        })
        stages.append({
            "name": "FD002 Condition Clustering & Normalization",
            "status": "COMPLETED" if has_results else ("RUNNING" if has_splits else "PENDING"),
            "info": "KMeans k=6 clusters on op1, op2, op3",
        })
        stages.append({
            "name": "FD002 Causal History & Feature Windows",
            "status": "COMPLETED" if has_results else ("RUNNING" if has_splits else "PENDING"),
            "info": "Rolling lag windows [5, 10, 15] & causal slopes",
        })
        stages.append({
            "name": "FD002 LightGBM Quantile Ensemble",
            "status": "COMPLETED" if has_results else ("RUNNING" if has_splits else "PENDING"),
            "info": "15 models: 5 seeds × 3 quantiles (0.10, 0.50, 0.90)",
        })
        stages.append({
            "name": "FD004 Dual Fault-Mode Evaluation",
            "status": "COMPLETED" if has_results else ("RUNNING" if has_splits else "PENDING"),
            "info": "249 train engines, HPC + Fan degradation",
        })
        stages.append({
            "name": "Export Artifacts & Test Metrics",
            "status": "COMPLETED" if has_results else "WAITING",
            "info": "results/rul_cmapss_multi.csv (NASA, RMSE, Cov80)",
        })

    elif job_type == "supply_shock":
        shock_json = ROOT / "artifacts" / "shock_results.json"
        shock_csv = ROOT / "results" / "supply_shock.csv"

        stages.append({
            "name": "Wing Configuration & Policy Setup",
            "status": "COMPLETED",
            "info": "54 aircraft, 3 squadrons, policy shikra_v2",
        })
        stages.append({
            "name": "Foreign LRU Shock Sweep (30 Seeds)",
            "status": "COMPLETED" if shock_json.exists() or shock_csv.exists() else "RUNNING",
            "info": "Embargo on Day 90–270: 5 foreign LRUs + All",
        })
        stages.append({
            "name": "Bootstrap CI & Recovery Calculation",
            "status": "COMPLETED" if shock_json.exists() or shock_csv.exists() else "PENDING",
            "info": "95% bootstrap percentile CI & recovery days",
        })
        stages.append({
            "name": "Contract §6 Indigenisation Ranking",
            "status": "COMPLETED" if shock_csv.exists() else "PENDING",
            "info": "Ranked priorities: results/supply_shock.csv",
        })

    else:
        stages.append({
            "name": "Process Execution & Simulation Loop",
            "status": "RUNNING",
            "info": "Evaluating 365-day fleet lifecycle",
        })

    return stages


def get_latest_artifacts_summary() -> list[str]:
    """Inspect and format key result artifacts."""
    lines = []

    # 1. Multi condition results
    multi_csv = ROOT / "results" / "rul_cmapss_multi.csv"
    if multi_csv.exists():
        try:
            with open(multi_csv, encoding="utf-8") as f:
                content = f.read().strip().split("\n")
                if len(content) > 1:
                    lines.append(f"{BOLD}{FG_BRIGHT_CYAN}MULTI-CONDITION C-MAPSS TEST EVALUATION (FD002 / FD004):{RESET}")
                    headers = [h.strip() for h in content[0].split(",")]
                    lines.append(f"{FG_GRAY}  ┌" + "─" * 68 + "┐" + RESET)
                    for row in content[1:]:
                        cols = [c.strip() for c in row.split(",")]
                        row_dict = dict(zip(headers, cols))
                        sub = row_dict.get("subset", "SUB")
                        rmse = row_dict.get("rmse", "N/A")
                        mae = row_dict.get("mae", "N/A")
                        nasa = row_dict.get("nasa_score", "N/A")
                        cov = row_dict.get("coverage_80", "N/A")
                        try:
                            cov_str = f"{float(cov)*100:.1f}%" if float(cov) <= 1.0 else f"{cov}%"
                        except Exception:
                            cov_str = cov
                        lines.append(
                            f"  │ {BOLD}{FG_BRIGHT_WHITE}{sub:7s}{RESET} "
                            f"RMSE: {FG_BRIGHT_GREEN}{rmse:6s}{RESET} "
                            f"MAE: {FG_CYAN}{mae:6s}{RESET} "
                            f"NASA: {FG_BRIGHT_YELLOW}{nasa:8s}{RESET} "
                            f"Cov80%: {FG_BRIGHT_MAGENTA}{cov_str:6s}{RESET} │"
                        )
                    lines.append(f"{FG_GRAY}  └" + "─" * 68 + "┘" + RESET)
        except Exception:
            pass

    # 2. Shock results
    shock_json = ROOT / "artifacts" / "shock_results.json"
    if shock_json.exists():
        try:
            with open(shock_json, encoding="utf-8") as f:
                data = json.load(f)
                lines.append(f"{BOLD}{FG_BRIGHT_YELLOW}SUPPLY SHOCK STRESS-TEST SUMMARY (30 SEEDS BOOTSTRAP):{RESET}")
                lines.append(f"{FG_GRAY}  ┌────────────────────────┬─────────────────────┬────────────────────┐{RESET}")
                lines.append(f"  │ {BOLD}{'SCENARIO':22s}{RESET} │ {BOLD}{'SVC LOSS (pp)':19s}{RESET} │ {BOLD}{'RECOVERY HORIZON':18s}{RESET} │")
                lines.append(f"{FG_GRAY}  ├────────────────────────┼─────────────────────┼────────────────────┤{RESET}")
                for k, v in data.items():
                    loss = v.get("loss_mean", 0.0)
                    rec = v.get("recov_mean", 0.0)
                    color = FG_BRIGHT_RED if loss > 2.0 else (FG_BRIGHT_YELLOW if loss > 0.5 else FG_BRIGHT_GREEN)
                    lines.append(
                        f"  │ {FG_BRIGHT_WHITE}{k:22s}{RESET} │ "
                        f"{color}{loss:6.3f} pp{RESET}             │ "
                        f"{FG_CYAN}{rec:5.1f} days{RESET}          │"
                    )
                lines.append(f"{FG_GRAY}  └────────────────────────┴─────────────────────┴────────────────────┘{RESET}")
        except Exception:
            pass

    return lines


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    s = int(seconds)
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    if hrs > 0:
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


def render_dashboard(mon: ProcessMonitor, proc: dict[str, Any] | None) -> str:
    """Build the full ANSI terminal visualizer frame."""
    width = 78
    now_str = time.strftime("%H:%M:%S")

    # Header Box
    out = []
    out.append(f"{FG_BRIGHT_CYAN}╔═" + "═" * (width - 4) + "═╗" + RESET)
    out.append(
        f"{FG_BRIGHT_CYAN}║{RESET}  "
        f"{BOLD}{FG_BRIGHT_WHITE}SHIKRA{RESET} {FG_CYAN}TELEMETRY VISUALIZER{RESET} "
        f"{DIM}v2.0{RESET}"
        + " " * (width - 48) +
        f"{FG_GRAY}[SYS TIME: {now_str}]{RESET}  "
        f"{FG_BRIGHT_CYAN}║{RESET}"
    )
    out.append(f"{FG_BRIGHT_CYAN}╠═" + "═" * (width - 4) + "═╣" + RESET)

    # Process Status Box
    if proc and proc["is_running"]:
        pid = proc["pid"]
        cpu_pct = proc["cpu_pct"]
        cpu_time = proc["cpu_time"]
        mem_mb = proc["mem_mb"]
        cmd = proc["command"]
        if len(cmd) > width - 18:
            cmd = "..." + cmd[-(width - 21):]

        status_badge = f"{BG_CYAN}{FG_BLACK}{BOLD} RUNNING {RESET}"
        out.append(f"{FG_BRIGHT_CYAN}║{RESET} STATUS: {status_badge}  {BOLD}PID: {pid}{RESET}  {DIM}CMD: {cmd}{RESET}")

        # CPU Metric Bar & Sparkline
        cpu_norm = min(100.0, cpu_pct / (os.cpu_count() or 1))
        cpu_bar = render_progress_bar(cpu_norm, width=22, color=FG_BRIGHT_CYAN)
        spark = render_sparkline(list(mon.cpu_history), max_val=100.0)
        out.append(
            f"{FG_BRIGHT_CYAN}║{RESET} CPU LOAD : [{cpu_bar}]  "
            f"Trend: [{spark}]"
        )

        # Memory Metric Bar & Total CPU Time
        mem_pct = min(100.0, (mem_mb / 4096.0) * 100.0)
        mem_bar = render_progress_bar(mem_pct, width=22, color=FG_BRIGHT_MAGENTA)
        out.append(
            f"{FG_BRIGHT_CYAN}║{RESET} MEMORY   : [{mem_bar}] {mem_mb:6.1f} MB  "
            f"CPU Accum: {FG_BRIGHT_YELLOW}{format_duration(cpu_time)}{RESET}"
        )
    else:
        status_badge = f"{BG_DARK_GRAY}{FG_WHITE}{BOLD} IDLE / COMPLETED {RESET}"
        out.append(f"{FG_BRIGHT_CYAN}║{RESET} STATUS: {status_badge}  No active compute tasks detected in terminal.")
        out.append(f"{FG_BRIGHT_CYAN}║{RESET} System is ready. Standby for next simulation or training job.")

    out.append(f"{FG_BRIGHT_CYAN}╠═" + "═" * (width - 4) + "═╣" + RESET)

    # Pipeline Stages Section
    job_type = proc.get("job_type", "multi_condition") if proc else "multi_condition"
    out.append(f"{FG_BRIGHT_CYAN}║{RESET} {BOLD}{FG_BRIGHT_WHITE}ACTIVE WORKLOAD PIPELINE: {job_type.upper()}{RESET}")
    stages = get_pipeline_status(job_type)

    for st in stages:
        s_name = st["name"]
        status = st["status"]
        info = st["info"]
        if status == "COMPLETED":
            icon = f"{FG_BRIGHT_GREEN}✔ COMPLETED{RESET}"
        elif status == "RUNNING":
            icon = f"{FG_BRIGHT_YELLOW}▶ IN PROGRESS{RESET}"
        elif status == "WAITING":
            icon = f"{FG_CYAN}⏳ WAITING{RESET}"
        else:
            icon = f"{FG_GRAY}• PENDING{RESET}"

        out.append(f"{FG_BRIGHT_CYAN}║{RESET}  {icon:18s} {BOLD}{s_name:42s}{RESET}")
        out.append(f"{FG_BRIGHT_CYAN}║{RESET}     {FG_GRAY}↳ {info}{RESET}")

    out.append(f"{FG_BRIGHT_CYAN}╠═" + "═" * (width - 4) + "═╣" + RESET)

    # Latest Artifacts & Metrics Section
    art_lines = get_latest_artifacts_summary()
    if art_lines:
        for al in art_lines:
            out.append(f"{FG_BRIGHT_CYAN}║{RESET} {al}")
        out.append(f"{FG_BRIGHT_CYAN}╠═" + "═" * (width - 4) + "═╣" + RESET)

    # Footer Controls
    out.append(
        f"{FG_BRIGHT_CYAN}║{RESET}  "
        f"{FG_GRAY}Press {BOLD}Ctrl+C{RESET}{FG_GRAY} to stop visualizer | "
        f"Web UI: {FG_CYAN}http://localhost:8765{RESET}{FG_GRAY} | "
        f"Refresh: 1.0s{RESET}"
        + " " * 8 +
        f"{FG_BRIGHT_CYAN}║{RESET}"
    )
    out.append(f"{FG_BRIGHT_CYAN}╚═" + "═" * (width - 4) + "═╝" + RESET)

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="SHIKRA Live Terminal Visualizer")
    parser.add_argument("--pid", type=int, default=None, help="Specific Process ID to monitor")
    parser.add_argument("--once", action="store_true", help="Print single snapshot and exit")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    # Enable ANSI escape processing in Windows console
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # STD_OUTPUT_HANDLE = -11
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    mon = ProcessMonitor(target_pid=args.pid)

    if args.once:
        proc = mon.scan_for_process()
        print(render_dashboard(mon, proc))
        return

    # Clear terminal screen once
    sys.stdout.write(f"{ESC}2J{ESC}H")
    sys.stdout.flush()

    try:
        while True:
            proc = mon.scan_for_process()
            dashboard = render_dashboard(mon, proc)

            # Move cursor to home position (0,0) and overwrite screen smoothly
            sys.stdout.write(f"{ESC}H")
            sys.stdout.write(dashboard)
            sys.stdout.write(f"{ESC}J")  # Clear any remaining lines below
            sys.stdout.flush()

            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write(f"\n{FG_CYAN}Visualizer detached.{RESET}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
