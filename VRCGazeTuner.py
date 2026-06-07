"""
VRCGazeTuner.py — OSC proxy for VRCFaceTracking → VRChat
Eye tracking focus: polar gaze visualization with live gain/offset tuning.

In VRCFT settings, set Send Port to 9100 and IP to 127.0.0.1.
"""

import json
import math
import threading
import time
import collections
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# ── Ports ─────────────────────────────────────────────────────────────────────
LISTEN_IP,  LISTEN_PORT = "127.0.0.1", 9100
VRCHAT_IP,  VRCHAT_PORT = "127.0.0.1", 9000
CONFIG_PATH = (Path(sys.executable).parent if getattr(sys, "frozen", False)
               else Path(__file__).parent) / "proxy_config.json"
HISTORY_LEN = 60

# ── Address suffix → (side, axis) mapping (handles any VRCFT prefix) ─────────
# Matches /avatar/parameters/v2/EyeX  AND  /avatar/parameters/FT/v2/EyeX etc.
EYE_SUFFIX_MAP = {
    "/v2/EyeX":         ("combined", "x"),
    "/v2/EyeY":         ("combined", "y"),
    "/v2/EyeLeftX":     ("left",     "x"),
    "/v2/EyeLeftY":     ("left",     "y"),
    "/v2/EyeRightX":    ("right",    "x"),
    "/v2/EyeRightY":    ("right",    "y"),
    "/v2/EyeLid":       ("combined", "lid"),
    "/v2/EyeLidLeft":   ("left",     "lid"),
    "/v2/EyeLidRight":  ("right",    "lid"),
}

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "/avatar/parameters/v2/EyeX":         {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeY":         {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeLeftX":     {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeLeftY":     {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeRightX":    {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeRightY":    {"gain": 1.0, "offset": 0.0, "min": -1.0, "max": 1.0},
    "/avatar/parameters/v2/EyeLid":       {"gain": 1.0, "offset": 0.0, "min": 0.0,  "max": 1.0},
    "/avatar/parameters/v2/EyeLidLeft":   {"gain": 1.0, "offset": 0.0, "min": 0.0,  "max": 1.0},
    "/avatar/parameters/v2/EyeLidRight":  {"gain": 1.0, "offset": 0.0, "min": 0.0,  "max": 1.0},
    "/avatar/parameters/v2/JawOpen":      {"gain": 1.0, "offset": 0.0, "min": 0.0,  "max": 1.0},
}

# Slider groups: (row label, [(addr, param, vmin, vmax, slider label), ...])
# Y is a single shared parameter across both eyes — one row for it.
SLIDER_GROUPS = [
    ("Left Eye X",  [("/avatar/parameters/v2/EyeLeftX",  "gain",   0.0,  3.0,  "Gain"),
                     ("/avatar/parameters/v2/EyeLeftX",  "offset", -0.5, 0.5,  "Offset")]),
    ("Right Eye X", [("/avatar/parameters/v2/EyeRightX", "gain",   0.0,  3.0,  "Gain"),
                     ("/avatar/parameters/v2/EyeRightX", "offset", -0.5, 0.5,  "Offset")]),
    ("Both Eyes Y", [("/avatar/parameters/v2/EyeY",      "gain",   0.0,  3.0,  "Gain"),
                     ("/avatar/parameters/v2/EyeY",      "offset", -0.5, 0.5,  "Offset")]),
]

# ── Shared state ──────────────────────────────────────────────────────────────
_state_lock  = threading.Lock()
_config_lock = threading.Lock()
_config: dict = {}
_passthrough  = False

_gaze = {
    side: {"rx": 0.0, "ry": 0.0, "ox": 0.0, "oy": 0.0,
           "lid_r": 0.75, "lid_o": 0.75}
    for side in ("left", "right", "combined")
}
_per_eye_x_ts = {"left": 0.0, "right": 0.0}
_per_eye_y_ts = {"left": 0.0, "right": 0.0}
_history    = {"left":  collections.deque(maxlen=HISTORY_LEN),
               "right": collections.deque(maxlen=HISTORY_LEN)}
_pkt_count  = 0
_fwd: SimpleUDPClient = None


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    global _config
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            with _config_lock:
                _config = data.get("parameters", {})
            print(f"[config] loaded {len(_config)} rules from {CONFIG_PATH.name}")
            return
        except Exception as e:
            print(f"[config] load error: {e}")
    with _config_lock:
        _config = dict(DEFAULT_PARAMS)
    print("[config] using built-in defaults")


def save_config():
    with _config_lock:
        snapshot = {"parameters": dict(_config)}
    CONFIG_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"[config] saved -> {CONFIG_PATH}")


def get_rule(addr: str) -> dict:
    with _config_lock:
        if addr in _config:
            return dict(_config[addr])
        # Both "/avatar/parameters/FT/v2/EyeX" and "/avatar/parameters/v2/EyeX"
        # share the same "/v2/EyeX" tail — match on that.
        if "/v2/" in addr:
            tail = addr[addr.rfind("/v2/"):]   # e.g. "/v2/EyeLeftX"
            for key, rule in _config.items():
                if key.endswith(tail):
                    return dict(rule)
        return {}


def set_param(addr: str, **kw):
    with _config_lock:
        _config.setdefault(addr, {}).update(kw)


# ── Transform ─────────────────────────────────────────────────────────────────
def apply_transform(addr: str, value):
    if _passthrough or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    rule = get_rule(addr)
    if not rule:
        return value
    out = value * rule.get("gain", 1.0) + rule.get("offset", 0.0)
    return max(rule.get("min", -1.0), min(rule.get("max", 1.0), out))


# ── OSC handler ───────────────────────────────────────────────────────────────
def osc_handler(address: str, *args):
    global _pkt_count

    new_args = [apply_transform(address, a) for a in args]
    try:
        _fwd.send_message(address, new_args[0] if len(new_args) == 1 else new_args)
    except Exception:
        pass

    _pkt_count += 1
    val_raw = args[0]    if args     else 0.0
    val_out = new_args[0] if new_args else 0.0

    for suffix, (side, axis) in EYE_SUFFIX_MAP.items():
        if address.endswith(suffix):
            now = time.monotonic()
            with _state_lock:
                g = _gaze[side]
                if axis == "x":
                    g["rx"], g["ox"] = val_raw, val_out
                    if side in ("left", "right"):
                        _per_eye_x_ts[side] = now
                elif axis == "y":
                    g["ry"], g["oy"] = val_raw, val_out
                    if side in ("left", "right"):
                        _per_eye_y_ts[side] = now
                elif axis == "lid":
                    g["lid_r"], g["lid_o"] = val_raw, val_out
            break


def read_eye(side: str):
    """Return (rx, ry, ox, oy, lid_r, lid_o).
    X and Y fall back to combined independently — handles avatars that send
    per-eye X but only a single combined Y (e.g. FT/v2/EyeLeftX + FT/v2/EyeY)."""
    now = time.monotonic()
    with _state_lock:
        has_x = now - _per_eye_x_ts[side] < 1.0
        has_y = now - _per_eye_y_ts[side] < 1.0
        g, c  = _gaze[side], _gaze["combined"]
        return (
            g["rx"]  if has_x else c["rx"],
            g["ry"]  if has_y else c["ry"],
            g["ox"]  if has_x else c["ox"],
            g["oy"]  if has_y else c["oy"],
            g["lid_r"] if has_x else c["lid_r"],
            g["lid_o"] if has_x else c["lid_o"],
        )


def xy_to_polar(x, y):
    return math.atan2(y, x), math.sqrt(x * x + y * y)


# ── Figure builder ────────────────────────────────────────────────────────────
BG    = "#0d0d1a"
PANEL = "#121220"
GRID  = "#1e1e38"
DIM   = "#404060"
LITE  = "#a0a0cc"
BLUE  = "#3388ff"
LBLUE = "#66aaff"

def _style_polar(ax, title: str):
    ax.set_title(title, color=LITE, pad=10, fontsize=11)
    ax.set_facecolor(PANEL)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color=DIM, fontsize=7)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    ax.set_xticklabels(["R →", "↑ U", "← L", "↓ D"], color="#6060a0", fontsize=9)
    ax.set_theta_zero_location("E")   # EyeX+ = East = right
    ax.set_theta_direction(1)         # counterclockwise (matches atan2 convention)
    ax.grid(color=GRID, linewidth=0.7, linestyle="--")
    ax.spines["polar"].set_color(DIM)


def _make_slider(fig, rect, vmin, vmax, vinit):
    ax = fig.add_axes(rect, facecolor="#111128")
    s  = Slider(ax, "", vmin, vmax, valinit=vinit, color="#335599", track_color="#0e0e22")
    # Move value text inside the slider (right-aligned) so it can't bleed into adjacent widgets
    s.valtext.set_position((0.97, 0.5))
    s.valtext.set_ha("right")
    s.valtext.set_va("center")
    s.valtext.set_color(LBLUE)
    s.valtext.set_fontsize(8)
    return s


def _make_button(fig, rect, label):
    ax = fig.add_axes(rect)
    b  = Button(ax, label, color="#1a1a30", hovercolor="#252545")
    b.label.set_color("white"); b.label.set_fontsize(7)
    return b


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _fwd, _passthrough

    ap = argparse.ArgumentParser(description="VRCFT OSC proxy with eye tracking visualizer")
    ap.add_argument("--no-gui",      action="store_true", help="Headless / no plot window")
    ap.add_argument("--passthrough", action="store_true", help="Forward without transforms")
    ap.add_argument("--listen-port", type=int, default=LISTEN_PORT,
                    help=f"Port proxy listens on (default {LISTEN_PORT})")
    ap.add_argument("--debug",       action="store_true",
                    help="Print every unique OSC address seen (use to find missing EyeY etc.)")
    cli = ap.parse_args()
    _passthrough = cli.passthrough

    load_config()

    _fwd = SimpleUDPClient(VRCHAT_IP, VRCHAT_PORT)
    disp = Dispatcher()
    disp.set_default_handler(osc_handler)
    server = ThreadingOSCUDPServer((LISTEN_IP, cli.listen_port), disp)
    if cli.debug:
        _log = open(Path(__file__).parent / "debug_addrs.txt", "w")
        _seen: set = set()
        _orig_handler = osc_handler
        def _debug_handler(address: str, *args):
            _orig_handler(address, *args)
            if address not in _seen:
                _seen.add(address)
                val = repr(args[0]) if args else "?"
                _log.write(f"{address}  val={val}\n")
                _log.flush()
        disp.set_default_handler(_debug_handler)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[proxy] listening {LISTEN_IP}:{cli.listen_port}  ->  VRChat {VRCHAT_IP}:{VRCHAT_PORT}")
    if _passthrough:
        print("[proxy] passthrough mode — transforms disabled")

    if cli.no_gui:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        server.shutdown()
        return

    # ── Figure ────────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "text.color":       LITE,
        "axes.labelcolor":  DIM,
        "xtick.color":      DIM,
        "ytick.color":      DIM,
    })

    fig = plt.figure(figsize=(13, 9))
    fig.canvas.manager.set_window_title("VRCFT Eye Proxy")
    fig.suptitle("VRCFT Eye Tracking Proxy", color="#c0c0e8", fontsize=13,
                 fontweight="bold", y=0.975)

    # Polar subplots occupy top portion
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.42)
    ax_l = fig.add_subplot(1, 2, 1, projection="polar")
    ax_r = fig.add_subplot(1, 2, 2, projection="polar")
    _style_polar(ax_l, "◉  Left Eye")
    _style_polar(ax_r, "◉  Right Eye")

    # Center crosshair marker on each plot
    for ax in (ax_l, ax_r):
        ax.plot([0], [0], "+", color="#ffffff", markersize=10, zorder=6, markeredgewidth=1.2)

    # Gaze dots: history trail, raw (gray), transformed (blue)
    hist_l, = ax_l.plot([], [], "o", color="#555577", markersize=4,  alpha=0.45, linestyle="none", zorder=2)
    hist_r, = ax_r.plot([], [], "o", color="#555577", markersize=4,  alpha=0.45, linestyle="none", zorder=2)
    raw_l,  = ax_l.plot([], [], "o", color="#cccccc", markersize=12, zorder=4,
                         markeredgecolor="#ffffff", markeredgewidth=0.8, label="Raw")
    raw_r,  = ax_r.plot([], [], "o", color="#cccccc", markersize=12, zorder=4,
                         markeredgecolor="#ffffff", markeredgewidth=0.8, label="Raw")
    out_l,  = ax_l.plot([], [], "o", color=BLUE,     markersize=12, zorder=5,
                         markeredgecolor=LBLUE, markeredgewidth=0.8, label="Transformed")
    out_r,  = ax_r.plot([], [], "o", color=BLUE,     markersize=12, zorder=5,
                         markeredgecolor=LBLUE, markeredgewidth=0.8, label="Transformed")

    # Connector line between raw → transformed (shows the shift clearly)
    line_l, = ax_l.plot([], [], "-", color="#224488", linewidth=1.2, zorder=3, alpha=0.7)
    line_r, = ax_r.plot([], [], "-", color="#224488", linewidth=1.2, zorder=3, alpha=0.7)

    for ax in (ax_l, ax_r):
        ax.legend(loc="lower right", fontsize=8, framealpha=0.5,
                  facecolor="#141428", edgecolor=DIM, labelcolor="white", markerscale=0.7)

    # ── Eyelid bars ───────────────────────────────────────────────────────────
    ELID_Y, ELID_H = 0.377, 0.030
    lid_l_ax = fig.add_axes([0.055, ELID_Y, 0.415, ELID_H], facecolor="#0a0a16")
    lid_r_ax = fig.add_axes([0.530, ELID_Y, 0.415, ELID_H], facecolor="#0a0a16")
    # Labels sit above the bars as figure text (avoids cramming title inside a 30px tall axes)
    fig.text(0.055, ELID_Y + ELID_H + 0.004, "Left Eyelid",
             color=DIM, fontsize=7, va="bottom")
    fig.text(0.530, ELID_Y + ELID_H + 0.004, "Right Eyelid",
             color=DIM, fontsize=7, va="bottom")

    for lax in (lid_l_ax, lid_r_ax):
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
        lax.axvline(0.75, color=DIM, linewidth=0.8, linestyle=":")   # "normally open" marker
        lax.set_xticks([0.0, 0.75, 1.0])
        lax.set_xticklabels(["Closed", "Open (0.75)", "Wide"], color=DIM, fontsize=6)
        lax.tick_params(axis="x", pad=2, length=3)
        lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_color(GRID)

    lid_raw_l, = lid_l_ax.plot([0.75, 0.75], [0.1, 0.9], color="#888888", linewidth=5,  zorder=2, solid_capstyle="round")
    lid_out_l, = lid_l_ax.plot([0.75, 0.75], [0.15, 0.85], color=BLUE,   linewidth=3,  zorder=3, solid_capstyle="round")
    lid_raw_r, = lid_r_ax.plot([0.75, 0.75], [0.1, 0.9], color="#888888", linewidth=5,  zorder=2, solid_capstyle="round")
    lid_out_r, = lid_r_ax.plot([0.75, 0.75], [0.15, 0.85], color=BLUE,   linewidth=3,  zorder=3, solid_capstyle="round")

    # ── Sliders ───────────────────────────────────────────────────────────────
    # 3 rows × 2 wide cols (Gain | Offset).  Y is shared so only needs one row.
    col_x  = [0.12, 0.555]
    row_y  = [0.305, 0.255, 0.205]
    SW, SH = 0.390, 0.028

    HDR_Y = row_y[0] + SH + 0.010
    for cx, lbl in zip(col_x, ["Gain", "Offset"]):
        fig.text(cx + SW / 2, HDR_Y, lbl, color=LITE, fontsize=8, ha="center", va="bottom")

    slider_map = {}   # (ri, ci) → (Slider, addr, param)

    for ri, (group_lbl, col_defs) in enumerate(SLIDER_GROUPS):
        fig.text(0.01, row_y[ri] + SH / 2, group_lbl, color=LITE, fontsize=8, va="center")
        for ci, (addr, param, vmin, vmax, _) in enumerate(col_defs):
            rule  = get_rule(addr)
            vinit = rule.get(param, 1.0 if param == "gain" else 0.0)
            s = _make_slider(fig, [col_x[ci], row_y[ri], SW, SH], vmin, vmax, vinit)

            def on_change(val, a=addr, p=param):
                set_param(a, **{p: val})
            s.on_changed(on_change)
            slider_map[(ri, ci)] = (s, addr, param)

    # ── Buttons ───────────────────────────────────────────────────────────────
    BW, BH = 0.044, 0.028
    bx = 0.953   # right-aligned; bx + BW = 0.997 → just inside figure boundary

    btn_save  = _make_button(fig, [bx, row_y[0] + (SH - BH) / 2, BW, BH], "Save\nConfig")
    btn_reset = _make_button(fig, [bx, row_y[1] + (SH - BH) / 2, BW, BH], "Reset\nAll")
    btn_pass  = _make_button(fig, [bx, row_y[2] + (SH - BH) / 2, BW, BH], "Pass-\nthrough")
    btn_pass.label.set_color("#ffcc44")

    btn_save.on_clicked(lambda _: save_config())

    def do_reset(_):
        for (ri, ci), (s, addr, param) in slider_map.items():
            s.set_val(1.0 if param == "gain" else 0.0)
    btn_reset.on_clicked(do_reset)

    _pt_on = [_passthrough]
    def toggle_pass(_):
        global _passthrough
        _pt_on[0] = not _pt_on[0]
        _passthrough = _pt_on[0]
        on = _pt_on[0]
        btn_pass.label.set_text("PASS\nON" if on else "Pass-\nthrough")
        btn_pass.label.set_color("#ff4444" if on else "#ffcc44")
        btn_pass.ax.set_facecolor("#330000" if on else "#1a1a30")
        fig.canvas.draw_idle()
    btn_pass.on_clicked(toggle_pass)

    # ── Status / value readouts ───────────────────────────────────────────────
    # Positioned below the slider rows, two lines: per-eye readout + proxy status
    INFO_Y = row_y[-1] - 0.062
    ax_info = fig.add_axes([0.02, INFO_Y, 0.96, 0.052])
    ax_info.set_axis_off()
    ax_info.set_facecolor(BG)

    readout_l = ax_info.text(0.01, 0.78, "", color="#7788aa", fontsize=8,
                              va="center", family="monospace")
    readout_r = ax_info.text(0.51, 0.78, "", color="#7788aa", fontsize=8,
                              va="center", family="monospace")
    status    = ax_info.text(0.01, 0.22,
        f"Proxy  {LISTEN_IP}:{cli.listen_port}  →  VRChat {VRCHAT_IP}:{VRCHAT_PORT}"
        "   |   waiting for VRCFT packets...",
        color="#404060", fontsize=8, va="center", family="monospace")

    # ── Animation ─────────────────────────────────────────────────────────────
    _fps_t = [time.monotonic()]
    _fps_n = [0]

    def animate(_frame):
        for side, h_dot, r_dot, o_dot, conn, lraw, lout, rout_txt in [
            ("left",  hist_l, raw_l, out_l, line_l, lid_raw_l, lid_out_l, readout_l),
            ("right", hist_r, raw_r, out_r, line_r, lid_raw_r, lid_out_r, readout_r),
        ]:
            rx, ry, ox, oy, lid_r, lid_o = read_eye(side)
            _history[side].append((rx, ry))

            # History trail
            if len(_history[side]) > 1:
                pts = np.array(_history[side])
                thetas = np.arctan2(pts[:, 1], pts[:, 0])
                rs     = np.hypot(pts[:, 0], pts[:, 1])
                h_dot.set_data(thetas, rs)
            else:
                h_dot.set_data([], [])

            # Raw dot
            rt, rr = xy_to_polar(rx, ry)
            r_dot.set_data([rt], [rr])

            # Transformed dot
            ot, or_ = xy_to_polar(ox, oy)
            o_dot.set_data([ot], [or_])

            # Connector raw → transformed (only draw if meaningfully different)
            if abs(rx - ox) > 0.005 or abs(ry - oy) > 0.005:
                conn.set_data([rt, ot], [rr, or_])
            else:
                conn.set_data([], [])

            # Eyelid bars
            lraw.set_xdata([lid_r, lid_r])
            lout.set_xdata([lid_o, lid_o])

            # Value readout
            rout_txt.set_text(
                f"{side.capitalize()}: "
                f"raw ({rx:+.3f}, {ry:+.3f}) → out ({ox:+.3f}, {oy:+.3f})   "
                f"lid {lid_r:.2f}→{lid_o:.2f}"
            )

        # Status bar (update at ~2 Hz to avoid thrash)
        now = time.monotonic()
        if now - _fps_t[0] >= 0.5:
            pps = (_pkt_count - _fps_n[0]) / (now - _fps_t[0])
            _fps_t[0] = now
            _fps_n[0] = _pkt_count
            pt_tag = "  [PASSTHROUGH]" if _passthrough else ""
            status.set_text(
                f"Proxy  {LISTEN_IP}:{cli.listen_port}  →  VRChat {VRCHAT_IP}:{VRCHAT_PORT}"
                f"   |   {pps:.0f} pkt/s   total {_pkt_count}{pt_tag}"
            )
            status.set_color("#ff6644" if _passthrough else "#404060")

    _ani = animation.FuncAnimation(  # noqa: F841 — must stay alive
        fig, animate, interval=33, blit=False, cache_frame_data=False
    )
    plt.show()
    server.shutdown()


if __name__ == "__main__":
    main()
