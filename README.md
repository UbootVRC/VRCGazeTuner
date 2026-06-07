# VRCGazeTuner

Real-time OSC proxy for VRCFaceTracking — visualize and tune eye gaze offsets and gain with live polar plots.

## What it does

Sits between VRCFaceTracking (VRCFT) and VRChat, intercepting OSC face tracking data. For eye tracking parameters it lets you apply per-eye gain and offset corrections, and shows you the raw vs. transformed gaze position in real time on polar plots.

All other face tracking data (mouth, brows, etc.) passes through untouched.

## Usage

### Compiled exe (easiest)
Download `VRCGazeTuner.exe` from the releases — no Python required. Just run it.

### From source
```
pip install python-osc matplotlib numpy
python VRCGazeTuner.py
```

## Setup

1. In VRCFT settings, set **Send Port** to `9100` and **IP** to `127.0.0.1`
2. Run `VRCGazeTuner.exe` (or `VRCGazeTuner.py` from source)

## UI

- **Polar plots** — left and right eye gaze direction and magnitude. Gray dot = raw tracker value, blue dot = after your transforms. The history trail shows recent movement.
- **Eyelid bars** — live readout of eyelid openness (0 = closed, 0.75 = normally open, 1.0 = wide).
- **Sliders** — tune gain and offset independently for left eye X, right eye X, and the shared vertical Y.
- **Save Config** — writes current slider values to `proxy_config.json` next to the exe, loaded automatically on next launch.
- **Reset All** — returns all sliders to defaults (gain 1.0, offset 0.0).
- **Passthrough** — disables all transforms instantly for A/B comparison. Turns red when active.

## Notes

- Your avatar must have eye tracking parameters in its OSC config. If the plots show no movement, reset the OSC config in VRChat (Action Menu → OSC → Reset Config) and reload your avatar.
- Some avatars send a single shared `EyeY` for both eyes rather than separate per-eye Y values — in that case the Both Eyes Y slider row controls vertical movement for both eyes simultaneously.
- Config is saved to `proxy_config.json` next to the executable. Delete it to reset to defaults.

## Command line options

| Flag | Description |
|------|-------------|
| `--no-gui` | Run headless (proxy only, no window) |
| `--passthrough` | Forward all data without any transforms |
| `--listen-port N` | Change the proxy listen port (default 9100) |
| `--debug` | Write all received OSC addresses to `debug_addrs.txt` |
