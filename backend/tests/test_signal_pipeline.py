"""Synthetic-signal validation harness for the rPPG signal pipeline.

Why this exists
---------------
Real BP/glucose accuracy cannot be validated without clinical ground truth.
What we CAN validate is that the signal-processing core is correct and robust:
given a synthetic pulse at a *known* heart rate, the pipeline should recover that
rate; SNR should track signal cleanliness; NaN/Inf inputs must not crash; and
non-uniform frame timing must be corrected. This harness measures exactly that
and doubles as a regression guard (POS vs CHROM, HR error budget).

Run it two ways:
    python backend/tests/test_signal_pipeline.py     # standalone, prints a report
    pytest backend/tests/test_signal_pipeline.py     # if pytest is installed

It only needs numpy + scipy (no cv2/mediapipe/tensorflow).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make `utils.signal_processing` importable whether run from repo root or here.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.signal_processing import (  # noqa: E402
    bandpass_filter,
    chrom_rppg,
    detrend_signal,
    effective_fps,
    estimate_hr,
    normalize_signal,
    pos_rppg,
    resample_uniform,
    signal_snr,
)


# ── synthetic data ───────────────────────────────────────────────────────────
def synth_rgb(hr_bpm, fs, duration_s, noise=0.01, drift=0.0, seed=0):
    """Build a (T, 3) RGB series with a pulse at hr_bpm embedded in all channels
    (green strongest), plus Gaussian noise and an optional linear lighting drift."""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    base = np.array([0.60, 0.50, 0.40])     # baseline R,G,B
    amp = np.array([0.010, 0.020, 0.005])   # pulsatile amplitude per channel
    rgb = np.empty((n, 3))
    ramp = (t / t[-1]) if t[-1] > 0 else np.zeros_like(t)
    for c in range(3):
        rgb[:, c] = base[c] + amp[c] * pulse + drift * ramp + noise * rng.standard_normal(n)
    return rgb, t


def recover_hr(rgb, fs, method="pos"):
    pulse = pos_rppg(rgb, fs) if method == "pos" else chrom_rppg(rgb)
    ppg = normalize_signal(bandpass_filter(pulse, fs=fs, low=0.7, high=4.0, order=3))
    hr, ok = estimate_hr(ppg, fs)
    return hr, ok


# ── checks (named test_* so pytest also collects them) ───────────────────────
def test_estimate_hr_on_clean_sine():
    fs, hr_true = 30.0, 72.0
    t = np.arange(int(fs * 12)) / fs
    sig = np.sin(2 * np.pi * (hr_true / 60.0) * t)
    hr, ok = estimate_hr(sig, fs)
    assert ok, "clean sine should be a valid HR"
    assert abs(hr - hr_true) <= 2.0, f"HR error {abs(hr - hr_true):.2f} bpm > 2"


def test_pos_recovers_hr_clean():
    fs, hr_true = 30.0, 78.0
    rgb, _ = synth_rgb(hr_true, fs, 12, noise=0.004, drift=0.05, seed=1)
    hr, ok = recover_hr(rgb, fs, "pos")
    assert ok and abs(hr - hr_true) <= 3.0, f"POS HR {hr:.1f} vs {hr_true} (err>{3})"


def test_pos_recovers_hr_noisy():
    fs, hr_true = 30.0, 66.0
    rgb, _ = synth_rgb(hr_true, fs, 12, noise=0.02, drift=0.1, seed=2)
    hr, ok = recover_hr(rgb, fs, "pos")
    assert abs(hr - hr_true) <= 6.0, f"POS noisy HR {hr:.1f} vs {hr_true} (err>6)"


def test_snr_orders_clean_above_noise():
    fs = 30.0
    rgb_clean, _ = synth_rgb(72, fs, 12, noise=0.003, seed=3)
    rgb_noisy, _ = synth_rgb(72, fs, 12, noise=0.05, seed=4)
    ppg_clean = normalize_signal(bandpass_filter(pos_rppg(rgb_clean, fs), fs=fs))
    ppg_noisy = normalize_signal(bandpass_filter(pos_rppg(rgb_noisy, fs), fs=fs))
    snr_clean = signal_snr(ppg_clean, fs)
    snr_noisy = signal_snr(ppg_noisy, fs)
    assert snr_clean > snr_noisy, f"clean SNR {snr_clean:.1f} !> noisy {snr_noisy:.1f}"


def test_nan_inputs_do_not_crash():
    fs = 30.0
    rgb, _ = synth_rgb(72, fs, 10, noise=0.01, seed=5)
    rgb[10, 1] = np.nan
    rgb[20, 0] = np.inf
    pulse = pos_rppg(rgb, fs)
    ppg = normalize_signal(bandpass_filter(detrend_signal(pulse), fs=fs))
    assert np.all(np.isfinite(ppg)), "pipeline must produce finite output for NaN/Inf input"
    hr, _ = estimate_hr(ppg, fs)
    assert np.isfinite(hr)


def test_resample_uniform_recovers_hr_from_jittered_timing():
    fs_nominal, hr_true = 30.0, 70.0
    rgb, t = synth_rgb(hr_true, fs_nominal, 12, noise=0.005, seed=6)
    # jitter the timestamps (browser setInterval drift) up to ±40%
    rng = np.random.default_rng(7)
    jitter = (rng.random(len(t)) - 0.5) * 0.8 / fs_nominal
    t_jit = np.cumsum(np.abs(np.diff(t, prepend=t[0]) + jitter))
    fs_eff = effective_fps(t_jit)
    assert fs_eff > 0
    target = round(fs_eff)
    rgb_uni = resample_uniform(rgb, t_jit, target)
    hr, ok = recover_hr(rgb_uni, target, "pos")
    assert abs(hr - hr_true) <= 6.0, f"jittered HR {hr:.1f} vs {hr_true} (err>6)"


def test_pos_vs_chrom_regression():
    """POS should be at least as accurate as CHROM on average across noise levels."""
    fs, hr_true = 30.0, 75.0
    pos_err, chrom_err = [], []
    for i, noise in enumerate([0.005, 0.01, 0.02, 0.04]):
        rgb, _ = synth_rgb(hr_true, fs, 12, noise=noise, drift=0.08, seed=10 + i)
        pos_hr, _ = recover_hr(rgb, fs, "pos")
        chrom_hr, _ = recover_hr(rgb, fs, "chrom")
        pos_err.append(abs(pos_hr - hr_true))
        chrom_err.append(abs(chrom_hr - hr_true))
    mean_pos, mean_chrom = float(np.mean(pos_err)), float(np.mean(chrom_err))
    # Allow a small slack; the point is POS must not regress materially vs CHROM.
    assert mean_pos <= mean_chrom + 1.5, f"POS mean err {mean_pos:.2f} >> CHROM {mean_chrom:.2f}"
    return mean_pos, mean_chrom


# ── standalone runner ────────────────────────────────────────────────────────
def _run():
    checks = [
        ("estimate_hr clean sine", test_estimate_hr_on_clean_sine),
        ("POS recovers HR (clean)", test_pos_recovers_hr_clean),
        ("POS recovers HR (noisy)", test_pos_recovers_hr_noisy),
        ("SNR orders clean>noisy", test_snr_orders_clean_above_noise),
        ("NaN/Inf inputs safe", test_nan_inputs_do_not_crash),
        ("resample fixes jittered fps", test_resample_uniform_recovers_hr_from_jittered_timing),
        ("POS vs CHROM regression", test_pos_vs_chrom_regression),
    ]
    print("rPPG signal pipeline - synthetic validation")
    print("=" * 52)
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    # Informational POS-vs-CHROM accuracy table.
    print("-" * 52)
    fs, hr_true = 30.0, 75.0
    print(f"POS vs CHROM HR error (bpm), true={hr_true:.0f}, fs={fs:.0f}:")
    for noise in [0.005, 0.01, 0.02, 0.04]:
        rgb, _ = synth_rgb(hr_true, fs, 12, noise=noise, drift=0.08, seed=99)
        p, _ = recover_hr(rgb, fs, "pos")
        c, _ = recover_hr(rgb, fs, "chrom")
        print(f"  noise={noise:<5}  POS={abs(p-hr_true):5.2f}  CHROM={abs(c-hr_true):5.2f}")
    print("=" * 52)
    print("ALL PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
