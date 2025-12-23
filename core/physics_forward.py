# backend/app/core/physics_forward.py
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# Optional: use torch autograd if available (much faster + stabler gradient)
try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False


# =========================
# Utilities
# =========================

EPS = 1e-12


def _db_to_lin(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db) / 20.0)


def _lin_to_db(lin: np.ndarray) -> np.ndarray:
    lin = np.maximum(lin, 1e-20)
    return 20.0 * np.log10(lin)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _freq_axis_log(f_min: float, f_max: float, n_points: int) -> np.ndarray:
    f_min = max(float(f_min), 1e-6)
    f_max = max(float(f_max), f_min * 1.001)
    n_points = max(int(n_points), 16)
    return np.geomspace(f_min, f_max, n_points).astype(np.float64)


def _rfft_freq_bins(n_fft: int, fs: float) -> np.ndarray:
    # 0..fs/2, length n_fft//2+1
    return np.fft.rfftfreq(n_fft, d=1.0 / fs).astype(np.float64)


# =========================
# Param model (simple + stable for demo/UI)
# - Low shelf (bass)
# - High shelf (treble)
# - Peaking resonance
# Then: minimum-phase reconstruction -> IR
# =========================

@dataclass
class SimParams:
    # frequency grid for display (SPL curve)
    f_min: float = 20.0
    f_max: float = 20000.0
    n_points: int = 400

    # IR settings
    fs: float = 48000.0
    ir_len: int = 65536
    target_peak_index: int = 48000  # like your pipeline idea

    # "Tone controls"
    bass_gain_db: float = 6.0
    treble_gain_db: float = 0.0

    # resonance (peaking EQ)
    resonance_hz: float = 3000.0
    resonance_q: float = 1.2
    resonance_gain_db: float = 3.0


# Reasonable bounds for fitting / random generation
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bass_gain_db": (-12.0, 12.0),
    "treble_gain_db": (-12.0, 12.0),
    "resonance_hz": (200.0, 12000.0),
    "resonance_q": (0.2, 10.0),
    "resonance_gain_db": (-12.0, 12.0),
}


def _low_shelf_mag(freq: np.ndarray, gain_db: float, f0: float = 120.0, slope: float = 1.0) -> np.ndarray:
    """
    Smooth low-shelf magnitude shape (not an exact biquad; stable for UI).
    """
    g = _db_to_lin(gain_db)
    x = (freq / max(f0, 1e-6)) ** slope
    # goes from g at low freq to 1 at high freq if gain positive? we want bass boost:
    # Make low freq boosted: mag -> g when x small, ->1 when x large
    mag = (g + x) / (1.0 + x)
    return mag.astype(np.float64)


def _high_shelf_mag(freq: np.ndarray, gain_db: float, f0: float = 8000.0, slope: float = 1.0) -> np.ndarray:
    """
    Smooth high-shelf magnitude shape (stable for UI).
    """
    g = _db_to_lin(gain_db)
    x = (freq / max(f0, 1e-6)) ** slope
    # low freq -> 1, high freq -> g
    mag = (1.0 + g * x) / (1.0 + x)
    return mag.astype(np.float64)


def _peaking_mag(freq: np.ndarray, f0: float, q: float, gain_db: float) -> np.ndarray:
    """
    Smooth peaking magnitude approx in log-frequency domain.
    """
    f0 = max(float(f0), 1e-6)
    q = max(float(q), 1e-3)
    gain = _db_to_lin(gain_db)

    # Gaussian bump in log frequency; width controlled by Q
    logf = np.log(freq + 1e-30)
    logf0 = math.log(f0)
    # map Q to sigma: higher Q -> narrower peak
    sigma = 0.35 / q  # heuristic
    bump = np.exp(-0.5 * ((logf - logf0) / max(sigma, 1e-6)) ** 2)

    # blend between 1 and gain
    mag = 1.0 + (gain - 1.0) * bump
    return mag.astype(np.float64)


# =========================
# Minimum-phase reconstruction (cepstrum method)
# Reference idea matches your TF version:
# log(mag) -> irfft -> lifter -> rfft -> phase -> complex spectrum
# :contentReference[oaicite:1]{index=1}
# =========================

def minimum_phase_from_mag(mag: np.ndarray, n_fft: int) -> np.ndarray:
    """
    mag: [N_BINS] linear magnitude for rfft bins of n_fft
    returns complex rfft spectrum [N_BINS] with minimum phase
    """
    mag = np.asarray(mag, dtype=np.float64)
    mag = np.clip(mag, 1e-20, 1e20)

    log_mag = np.log(mag)
    # real cepstrum via irfft of log magnitude (real)
    cep = np.fft.irfft(log_mag, n=n_fft)  # length n_fft

    # minimum-phase lifter:
    # keep c[0], double c[1..N/2-1], keep c[N/2] if even, zero the rest
    cep2 = np.zeros_like(cep)
    cep2[0] = cep[0]
    half = n_fft // 2
    if n_fft % 2 == 0:
        cep2[1:half] = 2.0 * cep[1:half]
        cep2[half] = cep[half]
    else:
        cep2[1:half+1] = 2.0 * cep[1:half+1]

    mp_log_spec = np.fft.rfft(cep2, n=n_fft)
    phase = np.imag(mp_log_spec)

    real = mag * np.cos(phase)
    imag = mag * np.sin(phase)
    return real + 1j * imag


def _center_peak_to_index(ir: np.ndarray, target_idx: int) -> np.ndarray:
    ir = np.asarray(ir, dtype=np.float64)
    T = ir.shape[0]
    target_idx = int(np.clip(target_idx, 0, T - 1))
    peak_idx = int(np.argmax(np.abs(ir)))
    shift = target_idx - peak_idx
    return np.roll(ir, shift)


# =========================
# Forward simulation
# =========================

def simulate(params: SimParams) -> Dict[str, Any]:
    """
    Returns:
      freq: [n_points]
      spl_db: [n_points]
      ir: [ir_len]
      t: [ir_len] seconds
      (also returns mag_full / freq_bins if you need)
    """
    # display freq axis (log)
    freq_disp = _freq_axis_log(params.f_min, params.f_max, params.n_points)

    # build a full-band magnitude on rfft bins for IR reconstruction
    n_fft = int(params.ir_len)
    fs = float(params.fs)
    f_bins = _rfft_freq_bins(n_fft, fs)
    f_bins = np.maximum(f_bins, 1e-6)

    # Magnitude model on bins
    mag_bins = np.ones_like(f_bins, dtype=np.float64)
    mag_bins *= _low_shelf_mag(f_bins, params.bass_gain_db, f0=120.0, slope=1.0)
    mag_bins *= _high_shelf_mag(f_bins, params.treble_gain_db, f0=8000.0, slope=1.0)
    mag_bins *= _peaking_mag(f_bins, params.resonance_hz, params.resonance_q, params.resonance_gain_db)

    # Use same model on display freqs (for SPL curve)
    mag_disp = np.ones_like(freq_disp, dtype=np.float64)
    mag_disp *= _low_shelf_mag(freq_disp, params.bass_gain_db, f0=120.0, slope=1.0)
    mag_disp *= _high_shelf_mag(freq_disp, params.treble_gain_db, f0=8000.0, slope=1.0)
    mag_disp *= _peaking_mag(freq_disp, params.resonance_hz, params.resonance_q, params.resonance_gain_db)
    spl_db = _lin_to_db(mag_disp)

    # minimum-phase spectrum -> irfft
    spec = minimum_phase_from_mag(mag_bins, n_fft=n_fft)
    ir = np.fft.irfft(spec, n=n_fft).astype(np.float64)

    # normalize and center peak
    mx = np.max(np.abs(ir)) + 1e-9
    ir = ir / mx
    ir = _center_peak_to_index(ir, params.target_peak_index)

    t = (np.arange(n_fft, dtype=np.float64) / fs).astype(np.float64)

    return {
        "freq": freq_disp.astype(np.float64),
        "spl_db": spl_db.astype(np.float64),
        "ir": ir.astype(np.float64),
        "t": t,
        "meta": {
            "fs": fs,
            "ir_len": n_fft,
            "target_peak_index": int(params.target_peak_index),
        }
    }


# =========================
# Random curve generator
# =========================

def random_params(
    base: Optional[SimParams] = None,
    seed: Optional[int] = None
) -> SimParams:
    rng = np.random.default_rng(seed)
    p = SimParams(**asdict(base)) if base is not None else SimParams()

    # randomize within bounds
    p.bass_gain_db = float(rng.uniform(*PARAM_BOUNDS["bass_gain_db"]))
    p.treble_gain_db = float(rng.uniform(*PARAM_BOUNDS["treble_gain_db"]))
    p.resonance_hz = float(rng.uniform(*PARAM_BOUNDS["resonance_hz"]))
    p.resonance_q = float(rng.uniform(*PARAM_BOUNDS["resonance_q"]))
    p.resonance_gain_db = float(rng.uniform(*PARAM_BOUNDS["resonance_gain_db"]))
    return p


# =========================
# Fitting (gradient descent)
# Target: fit SPL curve (dB) on a given frequency axis
# =========================

def _pack_fit_params(p: SimParams) -> np.ndarray:
    return np.array([
        p.bass_gain_db,
        p.treble_gain_db,
        p.resonance_hz,
        p.resonance_q,
        p.resonance_gain_db
    ], dtype=np.float64)


def _unpack_fit_params(p: SimParams, x: np.ndarray) -> SimParams:
    q = SimParams(**asdict(p))
    q.bass_gain_db = float(x[0])
    q.treble_gain_db = float(x[1])
    q.resonance_hz = float(x[2])
    q.resonance_q = float(x[3])
    q.resonance_gain_db = float(x[4])
    return q


def _clip_fit_params(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    keys = ["bass_gain_db", "treble_gain_db", "resonance_hz", "resonance_q", "resonance_gain_db"]
    for i, k in enumerate(keys):
        lo, hi = PARAM_BOUNDS[k]
        x[i] = np.clip(x[i], lo, hi)
    return x


def fit_spl_gradient_descent(
    target_freq: np.ndarray,
    target_spl_db: np.ndarray,
    init_params: SimParams,
    steps: int = 200,
    lr: float = 0.05,
    use_torch: bool = True,
    finite_diff_eps: float = 1e-3,
) -> Dict[str, Any]:
    """
    Fit only SPL curve (on target_freq) by adjusting 5 parameters.

    Returns dict:
      best_params: SimParams
      history: list of {step, loss, params...}
    """
    target_freq = np.asarray(target_freq, dtype=np.float64)
    target_spl_db = np.asarray(target_spl_db, dtype=np.float64)
    assert target_freq.shape == target_spl_db.shape, "target_freq and target_spl_db must have same shape"

    # Fix display axis to target axis during fitting (so loss is aligned)
    base = SimParams(**asdict(init_params))
    base.f_min = float(np.min(target_freq))
    base.f_max = float(np.max(target_freq))
    base.n_points = int(target_freq.size)

    x = _pack_fit_params(base)
    x = _clip_fit_params(x)

    history: List[Dict[str, Any]] = []
    best_x = x.copy()
    best_loss = float("inf")

    if use_torch and _TORCH_OK:
        # Torch autograd (recommended)
        device = torch.device("cpu")
        xt = torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)

        tfreq = torch.tensor(target_freq, dtype=torch.float64, device=device)
        tspl = torch.tensor(target_spl_db, dtype=torch.float64, device=device)

        def forward_spl_db_torch(xx: torch.Tensor) -> torch.Tensor:
            bass, treble, f0, qv, g = xx
            # build model on tfreq
            # low shelf
            def low_shelf(f, gain_db, f0=120.0, slope=1.0):
                gg = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), gain_db / 20.0)
                x_ = torch.pow(f / f0, slope)
                return (gg + x_) / (1.0 + x_)

            def high_shelf(f, gain_db, f0=8000.0, slope=1.0):
                gg = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), gain_db / 20.0)
                x_ = torch.pow(f / f0, slope)
                return (1.0 + gg * x_) / (1.0 + x_)

            def peaking(f, f0, q, gain_db):
                f0 = torch.clamp(f0, 1e-6, 1e9)
                q = torch.clamp(q, 1e-3, 100.0)
                gg = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), gain_db / 20.0)
                logf = torch.log(f + 1e-30)
                logf0 = torch.log(f0)
                sigma = 0.35 / q
                sigma = torch.clamp(sigma, 1e-6, 10.0)
                bump = torch.exp(-0.5 * ((logf - logf0) / sigma) ** 2)
                return 1.0 + (gg - 1.0) * bump

            mag = torch.ones_like(tfreq)
            mag = mag * low_shelf(tfreq, bass)
            mag = mag * high_shelf(tfreq, treble)
            mag = mag * peaking(tfreq, f0, qv, g)
            mag = torch.clamp(mag, 1e-20, 1e20)
            spl = 20.0 * torch.log10(mag)
            return spl

        for step in range(int(steps)):
            xt.grad = None
            pred = forward_spl_db_torch(xt)
            loss = torch.mean((pred - tspl) ** 2)
            loss_val = float(loss.detach().cpu().numpy())

            loss.backward()

            with torch.no_grad():
                xt -= float(lr) * xt.grad
                # clamp to bounds
                x_np = xt.detach().cpu().numpy()
                x_np = _clip_fit_params(x_np)
                xt[:] = torch.tensor(x_np, dtype=torch.float64, device=device)

            rec = {
                "step": step,
                "loss": loss_val,
                "bass_gain_db": float(xt[0].detach().cpu().numpy()),
                "treble_gain_db": float(xt[1].detach().cpu().numpy()),
                "resonance_hz": float(xt[2].detach().cpu().numpy()),
                "resonance_q": float(xt[3].detach().cpu().numpy()),
                "resonance_gain_db": float(xt[4].detach().cpu().numpy()),
            }
            history.append(rec)

            if loss_val < best_loss:
                best_loss = loss_val
                best_x = np.array([rec["bass_gain_db"], rec["treble_gain_db"], rec["resonance_hz"], rec["resonance_q"], rec["resonance_gain_db"]], dtype=np.float64)

    else:
        # Finite-difference gradient descent (slower)
        def loss_fn(xx: np.ndarray) -> float:
            pp = _unpack_fit_params(base, xx)
            # evaluate SPL on target_freq
            mag = np.ones_like(target_freq, dtype=np.float64)
            mag *= _low_shelf_mag(target_freq, pp.bass_gain_db, f0=120.0, slope=1.0)
            mag *= _high_shelf_mag(target_freq, pp.treble_gain_db, f0=8000.0, slope=1.0)
            mag *= _peaking_mag(target_freq, pp.resonance_hz, pp.resonance_q, pp.resonance_gain_db)
            pred = _lin_to_db(mag)
            return float(np.mean((pred - target_spl_db) ** 2))

        for step in range(int(steps)):
            base_loss = loss_fn(x)
            grad = np.zeros_like(x)

            for i in range(x.size):
                dx = np.zeros_like(x)
                dx[i] = finite_diff_eps
                l1 = loss_fn(_clip_fit_params(x + dx))
                l2 = loss_fn(_clip_fit_params(x - dx))
                grad[i] = (l1 - l2) / (2.0 * finite_diff_eps)

            x = x - float(lr) * grad
            x = _clip_fit_params(x)
            loss_val = loss_fn(x)

            rec = {
                "step": step,
                "loss": float(loss_val),
                "bass_gain_db": float(x[0]),
                "treble_gain_db": float(x[1]),
                "resonance_hz": float(x[2]),
                "resonance_q": float(x[3]),
                "resonance_gain_db": float(x[4]),
            }
            history.append(rec)

            if loss_val < best_loss:
                best_loss = float(loss_val)
                best_x = x.copy()

    best_params = _unpack_fit_params(base, best_x)
    return {
        "best_params": asdict(best_params),
        "best_loss": float(best_loss),
        "history": history,
    }
