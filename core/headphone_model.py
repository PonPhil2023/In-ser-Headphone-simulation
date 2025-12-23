import numpy as np

def logspace_freqs(f_min: float, f_max: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(f_min), np.log10(f_max), n)

def peaking_eq_db(f: np.ndarray, f0: float, q: float, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return np.zeros_like(f)
    x = np.log2(f / f0)
    sigma = 1.0 / max(q, 1e-6)
    return gain_db * np.exp(-0.5 * (x / sigma) ** 2)

def low_shelf_db(f: np.ndarray, f0: float, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return np.zeros_like(f)
    x = np.log10(f / f0)
    return gain_db * (1.0 / (1.0 + np.exp(6.0 * x)))

def high_shelf_db(f: np.ndarray, f0: float, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return np.zeros_like(f)
    x = np.log10(f / f0)
    return gain_db * (1.0 / (1.0 + np.exp(-6.0 * x)))

def simulate_fr(
    f_min: float = 20.0,
    f_max: float = 20000.0,
    n_points: int = 400,
    bass_gain_db: float = 0.0,
    treble_gain_db: float = 0.0,
    resonance_hz: float = 3000.0,
    resonance_q: float = 1.0,
    resonance_gain_db: float = 0.0,
):
    f = logspace_freqs(f_min, f_max, n_points)
    spl = np.zeros_like(f)
    spl += low_shelf_db(f, 120.0, bass_gain_db)
    spl += high_shelf_db(f, 6000.0, treble_gain_db)
    spl += peaking_eq_db(f, resonance_hz, resonance_q, resonance_gain_db)
    return f, spl
