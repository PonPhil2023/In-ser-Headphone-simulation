from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

from backend.app.core.fitter import fit_params_gd
from backend.app.core.physics_forward import forward_ir_and_mag

router = APIRouter()

# 這些 normalizer 你在 STEP1 有存 npz：normalizer_ir.npz :contentReference[oaicite:4]{index=4}
NORMALIZER = np.load("normalizer_ir.npz")
Xmin = tf.constant(NORMALIZER["Xmin"].astype(np.float32))
Xmax = tf.constant(NORMALIZER["Xmax"].astype(np.float32))

IR_LEN = 131072
FS = 48000

def downsample(x: np.ndarray, n: int):
    if len(x) <= n:
        return x
    idx = np.linspace(0, len(x)-1, n).astype(int)
    return x[idx]

class SimReq(BaseModel):
    params_norm: list[float]

class FitReq(BaseModel):
    target_spl_db: list[float]
    target_ir: list[float] | None = None
    init_params_norm: list[float] | None = None
    steps: int = 300
    lr: float = 0.02
    w_spl: float = 1.0
    w_ir: float = 0.0

@router.post("/simulate_physics")
def simulate_physics(req: SimReq):
    p = np.array(req.params_norm, dtype=np.float32)[None, :]  # [1,9]
    ir, mag = forward_ir_and_mag(tf.constant(p), Xmin, Xmax, tf.constant(IR_LEN, tf.float32), tf.constant(FS, tf.float32))
    ir = ir.numpy()[0].astype(np.float32)
    mag = mag.numpy()[0].astype(np.float32)
    spl_db = (20.0*np.log10(mag + 1e-8)).astype(np.float32)

    # 頻率軸（rfft bins）
    freqs = np.linspace(0, FS/2, mag.shape[0]).astype(np.float32)

    return {
        "freqs": downsample(freqs, 1024).tolist(),
        "spl_db": downsample(spl_db, 1024).tolist(),
        "ir": downsample(ir, 4096).tolist(),
    }

@router.post("/random_target")
def random_target():
    # random in [0,1]
    p = np.random.rand(9).astype(np.float32)
    out = simulate_physics(SimReq(params_norm=p.tolist()))
    out["target_params_norm"] = p.tolist()
    return out

@router.post("/fit_gd")
def fit_gd(req: FitReq):
    target_spl = np.array(req.target_spl_db, dtype=np.float32)
    target_ir = np.array(req.target_ir, dtype=np.float32) if req.target_ir is not None else None

    init = np.array(req.init_params_norm, dtype=np.float32) if req.init_params_norm is not None else np.random.rand(9).astype(np.float32)

    def forward_fn(p_norm):
        ir, mag = forward_ir_and_mag(p_norm, Xmin, Xmax, tf.constant(IR_LEN, tf.float32), tf.constant(FS, tf.float32))
        return ir, mag

    result = fit_params_gd(
        forward_fn=forward_fn,
        target_spl_db=target_spl,
        target_ir=target_ir,
        init_params_norm=init,
        steps=req.steps,
        lr=req.lr,
        w_spl=req.w_spl,
        w_ir=req.w_ir,
    )

    # 回傳時下採樣
    freqs = np.linspace(0, FS/2, len(result["best_spl_db"])).astype(np.float32)

    return {
        "best_params_norm": result["best_params_norm"].tolist(),
        "best_loss": result["best_loss"],
        "freqs": downsample(freqs, 1024).tolist(),
        "spl_db": downsample(result["best_spl_db"], 1024).tolist(),
        "ir": downsample(result["best_ir"], 4096).tolist(),
    }
