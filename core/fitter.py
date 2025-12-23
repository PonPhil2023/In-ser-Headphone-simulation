import numpy as np
import tensorflow as tf

EPS = 1e-8

def to_f32(x): 
    return tf.convert_to_tensor(x, dtype=tf.float32)

def spl_from_mag(mag: tf.Tensor) -> tf.Tensor:
    # mag: [B, N_BINS]
    return 20.0 * tf.math.log(mag + EPS) / tf.math.log(10.0)

@tf.function
def clip01(x):
    return tf.clip_by_value(x, 0.0, 1.0)

def fit_params_gd(
    forward_fn,              # (params_norm) -> (ir[B,T], mag[B,N])
    target_spl_db: np.ndarray,
    target_ir: np.ndarray | None,
    init_params_norm: np.ndarray,
    steps: int = 300,
    lr: float = 1e-2,
    w_spl: float = 1.0,
    w_ir: float = 0.0,
):
    # target
    t_spl = to_f32(target_spl_db)[None, :]   # [1, N]
    t_ir  = to_f32(target_ir)[None, :] if target_ir is not None else None

    # variable
    p = tf.Variable(to_f32(init_params_norm)[None, :])  # [1, 9]
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    best_loss = np.inf
    best_p = init_params_norm.copy()
    best_ir = None
    best_spl = None

    for k in range(steps):
        with tf.GradientTape() as tape:
            ir, mag = forward_fn(clip01(p))       # ir [1,T], mag [1,N]
            spl_db = spl_from_mag(mag)[0]         # [N]
            loss_spl = tf.reduce_mean((spl_db - t_spl[0])**2)

            loss = w_spl * loss_spl

            if (t_ir is not None) and (w_ir > 0.0):
                loss_ir = tf.reduce_mean((ir[0] - t_ir[0])**2)
                loss = loss + w_ir * loss_ir

        grads = tape.gradient(loss, p)
        opt.apply_gradients([(grads, p)])

        # keep inside [0,1]
        p.assign(clip01(p))

        if (k % 10) == 0 or k == steps - 1:
            l = float(loss.numpy())
            if l < best_loss:
                best_loss = l
                best_p = p.numpy()[0].astype(np.float32)
                best_ir = ir.numpy()[0].astype(np.float32)
                best_spl = spl_db.numpy().astype(np.float32)

    return {
        "best_params_norm": best_p,
        "best_loss": float(best_loss),
        "best_spl_db": best_spl,
        "best_ir": best_ir,
    }
