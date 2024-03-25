# --- Jax ---

import jax

print(f"[DEBUG] Jax version {jax.__version__}")

def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

if jax_has_gpu():
    print("[STATUS] GPU available")
else:
    print("[STATUS] GPU not available")
