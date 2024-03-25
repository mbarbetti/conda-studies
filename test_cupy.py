# --- CuPy ---

import cupy as cp

print(f"[DEBUG] CuPy version {cp.__version__}")

n_cuda_devices = cp.cuda.runtime.getDeviceCount()

if n_cuda_devices > 0:
    x = cp.array([1, 2, 3])
    x.device
    print("[STATUS] GPU available")
else:
    print("[STATUS] GPU not available")