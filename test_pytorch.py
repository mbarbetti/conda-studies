# --- PyTorch ---

import torch

print(f"[DEBUG] PyTorch version {torch.__version__}")

torch.cuda.init()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rnd = torch.randn(size=(100, 1)).to(device)
    print("[STATUS] GPU available")
else:
    print("[STATUS] GPU not available")
