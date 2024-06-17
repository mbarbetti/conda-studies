# --- PennyLane ---

import pennylane as qml

print(f"[DEBUG] PennyLane version {qml.__version__}")

dev = qml.device("lightning.gpu", wires=2)

@qml.qnode(dev)
def circuit():
  qml.Hadamard(wires=0)
  qml.CNOT(wires=[0,1])
  return qml.expval(qml.PauliZ(0))

try:
    print("[DEBUG] Circuit:", circuit())
    # Circuit: 0.0
    print("[STATUS] GPU available")
except:
    print("[STATUS] GPU not available")
