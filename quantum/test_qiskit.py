# --- Qiskit ---

import qiskit
from qiskit_aer import AerSimulator

print(f"[DEBUG] Qiskit version {qiskit.__version__}")

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Construct an ideal simulator
aersim = AerSimulator()

# Perform an ideal simulation
try:
    result_ideal = aersim.run(circ).result()
    counts_ideal = result_ideal.get_counts(0)
    print('[DEBUG] Counts(ideal):', counts_ideal)
    # Counts(ideal): {'000': 493, '111': 531}
    print("[STATUS] GPU available")
except:
    print("[STATUS] GPU not available")
