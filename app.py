import numpy as np
import matplotlib.pyplot as plt

def simulate_qubits(n_qubits):
    N = 2 ** n_qubits
    # Equal superposition: amplitude 1/sqrt(N)
    amplitudes = np.ones(N) / np.sqrt(N)
    # Probability
    probabilities = amplitudes**2
    # Labels: bit patterns
    labels = [format(i, f'0{n_qubits}b') for i in range(N)]
    return labels, probabilities

# --- Parameters ---
n_qubits = 3  # Change 1,2,3,... to see effect

labels, probabilities = simulate_qubits(n_qubits)

# --- Plot ---
plt.figure(figsize=(8,4))
plt.bar(labels, probabilities, color='skyblue')
plt.xlabel("Qubit basis states (bit pattern)")
plt.ylabel("Probability")
plt.title(f"Superposition of {n_qubits} Qubits")
plt.ylim(0, 1)
plt.show()
