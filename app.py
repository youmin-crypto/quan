import numpy as np
import matplotlib.pyplot as plt

def grover_step(amplitudes, correct_index):
    """Perform one Grover iteration: oracle + diffusion"""
    N = len(amplitudes)
    
    # --- Oracle: flip phase of correct state ---
    amplitudes[correct_index] *= -1
    
    # --- Diffusion: inversion about mean ---
    mean = amplitudes.mean()
    amplitudes = 2*mean - amplitudes
    return amplitudes

def simulate_grover(n_qubits, correct_index):
    N = 2 ** n_qubits
    amplitudes = np.ones(N) / np.sqrt(N)  # initial equal superposition
    
    history = [amplitudes.copy()]

    # Optimal number of iterations ≈ pi/4 * sqrt(N)
    n_iterations = int(np.round(np.pi/4 * np.sqrt(N)))

    print(f"Simulating Grover with {n_qubits} qubits, correct state: {correct_index}")
    print(f"Initial amplitudes: {amplitudes}")
    print(f"Performing {n_iterations} iterations...\n")
    
    for i in range(n_iterations):
        amplitudes = grover_step(amplitudes, correct_index)
        history.append(amplitudes.copy())
        probs = np.round(amplitudes**2, 3)
        print(f"Iteration {i+1}: Probabilities: {probs}")
    
    return history

def plot_history(history):
    history = np.array(history)
    iterations = history.shape[0]
    N = history.shape[1]
    
    plt.figure(figsize=(10,5))
    for state in range(N):
        plt.plot(range(iterations), history[:, state]**2, marker='o', label=f"|{state:0{int(np.log2(N))}b}>")
    plt.xlabel("Grover Iteration")
    plt.ylabel("Probability")
    plt.title("Grover Algorithm Amplitude Evolution")
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Parameters ---
n_qubits = 2        # Number of qubits (states = 2^n_qubits)
correct_index = 2   # Index of correct answer (binary 10)

# --- Simulation ---
history = simulate_grover(n_qubits, correct_index)

# --- Plot ---
plot_history(history)
