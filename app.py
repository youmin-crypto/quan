# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Grover Algorithm Simulation (Toy Example)")

# --- Parameters ---
n_qubits = st.slider("Number of Qubits", min_value=2, max_value=5, value=2)
correct_index = st.number_input("Correct state (0..{})".format(2**n_qubits - 1), min_value=0, max_value=2**n_qubits-1, value=2)

# --- Grover Simulation Functions ---
def grover_step(amplitudes, correct_index):
    N = len(amplitudes)
    amplitudes[correct_index] *= -1          # Oracle
    mean = amplitudes.mean()                 # Diffusion
    amplitudes = 2*mean - amplitudes
    return amplitudes

def simulate_grover(n_qubits, correct_index):
    N = 2 ** n_qubits
    amplitudes = np.ones(N) / np.sqrt(N)
    history = [amplitudes.copy()]
    n_iterations = int(round(np.pi/4 * np.sqrt(N)))
    for _ in range(n_iterations):
        amplitudes = grover_step(amplitudes, correct_index)
        history.append(amplitudes.copy())
    return history

def plot_history(history):
    history = np.array(history)
    iterations = history.shape[0]
    N = history.shape[1]
    fig, ax = plt.subplots(figsize=(8,5))
    for state in range(N):
        ax.plot(range(iterations), history[:,state]**2, marker='o', label=f"|{state:0{int(np.log2(N))}b}>")
    ax.set_xlabel("Grover Iteration")
    ax.set_ylabel("Probability")
    ax.set_title("Amplitude Evolution")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Run Simulation ---
history = simulate_grover(n_qubits, correct_index)
plot_history(history)
