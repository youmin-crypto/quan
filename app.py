import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Grover Algorithm Single Calculation Example")

st.write("""
- Slider ဖြင့် **number of qubits** ကိုရွေးပါ  
- Input → correct state ကိုသတ်မှတ်ပါ  
- Button → **Run Simulation** → probability calculation
""")

# --- Parameters ---
n_qubits = st.slider("Number of Qubits", min_value=1, max_value=5, value=2)
correct_index = st.number_input(f"Correct state (0..{2**n_qubits -1})", min_value=0, max_value=2**n_qubits -1, value=2)
run_button = st.button("Run Grover Calculation")

# --- Grover Functions ---
def grover_step(amplitudes, correct_index):
    amplitudes[correct_index] *= -1
    mean = amplitudes.mean()
    amplitudes = 2*mean - amplitudes
    return amplitudes

def simulate_grover(n_qubits, correct_index):
    N = 2 ** n_qubits
    amplitudes = np.ones(N) / np.sqrt(N)
    n_iterations = int(round(np.pi/4 * np.sqrt(N)))
    for _ in range(n_iterations):
        amplitudes = grover_step(amplitudes, correct_index)
    probabilities = amplitudes**2
    return probabilities

# --- Run Simulation on Button Click ---
if run_button:
    probabilities = simulate_grover(n_qubits, correct_index)
    labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    colors = ['red' if idx == correct_index else 'skyblue' for idx in range(2**n_qubits)]

    st.write(f"Simulation result for {n_qubits} qubits, correct state = {correct_index}")
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0,1)
    ax.set_xlabel("Qubit basis states (bit pattern)")
    ax.set_ylabel("Probability")
    ax.set_title("Grover Algorithm - Final Probabilities")
    st.pyplot(fig)

    st.write("Probability of correct state:", probabilities[correct_index])
