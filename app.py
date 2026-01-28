import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("Grover Algorithm Animated Visualization")

st.write("""
- Slider ဖြင့် **number of qubits** ကို ပြောင်းနိုင်သည်  
- Correct state ကို select → Grover algorithm amplitude amplification  
- Animation → iteration တိုးတိုင်း probability bar auto update  
""")

# --- Parameters ---
n_qubits = st.slider("Number of Qubits", min_value=1, max_value=5, value=2)
correct_index = st.number_input(f"Correct state (0..{2**n_qubits -1})", min_value=0, max_value=2**n_qubits -1, value=2)
delay = st.slider("Animation delay (seconds per iteration)", min_value=0.1, max_value=1.0, value=0.3)

# --- Grover Functions ---
def grover_step(amplitudes, correct_index):
    amplitudes[correct_index] *= -1  # Oracle
    mean = amplitudes.mean()          # Diffusion
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

# --- Run Simulation ---
history = simulate_grover(n_qubits, correct_index)
labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]

# --- Animated Plot ---
st.write("Grover Iteration Animation:")
placeholder = st.empty()  # Placeholder for matplotlib

for i, amplitudes in enumerate(history):
    probabilities = amplitudes**2
    colors = ['red' if idx == correct_index else 'skyblue' for idx in range(2**n_qubits)]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0,1)
    ax.set_xlabel("Qubit basis states (bit pattern)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Iteration {i}/{len(history)-1}")
    
    placeholder.pyplot(fig)
    time.sleep(delay)  # animation delay
