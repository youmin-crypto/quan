# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactive Qubit + Grover Visualization")
st.write("""
- Slider ဖြင့် **number of qubits** ကို ပြောင်းနိုင်သည်  
- Graph တွင် **bit patterns** ကို ပြသပြီး  
  **bar height = probability**  
- Grover iteration လုပ်ပြီး **amplitude interference** ကို simulation
""")

# --- Parameters ---
n_qubits = st.slider("Number of Qubits", min_value=1, max_value=5, value=2)
correct_index = st.number_input(f"Correct state (0..{2**n_qubits - 1})", min_value=0, max_value=2**n_qubits -1, value=2)

# --- Grover Functions ---
def grover_step(amplitudes, correct_index):
    N = len(amplitudes)
    amplitudes[correct_index] *= -1  # Oracle
    mean = amplitudes.mean()         # Diffusion
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

# --- Plot Bar Graph for Last Iteration ---
final_amplitudes = history[-1]
probabilities = final_amplitudes**2
labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, probabilities, color='skyblue')
ax.set_xlabel("Qubit basis states (bit pattern)")
ax.set_ylabel("Probability")
ax.set_title(f"Final probabilities after Grover iteration for {n_qubits} qubits")
ax.set_ylim(0, 1)
st.pyplot(fig)

# --- Probability Evolution Plot ---
st.write("Probability Evolution per Iteration:")
history_array = np.array(history)
fig2, ax2 = plt.subplots(figsize=(8,5))
for state in range(2**n_qubits):
    ax2.plot(range(len(history)), history_array[:,state]**2, marker='o', label=f"|{labels[state]}>")
ax2.set_xlabel("Grover Iteration")
ax2.set_ylabel("Probability")
ax2.set_title("Amplitude Evolution During Grover Search")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.write(f"Number of basis states: {2**n_qubits}")
st.write("As you increase qubits, the number of 01 patterns doubles (2ⁿ growth).")
