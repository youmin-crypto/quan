# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Qubit Superposition Visualization")

st.write("""
ဒီနေရာမှာ **qubit တိုးတိုင်း state doubling** ကို မြင်နိုင်ပါတယ်။  
- Bar = basis state (bit pattern)  
- Height = probability
""")

# --- Interactive slider ---
n_qubits = st.slider("Number of Qubits", min_value=1, max_value=5, value=2)

# --- Compute superposition ---
N = 2 ** n_qubits
amplitudes = np.ones(N) / np.sqrt(N)  # equal superposition
probabilities = amplitudes**2
labels = [format(i, f'0{n_qubits}b') for i in range(N)]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, probabilities, color='skyblue')
ax.set_xlabel("Qubit basis states (bit pattern)")
ax.set_ylabel("Probability")
ax.set_title(f"Superposition of {n_qubits} Qubits")
ax.set_ylim(0, 1)
st.pyplot(fig)

st.write(f"Number of basis states: {N} = 2^{n_qubits}")
st.write("As you increase qubits, the number of bit patterns doubles (2ⁿ growth).")
