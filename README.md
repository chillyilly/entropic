# Entropic

This project is a hybrid entropy benchmarking and analysis suite that performs side-by-side evaluation of cryptographic randomness and time-based entropy, enhanced by quantum-inspired metrics. It collects and evaluates random data over a user-defined time window and applies both statistical and quantum-theoretical analysis to assess the quality of the generated entropy.

---
## Installation
```bash
git clone https://github.com/chillyilly/entropic.git
cd entropic
```
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
```bash
pip3 install numpy requests mpmath
```
## To Run:
```bash
python3 entropic.py
```
## How It Works

### 1. Real-Time Sampling
Each second:
- `epoch_time`: Current Unix timestamp in seconds.
- `real_time_str`: Human-readable timestamp.
- `delta_time`: Difference between epoch and formatted time (adds entropy).

### 2. RNG Generation
- **Crypto RNG**: From `secrets.randbits(64)`
- **Time RNG**: SHA256 of time string + epoch time
- **Optional QRNG**: Single seed from ANU QRNG, extrapolated

### 3. Quantum Metrics
For each second, we compute several custom metrics inspired by quantum mechanics and chaos theory.

---

## Quantum-Entanglement Inspired Equations

### **Von Neumann Entropy**
Entropy of a quantum state with two probabilities \( p_1 \) and \( p_2 \):

$$
S = -\sum_i \lambda_i \log_2(\lambda_i)
$$

Where \( \lambda_i \) are eigenvalues of the 2x2 density matrix \( \rho \):

$$
\rho = \begin{bmatrix}
p_1 & 0 \\
0 & p_2
\end{bmatrix}
$$

---

### **Amplitude Components**
Given \( p_1 + p_2 = 1 \), amplitudes \( \alpha \) and \( \beta \) are:

$$
\alpha = \sqrt{p_1}, \quad \beta = \sqrt{p_2}
$$

---

### **Concurrence**
Quantum measure of entanglement:

$$
C = 2 \alpha \beta
$$

---

### **Ξ – Original Entanglomorphic Index**

Amplification of entanglement waveforms:

$$
\Xi = (\alpha^\beta + \beta^\alpha) \cdot \left[\sin(\pi C) + \cos(\pi S)\right]^2
$$

---

### **Ψ – Chaos-Extended Index**

Chaos-injected amplification:

---

### **Ψ₁ – Fractal-Driven Entropy Bloom**



Where \( A = \alpha^\beta + \beta^\alpha \)

---

### **Ψ₂ – Entropic Möbius Warp**


---

### **Ψ₃ – Recursive Entanglomorphic Lens**

$$
\Psi_3 = A \cdot \left[ \log_2(1 + \Psi_{\text{prev}} \cdot \sin(\pi C)) + \cos(\pi S)^{1 + (\Psi_{\text{prev}} \bmod 3)} \right]
$$

---

## Statistical Tests

These tests are applied to both RNG streams:

| Test                    | Description |
|-------------------------|-------------|
| Shannon Entropy         | Bit distribution entropy |
| Min-Entropy             | Max single-value probability |
| Monobit Test            | 0s vs 1s balance |
| Chi-Square Test         | Byte distribution uniformity |
| Serial Correlation      | Self-similarity in sequence |
| G-Test                  | Likelihood-ratio frequency test |
| Runs Test               | Measures non-random clusters |
| GRIP Test               | Dot product variance in modular vectors |
| Cross-Entropy Benchmark | Entropy vs ideal model divergence |

---

## Use Cases

- Quantum simulation benchmarking  
- Seeding public entropy pools  
- AI prompt entropy injection  
- Cryptographic entropy validation  
- Red team entropy evaluation  
## License

MIT License. See `LICENSE` file for usage terms.
