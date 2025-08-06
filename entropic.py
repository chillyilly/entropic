import time
import secrets
import hashlib
import requests
import mpmath
import numpy as np
from collections import Counter
from math import log2, sin, cos, pi, tan, e, sqrt
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# --- RNG Implementations ---

def generate_time_based_random(epoch_time, real_time_str):
    combined = f"{epoch_time}-{real_time_str}"
    hash_object = hashlib.sha256(combined.encode())
    hash_hex = hash_object.hexdigest()
    return int(hash_hex[:12], 16)

def generate_entangled_random():
    t = int(time.time() * 1e9)
    jitter = time.perf_counter_ns()
    secret = secrets.randbits(64)
    combined = t ^ jitter ^ secret
    hash_object = hashlib.sha256(str(combined).encode())
    return int(hash_object.hexdigest()[:12], 16)

def get_qrng():
    try:
        r = requests.get("https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16", timeout=2)
        if r.ok:
            return r.json()['data'][0]
    except Exception:
        pass
    return None

# --- Entanglement + Chaos Metrics ---

def quantum_metrics(epoch_val, delta_val, crypto_rand, time_rand, prev_psi):
    total = epoch_val + delta_val
    if total == 0:
        total = 1e-9

    p1 = epoch_val / total
    p2 = delta_val / total

    rho = np.array([[p1, 0], [0, p2]])
    eigvals = np.linalg.eigvals(rho)
    von_entropy = -sum(p * log2(p) for p in eigvals if p > 0)

    alpha = np.sqrt(p1)
    beta = np.sqrt(p2)
    concurrence = 2 * alpha * beta

    amplification = (alpha ** beta) + (beta ** alpha)
    chaos_wave = (sin(pi * concurrence) + cos(pi * von_entropy)) ** 2
    xi = amplification * chaos_wave

    epoch_nano = int((epoch_val * 1e9) % 256)
    delta_nano = int((delta_val * 1e9) % 256)
    chaos_bitfield = (epoch_nano ^ delta_nano) ^ (crypto_rand ^ time_rand) & 0xFF
    chaos_factor = (chaos_bitfield % 9973) / 9973
    psi = xi * (1 + log2(1 + chaos_factor))

    epsilon = 1e-9
    bloom_exponent = 1 + (von_entropy * concurrence) / (log2(1 + chaos_factor) + epsilon)
    psi1 = amplification * ((sin(pi * concurrence) + cos(pi * von_entropy)) / sqrt(abs(alpha - beta) + epsilon)) ** bloom_exponent

    try:
        warp = (
            (sin(pi * (e ** concurrence / (pi ** von_entropy))) *
             cos(pi * (von_entropy ** 2 / (1 + chaos_factor)))) ** 2
        ) + tan(concurrence - von_entropy) ** 2
    except:
        warp = 0.0
    psi2 = amplification * warp

    try:
        lens = log2(1 + prev_psi * sin(pi * concurrence)) + cos(pi * von_entropy) ** (1 + (prev_psi % 3))
    except:
        lens = 0.0
    psi3 = amplification * lens

    return von_entropy, alpha, beta, concurrence, xi, psi, psi1, psi2, psi3

# --- Randomness Tests ---

def entropy(bits_list):
    total = len(bits_list)
    if total <= 1:
        return 0
    counts = Counter(bits_list)
    probs = [count / total for count in counts.values()]
    return -sum(p * log2(p) for p in probs)

def monobit_test(numbers):
    binary_string = ''.join(f'{num:064b}' for num in numbers)
    ones = binary_string.count('1')
    zeros = binary_string.count('0')
    return ones, zeros

def chi_square_test(numbers, bits=64):
    byte_counts = Counter()
    for num in numbers:
        for i in range(bits // 8):
            byte = (num >> (i * 8)) & 0xFF
            byte_counts[byte] += 1
    expected = len(numbers) * (bits // 8) / 256
    chi2 = sum((count - expected) ** 2 / expected for count in byte_counts.values())
    return chi2

def serial_correlation(data):
    n = len(data)
    if n < 2:
        return 0.0
    mean = sum(data) / n
    num = sum((data[i] - mean) * (data[i+1] - mean) for i in range(n - 1))
    den = sum((x - mean)**2 for x in data)
    return num / den if den else 0.0

def min_entropy(data):
    if not data:
        return 0.0
    counts = Counter(data)
    max_prob = max(counts.values()) / len(data)
    return -log2(max_prob)

def g_test(data):
    if not data:
        return 0.0
    counts = Counter(data)
    N = sum(counts.values())
    if len(counts) < 2:
        return 0.0
    expected = N / len(counts)
    g = 2 * sum(count * log2(count / expected) for count in counts.values() if count > 0)
    return g

def runs_test(data):
    median = np.median(data)
    signs = ['+' if x >= median else '-' for x in data]
    runs = 1 + sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
    return runs

def grip_test(data):
    vectors = [(x % 1000, (x // 1000) % 1000) for x in data]
    dot_products = [x * y for x, y in vectors]
    return np.mean(dot_products), np.std(dot_products)

def cross_entropy_score(data):
    probs = Counter(data)
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return -sum(p * log2(p) for p in probs.values() if p > 0)

# --- Main ---

def main():
    all_crypto = []
    all_time = []
    try:
        duration = int(input("Enter number of seconds to run (e.g., 10): "))
        if duration <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
        sys.exit(1)

    prev_psi = 0
    print("\nGenerating random numbers and quantum metrics...\n")

    for i in range(duration):
        epoch_time = time.time()
        real_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_time))
        real_time_sec = time.mktime(time.strptime(real_time_str, '%Y-%m-%d %H:%M:%S'))
        delta_time = abs(epoch_time - real_time_sec)

        crypto_rand = secrets.randbits(64)
        time_rand = generate_time_based_random(epoch_time, real_time_str)
        all_crypto.append(crypto_rand)
        all_time.append(time_rand)

        v_entropy, alpha, beta, concurrence, xi, psi, psi1, psi2, psi3 = quantum_metrics(
            epoch_time, delta_time, crypto_rand, time_rand, prev_psi
        )
        prev_psi = psi3

        print(f"[{real_time_str}] Epoch: {epoch_time:.6f}  Δ: {delta_time:.6f}")
        print(f"  Crypto RNG: {crypto_rand}")
        print(f"  Time RNG:   {time_rand}")
        print(f"  Von Neumann Entropy: {v_entropy:.6f}")
        print(f"  α: {alpha:.4f}, β: {beta:.4f}, Concurrence: {concurrence:.6f}")
        print(f"  ξ (Original Entanglomorphic Index): {xi:.6f}")
        print(f"  ψ (Chaos-Extended Index): {psi:.6f}")
        print(f"  ψ₁ (Fractal-Driven Entropy Bloom): {psi1:.6f}")
        print(f"  ψ₂ (Entropic Möbius Warp): {psi2:.6f}")
        print(f"  ψ₃ (Recursive Entanglomorphic Lens): {psi3:.6f}\n")

    time.sleep(1)

    print("\n--- Statistical Analysis ---")
    ones, zeros = monobit_test(all_crypto)
    print(f"Crypto RNG Monobit Test: {ones} ones, {zeros} zeros")
    ones, zeros = monobit_test(all_time)
    print(f"Time RNG Monobit Test:   {ones} ones, {zeros} zeros")

    print(f"Crypto RNG Entropy: {entropy(all_crypto):.6f}")
    print(f"Time RNG Entropy:   {entropy(all_time):.6f}")

    print(f"Crypto RNG Chi-Square: {chi_square_test(all_crypto):.2f}")
    print(f"Time RNG Chi-Square:   {chi_square_test(all_time):.2f}")

    print(f"Crypto RNG Serial Correlation: {serial_correlation(all_crypto):.6f}")
    print(f"Time RNG Serial Correlation:   {serial_correlation(all_time):.6f}")

    print(f"Crypto RNG Min-Entropy: {min_entropy(all_crypto):.6f}")
    print(f"Time RNG Min-Entropy:   {min_entropy(all_time):.6f}")

    print(f"Crypto RNG G-Test: {g_test(all_crypto):.2f}")
    print(f"Time RNG G-Test:   {g_test(all_time):.2f}")

    print(f"Crypto RNG Runs: {runs_test(all_crypto)}")
    print(f"Time RNG Runs:   {runs_test(all_time)}")

    mean_dot_c, std_dot_c = grip_test(all_crypto)
    mean_dot_t, std_dot_t = grip_test(all_time)
    print(f"Crypto RNG GRIP Mean Dot: {mean_dot_c:.2f}, Std: {std_dot_c:.2f}")
    print(f"Time RNG GRIP Mean Dot:   {mean_dot_t:.2f}, Std: {std_dot_t:.2f}")

    print(f"Crypto RNG Cross-Entropy: {cross_entropy_score(all_crypto):.6f}")
    print(f"Time RNG Cross-Entropy:   {cross_entropy_score(all_time):.6f}")

if __name__ == "__main__":
    main()
