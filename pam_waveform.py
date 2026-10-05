#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:06:50 2025

@author: pantelis
"""

# pam_waveform.py
# Python 3.12 | Spyder
# Μέρος Β: Bits -> Gray -> PAM σύμβολα -> κυματομορφή με τριγωνικούς παλμούς

import numpy as np
import matplotlib.pyplot as plt
from math import log2, ceil

# -----------------------------
# ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ
# -----------------------------
input_bits = '110101100011100010010011'   # μπορείς να αλλάξεις το string
M = 16                                    # Τάξη διαμόρφωσης (π.χ. 4, 16, 256). Πρέπει power-of-two.
N = 32                                     # Δείγματα ανά διάρκεια συμβόλου Ts (oversampling)
Rb = 1e9                                   # Ρυθμός μετάδοσης bit: 1 Gbit/s (απαίτηση εκφώνησης)
beta = 1.0                                 # Κλίμακα σταθμών (β=1 από την εκφώνηση)
show_samples = 1000                        # πόσα δείγματα να δείξει στο zoom plot (προαιρετικό)

# -----------------------------
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# -----------------------------
def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def binstr_to_int(b: str) -> int:
    return int(b, 2) if b else 0

def int_to_binstr(x: int, k: int) -> str:
    return format(x, f'0{k}b')

def binary_to_gray(b: int) -> int:
    return b ^ (b >> 1)

def gray_to_binary(g: int) -> int:
    # αναστροφή gray->binary (bitwise)
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def bits_to_symbols_gray(bits: str, M: int) -> np.ndarray:
    """
    Κόβει τα bits σε τμήματα μήκους k=log2(M).
    Κάθε τμήμα το ερμηνεύει ως Gray word -> το γυρίζει σε binary index m (0..M-1).
    Επιστρέφει τον πίνακα δεικτών m_k.
    """
    if not is_power_of_two(M):
        raise ValueError("M must be a power of two.")
    k = int(log2(M))
    # pad με μηδενικά αν δεν είναι πολλαπλάσιο του k
    if len(bits) % k != 0:
        pad = k - (len(bits) % k)
        bits = bits + '0' * pad

    m_list = []
    for i in range(0, len(bits), k):
        g_word = bits[i:i+k]
        g_val = binstr_to_int(g_word)
        m = gray_to_binary(g_val)  # index m στη φυσική (binary) διάταξη
        m_list.append(m)
    return np.array(m_list, dtype=int)

def pam_levels_from_indices(m_idx: np.ndarray, M: int, beta: float = 1.0) -> np.ndarray:
    """
    A_m = beta * (2m - M + 1), m=0..M-1  -> στάθμες συμμετρικές γύρω από το 0
    """
    return beta * (2*m_idx - (M - 1))

def triangular_pulse(N: int) -> np.ndarray:
    """
    Τριγωνικός παλμός μήκους N δειγμάτων, peak = 1 στο κέντρο,
    υποστήριξη Ts. Γραμμική άνοδος/κάθοδος.
    """
    # δείκτες 0..N-1, peak στο (N-1)/2
    n = np.arange(N)
    center = (N - 1) / 2.0
    tri = 1.0 - np.abs(n - center) / center
    tri[tri < 0] = 0.0
    return tri

def symbols_to_waveform(a_k: np.ndarray, N: int) -> np.ndarray:
    """
    Upsample by N (zero-order hold on impulses) κι έπειτα συνέλιξη με τον τριγωνικό παλμό p (μήκους N).
    Αποδοτικά με np.convolve.
    Επιστρέφει το x[n] μήκους len(a_k)*N + N - 1.
    """
    pulse = triangular_pulse(N)                        # p[n], peak=1
    s = np.zeros(len(a_k) * N)
    s[::N] = a_k                                       # δειγματικός παλμός ανά σύμβολο
    x = np.convolve(s, pulse, mode='full')             # x = s (*) p
    return x, pulse

# -----------------------------
# ΚΕΝΤΡΙΚΗ ΡΟΗ
# -----------------------------
# 1) Έλεγχοι & βασικές ποσότητες
if not is_power_of_two(M):
    raise ValueError("M must be power-of-two (π.χ. 4, 16, 256).")
k = int(log2(M))              # bits per symbol
Rs = Rb / k                   # symbol rate
Ts = 1.0 / Rs                 # symbol duration (απαίτηση: Rb=1 Gbps)
Fs = N / Ts                   # sample rate (N δείγματα/σύμβολο)
Tsamp = 1.0 / Fs

# 2) Bits -> Gray -> m_k -> στάθμες a_k
m_idx = bits_to_symbols_gray(input_bits, M)    # m_k από Gray
a_k = pam_levels_from_indices(m_idx, M, beta)  # στάθμες

# 3) Κυματομορφή
x, p = symbols_to_waveform(a_k, N)

# 4) Άξονας χρόνου (σε ns για αναγνωσιμότητα)
t = np.arange(len(x)) * Tsamp      # σε δευτερόλεπτα
t_ns = t * 1e9

# -----------------------------
# PLOTS
# -----------------------------
plt.figure()
plt.title(f'PAM waveform (triangular pulse), M={M}, Rb=1 Gbps, N={N}, k={k}, Ts={Ts*1e9:.3f} ns')
plt.plot(t_ns, x)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.grid(True)

# Zoom σε αρχικά δείγματα (προαιρετικό)
if show_samples is not None and show_samples > 0:
    plt.figure()
    mx = min(show_samples, len(x))
    plt.title(f'Zoom of first {mx} samples')
    plt.plot(t_ns[:mx], x[:mx])
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Προβολή και του παλμού p[n]
plt.figure()
plt.title(f'Triangular pulse p[n], length N={N}')
n = np.arange(len(p))
plt.plot(n, p)
plt.xlabel('n (samples)')
plt.ylabel('p[n]')
plt.grid(True)

plt.show()

# -----------------------------
# Εκτυπώσεις χρήσιμων τιμών
# -----------------------------
print(f"Bits per symbol k = {k}")
print(f"Symbol rate Rs = {Rs/1e9:.3f} Gsym/s")
print(f"Symbol duration Ts = {Ts*1e9:.3f} ns")
print(f"Sample rate Fs = {Fs/1e9:.3f} Gsamples/s")
print(f"Waveform length samples = {len(x)}")
