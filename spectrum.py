#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:23:13 2025

@author: pantelis
"""

# spectrum.py
# Μέρος Γ: Θεωρητική PSD και σύγκριση με FFT + ΑΠΟΘΗΚΕΥΣΕΙΣ (PNG/CSV/TXT)

import os
import numpy as np
import matplotlib.pyplot as plt
from math import log2

# -----------------------------
# ΡΥΘΜΙΣΕΙΣ
# -----------------------------
M = 16                  # 4, 16, 256 ...
Rb = 1e9                # 1 Gbit/s
N = 32                  # δείγματα ανά σύμβολο Ts
beta = 1.0              # κλίμακα σταθμών (β=1)
Kbits = 4096            # μήκος bitstream (μπορείς να το αυξήσεις)
RNG_SEED = 123
SAVE = True             # αποθήκευση αρχείων
SAVE_DIR = "outputs_MG" # φάκελος εξόδου (δημιουργείται αν δεν υπάρχει)
FIG_DPI = 180

# -----------------------------
# ΒΟΗΘΗΤΙΚΑ
# -----------------------------
def is_pow2(x): return x>0 and (x & (x-1))==0

def gray_to_binary(g: int) -> int:
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def bits_to_m_gray(bits: str, M: int) -> np.ndarray:
    if not is_pow2(M):
        raise ValueError("M must be power of two.")
    k = int(log2(M))
    if len(bits) % k != 0:
        bits = bits + '0' * (k - (len(bits) % k))
    m = []
    for i in range(0, len(bits), k):
        g = int(bits[i:i+k], 2)
        m.append(gray_to_binary(g))
    return np.asarray(m, dtype=int)

def pam_levels(m_idx: np.ndarray, M: int, beta: float=1.0) -> np.ndarray:
    return beta * (2*m_idx - (M - 1))

def triangular_pulse(N: int) -> np.ndarray:
    n = np.arange(N)
    c = (N - 1) / 2.0
    tri = 1.0 - np.abs(n - c) / c
    tri[tri < 0] = 0.0
    return tri

def waveform_from_symbols(a_k: np.ndarray, N: int):
    p = triangular_pulse(N)
    s = np.zeros(len(a_k) * N)
    s[::N] = a_k
    x = np.convolve(s, p, mode='full')
    return x, p

def Ea2_uniform(M: int, beta: float=1.0) -> float:
    m = np.arange(M, dtype=float)
    A = beta * (2*m - (M - 1))
    return np.mean(A**2)

def P_tri_abs(f, Ts):
    # |P(f)| = (Ts/2) * sinc^2(f*Ts/2),  με numpy sinc(x)=sin(pi x)/(pi x)
    return (Ts/2.0) * (np.sinc(f*Ts/2.0)**2)

# -----------------------------
# ΚΥΡΙΟ
# -----------------------------
rng = np.random.default_rng(RNG_SEED)
bits = ''.join(rng.choice(['0','1'], size=Kbits))

k = int(log2(M))
Rs = Rb / k
Ts = 1.0 / Rs
Fs = N / Ts
Tsamp = 1.0 / Fs

m_idx = bits_to_m_gray(bits, M)
a_k = pam_levels(m_idx, M, beta)
x, p = waveform_from_symbols(a_k, N)

L = len(x)
t = np.arange(L) * Tsamp

# Θεωρητικό S_X(f)
Ea2 = Ea2_uniform(M, beta)
f = np.fft.fftfreq(L, d=Tsamp)
f = np.fft.fftshift(f)
Pf_abs = P_tri_abs(f, Ts)
S_theory = (Ea2 / Ts) * (Pf_abs**2)

# Αριθμητικό (Periodogram)
X = np.fft.fftshift(np.fft.fft(x))
Tobs = L * Tsamp
S_num = (Tsamp / Tobs) * (np.abs(X)**2)

# Κανονικοποιημένα για σύγκριση σχήματος
eps = 1e-18
S_theory_norm = S_theory / (np.max(S_theory) + eps)
S_num_norm = S_num / (np.max(S_num) + eps)

# -----------------------------
# PLOTS
# -----------------------------
# 1) Σύγκριση κανονικοποιημένων
plt.figure()
plt.plot(f/1e9, S_theory_norm, label='Theory (normalized)')
plt.plot(f/1e9, S_num_norm, '--', label='FFT/periodogram (normalized)')
plt.xlim([-Fs/(2e9), Fs/(2e9)])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Normalized PSD')
plt.title(f'PAM PSD comparison (M={M}, Rb=1 Gbps, N={N}, Ts={Ts*1e9:.3f} ns)')
plt.grid(True)
plt.legend()
fig1 = plt.gcf()

# 2) Απόλυτες (λογάριθμος)
plt.figure()
plt.semilogy(f/1e9, S_theory + eps, label='Theory')
plt.semilogy(f/1e9, S_num + eps, '--', label='FFT/periodogram')
plt.xlim([-Fs/(2e9), Fs/(2e9)])
plt.xlabel('Frequency (GHz)')
plt.ylabel('PSD (arb. units)')
plt.title('Absolute PSD (log scale)')
plt.grid(True, which='both')
plt.legend()
fig2 = plt.gcf()

plt.show()

# -----------------------------
# ΑΠΟΘΗΚΕΥΣΕΙΣ
# -----------------------------
if SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # PNGs
    fig1.savefig(os.path.join(SAVE_DIR, f"PSD_comparison_M{M}_N{N}.png"), dpi=FIG_DPI, bbox_inches='tight')
    fig2.savefig(os.path.join(SAVE_DIR, f"PSD_absolute_M{M}_N{N}.png"), dpi=FIG_DPI, bbox_inches='tight')

    # CSV με φάσμα: f(Hz), S_theory, S_num, S_theory_norm, S_num_norm
    data = np.column_stack([f, S_theory, S_num, S_theory_norm, S_num_norm])
    header = "f_Hz,S_theory,S_num,S_theory_norm,S_num_norm"
    np.savetxt(os.path.join(SAVE_DIR, f"PSD_data_M{M}_N{N}.csv"), data, delimiter=",", header=header, comments='')

    # Summary TXT
    with open(os.path.join(SAVE_DIR, f"summary_M{M}_N{N}.txt"), "w") as fh:
        fh.write("PAM PSD Summary\n")
        fh.write("----------------\n")
        fh.write(f"M               : {M}\n")
        fh.write(f"k (bits/symbol) : {k}\n")
        fh.write(f"Rb (bit/s)      : {Rb:.0f}\n")
        fh.write(f"Rs (sym/s)      : {Rs:.3e}\n")
        fh.write(f"Ts (s)          : {Ts:.3e}\n")
        fh.write(f"N (samples/sym) : {N}\n")
        fh.write(f"Fs (samples/s)  : {Fs:.3e}\n")
        fh.write(f"L (samples)     : {L}\n")
        fh.write(f"Tobs (s)        : {Tobs:.3e}\n")
        fh.write(f"E[a^2]          : {Ea2:.6f}\n")
        fh.write(f"RNG_SEED        : {RNG_SEED}\n")

print(f"Έτοιμο! Saved to: {os.path.abspath(SAVE_DIR)}")
