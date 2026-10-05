#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:00:09 2025

@author: pantelis
"""

# gray_code.py
# Python 3.12 | Spyder
# Μέρος Α: Αναδρομική & μη-αναδρομική υλοποίηση Gray code τάξης p (2^p κωδικές λέξεις)

from typing import List

def gray_code_recursive(p: int) -> List[int]:
    """
    Αναδρομικός υπολογισμός Gray code τάξης p, ως λίστα ακεραίων [0..2^p-1] σε Gray σειρά.
    Ιδέα: G(p) = [ 0 + G(p-1), 1 + reverse(G(p-1)) ] με prefix στα bits.
    Χρονική/χωρική πολυπλοκότητα: O(2^p).
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    if p == 0:
        return [0]
    prev = gray_code_recursive(p - 1)
    head = prev
    tail = [x | (1 << (p - 1)) for x in reversed(prev)]
    return head + tail

def gray_code_iterative(p: int) -> List[int]:
    """
    Μη-αναδρομικός (κλειστή μορφή): i -> i ^ (i >> 1).
    Παράγει την ίδια ακριβώς ακολουθία με τη συμβατική αριθμητική σειρά i=0..2^p-1.
    Χρονική/χωρική πολυπλοκότητα: O(2^p).
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    n = 1 << p
    return [i ^ (i >> 1) for i in range(n)]

def to_bitstrings(seq: List[int], p: int) -> List[str]:
    """Μετατρέπει ακεραίους σε bitstrings μήκους p (με leading zeros)."""
    fmt = f'0{p}b'
    return [format(x, fmt) for x in seq]

def demo_for_MG(MG: int):
    """
    Βοηθητικό: από MG=2^p υπολογίζει p και τυπώνει
    - recursive & iterative ακολουθίες (πρώτα/τελευταία στοιχεία για μεγάλες λίστες)
    - επιβεβαίωση ότι είναι ίδιες
    """
    # Υπολογισμός p από MG = 2^p (έλεγχος εγκυρότητας)
    if MG <= 0 or (MG & (MG - 1)) != 0:
        raise ValueError("MG must be power of two (MG = 2^p).")
    p = (MG.bit_length() - 1)

    rec_seq = gray_code_recursive(p)
    it_seq  = gray_code_iterative(p)
    assert rec_seq == it_seq, "Οι δύο υλοποιήσεις πρέπει να παράγουν την ίδια ακολουθία."

    rec_bits = to_bitstrings(rec_seq, p)

    print(f"\n=== Gray code για MG={MG} (p={p}) ===")
    print(f"Πλήθος κωδικών: {len(rec_seq)}")
    # Για μικρά MG τυπώνουμε όλα, για μεγάλα δείγματα
    if MG <= 16:
        for i, (val, bits) in enumerate(zip(rec_seq, rec_bits)):
            print(f"{i:>3}: dec={val:>3}  bits={bits}")
    else:
        # δείξε τα πρώτα 8 και τα τελευταία 8
        show = 8
        for i in range(show):
            print(f"{i:>3}: dec={rec_seq[i]:>3}  bits={rec_bits[i]}")
        print(" ...")
        for i in range(MG - show, MG):
            print(f"{i:>3}: dec={rec_seq[i]:>3}  bits={rec_bits[i]}")

if __name__ == "__main__":
    # Ζητούμενα της εκφώνησης:
    for MG in (4, 16, 256):
        demo_for_MG(MG)
