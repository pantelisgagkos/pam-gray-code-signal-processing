# TM6M666666666666666666elecommunications Systems Project

Python implementation developed for the Telecommunications Systems course.

## Project Overview

This project includes three main parts:

- Gray code generation using recursive and iterative methods
- M-PAM waveform generation with Gray mapping and triangular pulse shaping
- Power Spectral Density analysis using theoretical calculations and FFT/periodogram comparison
^
## Files

### `gray_code.py`
Implements recursive and iterative Gray code generation for different modulation orders.

### `pam_waveform.py`
Converts binary input into Gray-coded M-PAM symbols and generates the corresponding waveform using triangular pulses.

### `spectrum.py`
Computes the theoretical Power Spectral Density of the PAM signal and compares it with a numerical FFT/periodogram estimate.

### `report.pdf`
Course report containing the theoretical background, methodology and results.

## Requirements

- Python 3
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 gray_code.py
python3 pam_waveform.py
python3 spectrum.py
```

## Topics
- Digital Communications
- Gray Coding
- Pulse Amplitude Modulation (PAM)
- Signal Processing
- Fourier Transform
- Power Spectral Density
- Python / NumPy / Matplotlib
