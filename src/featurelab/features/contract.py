"""The feature contract - the interface between creative agents and hard science.

Every candidate feature, whether invented by an LLM or written by a human, is a
single pure Python function::

    def feature(x, fs, freqs, psd) -> float
        x     : np.ndarray (3000,)  one 30 s epoch of EEG, microvolts
        fs    : float               sampling rate (100 Hz)
        freqs : np.ndarray (201,)   frequency axis of psd (0 - 50 Hz, 0.25 Hz steps)
        psd   : np.ndarray (201,)   Welch power spectral density of this epoch

Design reasons (worth saying in an interview):

* **Narrow interface = checkable output.**  One epoch in, one number out.  The
  function cannot see labels, other epochs or other subjects, so it *cannot*
  leak the answer.  Context across epochs is added later by the evaluator, the
  same way for every feature.
* **Shared precomputation.**  The PSD is computed once per cohort; most ideas
  need it, so generated code stays short and fast.
* **Pure + deterministic.**  Same input -> same output, no state, no I/O.  This
  is what makes results reproducible and lets us run it in a sandbox.
* **Language of the model is plain numpy/scipy.**  LLMs write that well, and so
  do teammates - no framework to learn.
"""

from __future__ import annotations

ALLOWED_IMPORTS = {"numpy", "scipy", "math"}

CONTRACT_TEXT = '''\
Write exactly one Python function with this signature:

    def feature(x, fs, freqs, psd):
        """one-line description"""
        ...
        return value  # a single float

Inputs:
  x     numpy array, shape (3000,): one 30-second epoch of single-channel EEG (Fpz-Cz), microvolts
  fs    float: sampling rate, 100.0 Hz
  freqs numpy array, shape (201,): frequency axis (0..50 Hz, 0.25 Hz resolution)
  psd   numpy array, shape (201,): Welch power spectral density of x (4 s windows)

Rules:
  - Only `import numpy as np`, `import scipy...` (e.g. `from scipy import signal`) and `import math`.
  - Imports go at the top of the code, outside the function.
  - No file, network, OS or randomness. No global state. Must be deterministic.
  - Must return one finite float for typical EEG; guard divisions with a small epsilon.
  - Budget: under ~20 ms per epoch (avoid O(n^2) loops over samples).
  - Prefer amplitude-scale-invariant definitions (ratios, normalised measures):
    the target device is ear-EEG, whose amplitude and montage differ from Fpz-Cz.
'''
