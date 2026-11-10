from featurelab.features.gate import check_code

GOOD = '''
import numpy as np
from scipy import signal

def feature(x, fs, freqs, psd):
    """ok"""
    return float(np.std(x))
'''


def test_good_code_passes():
    assert check_code(GOOD).ok


def test_forbidden_import_is_rejected():
    result = check_code("import os\n" + GOOD)
    assert not result.ok and "os" in result.message()


def test_wrong_signature_is_rejected():
    assert not check_code("def feature(x):\n    return 1.0\n").ok


def test_dunder_escape_is_rejected():
    code = GOOD.replace("return float(np.std(x))", "return x.__class__.__bases__")
    assert not check_code(code).ok


def test_randomness_is_rejected():
    assert not check_code(GOOD.replace("np.std(x)", "np.random.rand()")).ok


def test_top_level_logic_is_rejected():
    assert not check_code(GOOD + "\nprint(feature(1, 2, 3, 4))\n").ok
