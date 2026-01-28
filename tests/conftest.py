import os
import sys
from unittest.mock import MagicMock


# Helper to mock a module structure if it's missing
def mock_module(name):
    if name in sys.modules:
        return sys.modules[name]

    try:
        # Check if we can import it
        __import__(name)
        return sys.modules[name]
    except ImportError:
        pass

    m = MagicMock()
    sys.modules[name] = m
    return m


# Mock missing dependencies to allow running tests in environments without heavy libs
mock_module("torch")
mock_module("ase")
mock_module("ase.data")
mock_module("ase.optimize")
mock_module("fairchem")
mock_module("fairchem.core")
mock_module("morfeus")
mock_module("morfeus.conformer")
mock_module("rdkit")
mock_module("rdkit.Chem")
mock_module("tables")
mock_module("hf_xet")

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
