"""
Pytest configuration — shared fixtures and path setup.
Eliminates the need for sys.path.insert in every test file.
"""

import sys
from pathlib import Path

# Add project root to Python path once
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
