"""
conftest.py
-----------
Global pytest configuration setting headless matplotlib Agg backend.
"""

import matplotlib
matplotlib.use("Agg")
