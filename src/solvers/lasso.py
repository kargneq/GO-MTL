"""
lasso.py

Coordinate-descent LASSO to update each task weight s.
"""
import numpy as np

def update_s_for_task(X, y, L, mu, max_iter=100):
    """Solve min_s ||y - X L s||^2 + μ||s||_1."""
    raise NotImplementedError
