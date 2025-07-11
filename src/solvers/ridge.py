"""
ridge.py

Closed-form ridge regression update for L.
"""
import numpy as np

def update_L(X_list, Y_list, S, reg_lambda):
    """
    Compute L = argmin ||Y - X(L S)||^2 + λ||L||_F^2
    """
    raise NotImplementedError
