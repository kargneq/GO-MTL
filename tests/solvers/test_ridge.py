from src.solvers.ridge import update_L

def test_update_L_shape():
    # simple shape test
    import numpy as np
    X_list = [np.ones((5,3))]
    Y_list = [np.ones((5,1))]
    S = np.ones((2,1))
    L = update_L(X_list, Y_list, S, 1e-2)
    assert L.shape == (3,2)
