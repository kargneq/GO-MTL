from src.solvers.lasso import update_s_for_task
import numpy as np

def test_update_s_for_task_shape():
    X = np.ones((10,4))
    y = np.ones(10)
    L = np.ones((4,2))
    s = update_s_for_task(X, y, L, 0.1, max_iter=5)
    assert s.shape == (2,)
