from src.data.preprocessing import standardize
import numpy as np

def test_standardize():
    X = np.array([[1, 2], [3, 4]], float)
    Xs = standardize(X)
    assert pytest.approx(Xs.mean(axis=0)) == 0
