"""
go_mtl.py

Go-MTL core class.
"""
class GoMTL:
    def __init__(self, d, k, λ, μ, max_iter=50, tol=1e-4):
        self.d = d
        self.k = k
        self.λ = λ
        self.μ = μ
        self.max_iter = max_iter
        self.tol = tol
        # initialize L, S here
    def fit(self, X_list, Y_list):
        """Alternating updates to learn L and S."""
        raise NotImplementedError
    def predict(self, X_new, task_id):
        """Predict for a held-out task."""
        raise NotImplementedError
