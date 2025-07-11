from src.models.go_mtl import GoMTL

def test_initialization():
    model = GoMTL(d=5, k=3, λ=1e-2, μ=1e-1)
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
