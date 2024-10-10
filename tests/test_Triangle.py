import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
import pymatsolver
import pytest

TOL = 1e-12

@pytest.mark.parametrize("solver", [pymatsolver.Triangle, pymatsolver.Forward, pymatsolver.Backward])
@pytest.mark.parametrize("transpose", [True, False])
def test_solve(solver, transpose):
    n = 50
    nrhs = 20
    A = sp.rand(n, n, 0.4) + sp.identity(n)
    sol = np.ones((n, nrhs))
    if solver is pymatsolver.Backward:
        A = sp.triu(A)
        lower = False
    else:
        A = sp.tril(A)
        lower = True

    if transpose:
        rhs = A.T @ sol
        Ainv = solver(A, lower=lower).T
    else:
        rhs = A @ sol
        Ainv = solver(A, lower=lower)

    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)
    npt.assert_allclose(Ainv * rhs[:, 0], sol[:, 0], atol=TOL)


def test_triangle_errors():
    A = sp.eye(5, format='csc')

    with pytest.raises(TypeError, match="lower must be a bool."):
        Ainv = pymatsolver.Forward(A)
        Ainv.lower = 1


def test_mat_convert():
    Ainv = pymatsolver.Forward(sp.eye(5, format='coo'))
    x = np.arange(5)
    npt.assert_allclose(Ainv @ x, x)


