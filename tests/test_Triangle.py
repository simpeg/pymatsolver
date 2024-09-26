import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
import pymatsolver
import pytest

TOL = 1e-12

@pytest.mark.parametrize("solver", [pymatsolver.Forward, pymatsolver.Backward])
def test_solve(solver):
    n = 50
    nrhs = 20
    A = sp.rand(n, n, 0.4) + sp.identity(n)
    sol = np.ones((n, nrhs))
    if solver is pymatsolver.Backward:
        A = sp.triu(A)
    else:
        A = sp.tril(A)
    rhs = A @ sol

    Ainv = solver(A)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)
    npt.assert_allclose(Ainv * rhs[:, 0], sol[:, 0], atol=TOL)
