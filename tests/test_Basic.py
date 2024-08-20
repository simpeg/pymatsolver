import numpy as np
import pytest
import scipy.sparse as sp
from pymatsolver import Diagonal

TOL = 1e-12


def test_DiagonalSolver():

    A = sp.identity(5)*2.0
    rhs = np.c_[np.arange(1, 6), np.arange(2, 11, 2)]
    X = Diagonal(A) * rhs
    x = Diagonal(A) * rhs[:, 0]

    sol = rhs/2.0

    with pytest.raises(TypeError):
        Diagonal(A, check_accuracy=np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        Diagonal(A, accuracy_tol=0)

    assert np.linalg.norm(sol-X, np.inf) < TOL
    assert np.linalg.norm(sol[:, 0]-x, np.inf) < TOL
