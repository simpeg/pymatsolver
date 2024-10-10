import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
import pymatsolver
from pymatsolver import Diagonal

TOL = 1e-12

class IdentitySolver(pymatsolver.solvers.Base):
    """"A concrete implementation of Base, for testing purposes"""
    def _solve_single(self, rhs):
        return rhs

    def _solve_multiple(self, rhs):
        return rhs

    def clean(self):
        # this is to test that the __del__ still executes if the object doesn't successfully clean.
        raise MemoryError("Nothing to cleanup!")

class NotTransposableIdentitySolver(IdentitySolver):
    """ A class that can't be transposed."""

    @property
    def _transpose_class(self):
        return None


def test_basics():

    Ainv = IdentitySolver(np.eye(4))
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert Ainv.shape == (4, 4)
    assert Ainv.is_real

    Ainv = IdentitySolver(np.eye(4) + 0j)
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert not Ainv.is_real

    Ainv = IdentitySolver(sp.eye(4))
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert Ainv.shape == (4, 4)

    Ainv = IdentitySolver(sp.eye(4).astype(np.complex128))
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert Ainv.shape == (4, 4)
    assert not Ainv.is_real

def test_basic_solve():
    Ainv = IdentitySolver(np.eye(4))

    rhs = np.arange(4)
    rhs2d = np.arange(8).reshape(4, 2)
    rhs3d = np.arange(24).reshape(3, 4, 2)

    npt.assert_equal(Ainv @ rhs, rhs)
    npt.assert_equal(Ainv @ rhs2d, rhs2d)
    npt.assert_equal(Ainv @ rhs3d, rhs3d)

    npt.assert_equal(rhs @ Ainv, rhs)
    npt.assert_equal(rhs * Ainv, rhs)

    npt.assert_equal(rhs2d.T @ Ainv, rhs2d.T)
    npt.assert_equal(rhs2d.T * Ainv, rhs2d.T)

    npt.assert_equal(rhs3d.swapaxes(-1, -2) @ Ainv, rhs3d.swapaxes(-1, -2))
    npt.assert_equal(rhs3d.swapaxes(-1, -2) * Ainv, rhs3d.swapaxes(-1, -2))


def test_errors_and_warnings():

    # from Base...
    with pytest.raises(ValueError, match="A must be 2-dimensional."):
        IdentitySolver(np.full((3, 3, 3), 1))

    with pytest.raises(ValueError, match="A is not a square matrix."):
        IdentitySolver(np.full((3, 5), 1))

    with pytest.warns(FutureWarning, match="accuracy_tol is deprecated.*"):
        IdentitySolver(np.full((4, 4), 1), accuracy_tol=0.41)

    with pytest.warns(UserWarning, match="Unused keyword arguments.*"):
        IdentitySolver(np.full((4, 4), 1), not_an_argument=4)

    with pytest.raises(TypeError, match="is_symmetric must be a boolean."):
        IdentitySolver(np.full((4, 4), 1), is_symmetric="True")

    with pytest.raises(TypeError, match="is_hermitian must be a boolean."):
        IdentitySolver(np.full((4, 4), 1), is_hermitian="True")

    with pytest.raises(TypeError, match="is_positive_definite must be a boolean."):
        IdentitySolver(np.full((4, 4), 1), is_positive_definite="True")

    with pytest.raises(ValueError, match="check_rtol must.*"):
        IdentitySolver(np.full((4, 4), 1), check_rtol=0.0)

    with pytest.raises(ValueError, match="check_atol must.*"):
        IdentitySolver(np.full((4, 4), 1), check_atol=-1.0)

    with pytest.raises(ValueError, match="Expected a vector of length.*"):
        Ainv = IdentitySolver(np.eye(4, 4))
        Ainv @ np.ones(3)

    with pytest.raises(ValueError, match="Second to last dimension should be.*"):
        Ainv = IdentitySolver(np.eye(4, 4))
        Ainv @ np.ones((3, 2))

    with pytest.warns(FutureWarning, match="In Future pymatsolver v0.4.0, passing a vector.*"):
        Ainv = IdentitySolver(np.eye(4, 4))
        Ainv @ np.ones((4, 1))

    with pytest.raises(NotImplementedError, match="The transpose for the.*"):
        Ainv = NotTransposableIdentitySolver(np.eye(4, 4), is_symmetric=False)
        Ainv.T



def test_DiagonalSolver():

    A = sp.identity(5)*2.0
    rhs = np.c_[np.arange(1, 6), np.arange(2, 11, 2)]
    X = Diagonal(A) * rhs
    x = Diagonal(A) * rhs[:, 0]

    sol = rhs/2.0

    with pytest.raises(TypeError):
        Diagonal(A, check_accuracy=np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        Diagonal(A, check_rtol=0)

    npt.assert_allclose(sol, X, atol=TOL)
    npt.assert_allclose(sol[:, 0], x, atol=TOL)

def test_diagonal_errors():

    with pytest.raises(TypeError, match="A must have a diagonal.*"):
        Diagonal(
            [
                [2, 0],
                [0, 1]
            ]
        )

    with pytest.raises(ValueError, match="Diagonal matrix has a zero along the diagonal."):
        Diagonal(
            np.array(
                [
                    [0, 0],
                    [0, 1]
                ]
            )
        )

def test_diagonal_inferance():

    Ainv = Diagonal(
        np.array(
            [
                [2., 0.],
                [0., 1.],
            ]
        ),
    )

    assert Ainv.is_symmetric
    assert Ainv.is_positive_definite
    assert Ainv.is_hermitian
    assert Ainv.is_real

    Ainv = Diagonal(
        np.array(
            [
                [2.0, 0],
                [0, -1.0],
            ]
        ),
    )

    assert Ainv.is_symmetric
    assert not Ainv.is_positive_definite
    assert Ainv.is_hermitian
    assert Ainv.is_real

    Ainv = Diagonal(
        np.array(
            [
                [2 + 0j, 0],
                [0, 2 + 0j],
            ]
        )
    )
    assert not Ainv.is_real
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert Ainv.is_positive_definite

    Ainv = Diagonal(
        np.array(
            [
                [2 + 0j, 0],
                [0, -2 + 0j],
            ]
        )
    )
    assert not Ainv.is_real
    assert Ainv.is_symmetric
    assert Ainv.is_hermitian
    assert not Ainv.is_positive_definite

    Ainv = Diagonal(
        np.array(
            [
                [2 + 1j, 0],
                [0, 2 + 0j],
            ]
        )
    )
    assert not Ainv.is_real
    assert Ainv.is_symmetric
    assert not Ainv.is_hermitian
    assert not Ainv.is_positive_definite
