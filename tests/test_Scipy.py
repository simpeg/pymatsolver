import unittest
from pymatsolver import Solver, Diagonal, SolverCG, SolverLU
import scipy.sparse as sp
import numpy as np

TOLD = 1e-10
TOLI = 1e-3
numRHS = 5

np.random.seed(77)


def dotest(MYSOLVER, multi=False, A=None, **solverOpts):
    if A is None:
        nx, ny, nz = 10, 10, 10
        n = nx * ny * nz
        Gz = sp.kron(
            sp.eye(nx),
            sp.kron(
                sp.eye(ny),
                sp.diags([-1, 1], [-1, 0], shape=(nz+1, nz))
            )
        )
        Gy = sp.kron(
            sp.eye(nx),
            sp.kron(
                sp.diags([-1, 1], [-1, 0], shape=(ny+1, ny)),
                sp.eye(nz),
            )
        )
        Gx = sp.kron(
            sp.diags([-1, 1], [-1, 0], shape=(nx+1, nx)),
            sp.kron(
                sp.eye(ny),
                sp.eye(nz),
            )
        )
        A = Gx.T @ Gx + Gy.T @ Gy + Gz.T @ Gz
    else:
        n = A.shape[0]

    Ainv = MYSOLVER(A, **solverOpts)
    if multi:
        e = np.ones(n)
    else:
        e = np.ones((n, numRHS))
    rhs = A * e
    x = Ainv * rhs
    Ainv.clean()
    return np.linalg.norm(e-x, np.inf)


class TestSolver(unittest.TestCase):

    def test_direct_spsolve_1(self):
        self.assertLess(dotest(Solver, False), TOLD)

    def test_direct_spsolve_M(self):
        self.assertLess(dotest(Solver, True), TOLD)

    def test_direct_splu_1(self):
        self.assertLess(dotest(SolverLU, False), TOLD)

    def test_direct_splu_M(self):
        self.assertLess(dotest(SolverLU, True), TOLD)

    def test_iterative_diag_1(self):
        self.assertLess(dotest(
            Diagonal, False,
            A=sp.diags(np.random.rand(10)+1.0)
        ), TOLI)

    def test_iterative_diag_M(self):
        self.assertLess(dotest(
            Diagonal, True,
            A=sp.diags(np.random.rand(10)+1.0)
        ), TOLI)

    def test_iterative_cg_1(self):
        self.assertLess(dotest(SolverCG, False), TOLI)

    def test_iterative_cg_M(self):
        self.assertLess(dotest(SolverCG, True), TOLI)


if __name__ == '__main__':
    unittest.main()
