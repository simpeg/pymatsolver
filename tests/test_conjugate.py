import pytest
import pymatsolver
import numpy as np
import scipy.sparse as sp
import numpy.testing as npt


@pytest.mark.parametrize('solver_class', [pymatsolver.Solver, pymatsolver.SolverLU, pymatsolver.Pardiso, pymatsolver.Mumps])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('n_rhs', [1, 4])
def test_conjugate_solve(solver_class, dtype, n_rhs):
    if solver_class is pymatsolver.Pardiso and not pymatsolver.AvailableSolvers['Pardiso']:
        pytest.skip("pydiso not installed.")
    if solver_class is pymatsolver.Mumps and not pymatsolver.AvailableSolvers['Mumps']:
        pytest.skip("python-mumps not installed.")

    n = 10
    D = sp.diags(np.linspace(1, 10, n))
    if dtype == np.float64:
        L = sp.diags([1, -1], [0, -1], shape=(n, n))

        sol = np.linspace(0.9, 1.1, n)
        # non-symmetric real matrix
    else:
        # non-symmetric
        L = sp.diags([1, -1j], [0, -1], shape=(n, n))
        sol = np.linspace(0.9, 1.1, n) - 1j * np.linspace(0.9, 1.1, n)[::-1]

    if n_rhs > 1:
        sol = np.pad(sol[:, None], [(0, 0), (0, n_rhs - 1)], mode='constant')

    A = D @ L @ D @ L.T

    # double check it solves
    rhs = A @ sol
    Ainv = solver_class(A)
    npt.assert_allclose(Ainv @ rhs, sol)

    # is conjugate solve correct?
    rhs_conj = A.conjugate() @ sol
    Ainv_conj = Ainv.conjugate()
    npt.assert_allclose(Ainv_conj @ rhs_conj, sol)

    # is conjugate -> conjugate solve correct?
    Ainv2 = Ainv_conj.conjugate()
    npt.assert_allclose(Ainv2 @ rhs, sol)