import pymatsolver
import pytest
import scipy.sparse as sp

@pytest.mark.skipif(pymatsolver.AvailableSolvers["Mumps"], reason="Mumps is installed.")
def test_mumps_uninstalled():
    with pytest.raises(ImportError):
        pymatsolver.Mumps(sp.eye(4))

@pytest.mark.skipif(pymatsolver.AvailableSolvers["Pardiso"], reason="Pardiso is installed.")
def test_pydiso_uninstalled():
    with pytest.raises(ImportError):
        pymatsolver.Pardiso(sp.eye(4))
