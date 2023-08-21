from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import gc
import warnings

import numpy as np

from pymatsolver.solvers import Base
from . import MumpsInterface as _MUMPSINT


def _mumps_message_from_exit_code(ierr):
    """Convert a MUMPS routine's exit code to a message.

    See section 8 of the MUMPS user guide for a full catalog of these messages.
    """
    error_messages = {
        -1: "An error occurred on processor INFO(2).",
        -2: "NNZ/NZ, NNZ loc/NZ loc or P NNZ loc/P NZ loc are out of range. INFO(2)=NNZ/NZ, NNZ loc/NZ loc or P NNZ loc/P NZ loc.",
        -3: "MUMPS was called with an invalid value for JOB. This may happen if the analysis (JOB=1) was not performed (or failed) before the factorization (JOB=2), or the factorization was not performed (or failed) before the solve (JOB=3), or the initialization phase (JOB=-1) was performed a second time on an instance not freed (JOB=-2). See description of JOB in Section 4. This error also occurs if JOB does not contain the same value on all processes on entry to MUMPS. INFO(2) is then set to the local value of JOB.",
        -4: "Error in user-provided permutation array PERM IN at position INFO(2). This error may only occur on the host.",
        -5: 'Problem of real workspace allocation of size INFO(2) during analysis. The unit for INFO(2) is the number of real values (single precision for SMUMPS/CMUMPS, double precision for DMUMPS/ZMUMPS), in the Fortran "ALLOCATE" statement that did not succeed. If INFO(2) is negative, then its absolute value should be multiplied by 1 million.',
        -6: "Matrix is singular in structure. INFO(2) holds the structural rank.",
        -7: "Problem of integer workspace allocation of size INFO(2) during analysis. The unit for INFO(2) is the number of integer values that MUMPS tried to allocate in the Fortran ALLOCATE statement that did not succeed. If INFO(2) is negative, then its absolute value should be multiplied by 1 million.",
        -8: "Main internal integer workarray IS too small for factorization. This may happen, for example, if numerical pivoting leads to significantly more fill-in than was predicted by the analysis. The user should increase the value of ICNTL(14) before calling the factorization again (JOB=2).",
        -9: "The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number of entries that are missing in S at the moment when the error is raised is available in INFO(2). If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error -9 occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=2) again, except if LWK USER is provided LWK USER should be increased.",
        -10: "Numerically singular matrix. INFO(2) holds the number of eliminated pivots.",
        -11: "Internal real/complex workarray S or LWK USER too small for solution. If INFO(2) is positive, then the number of entries that are missing in S/LWK USER at the moment when the error is raised is available in INFO(2). If the numerical phases are out-of-core and LWK USER is provided for the solution phase and is smaller than the value provided for the factorization, it should be increased by at least INFO(2). In other cases, please contact us.",
        -12: "Internal real/complex workarray S too small for iterative refinement. Please contact us.",
        -13: "Problem of workspace allocation of size INFO(2) during the factorization or solve steps. The size that the package tried to allocate with a Fortran ALLOCATE statement is available in INFO(2). If INFO(2) is negative, then the size that the package requested is obtained by multiplying the absolute value of INFO(2) by 1 million. In general, the unit for INFO(2) is the number of scalar entries of the type of the input matrix (real, complex, single or double precision).",
        -14: "Internal integer workarray IS too small for solution. See error INFO(1) = -8.",
        -15: "Integer workarray IS too small for iterative refinement and/or error analysis. See error INFO(1) = -8.",
        -16: "N is out of range. INFO(2)=N.",
        -17: "The internal send buffer that was allocated dynamically by MUMPS on the processor is too small. The user should increase the value of ICNTL(14) before calling MUMPS again.",
        -18: "The blocking size for multiple RHS (ICNTL(27)) is too large and may lead to an integer overflow. This error may only occurs for very large matrices with large values of ICNTL(27) (e.g., several thousands). INFO(2) provides an estimate of the maximum value of ICNTL(27) that should be used.",
        -19: "The maximum allowed size of working memory ICNTL(23) is too small to run the factorization phase and should be increased. If INFO(2) is positive, then the number of entries that are missing at the moment when the error is raised is available in INFO(2). If INFO(2) is negative, then its absolute value should be multiplied by 1 million.",
        -20: "The internal reception buffer that was allocated dynamically by MUMPS is too small. Normally, this error is raised on the sender side when detecting that the message to be sent is too large for the reception buffer on the receiver. INFO(2) holds the minimum size of the reception buffer required (in bytes). The user should increase the value of ICNTL(14) before calling MUMPS again.",
        -21: "Value of PAR=0 is not allowed because only one processor is available; Running MUMPS in host-node mode (the host is not a slave processor itself) requires at least two processors. The user should either set PAR to 1 or increase the number of processors.",
        -22: "A pointer array is provided by the user that is either not associated, or has insufficient size, or is associated and should not be associated (for example, RHS on non-host processors).",
        -23: "MPI was not initialized by the user prior to a call to MUMPS with JOB = -1.",
        -24: "NELT is out of range. INFO(2)=NELT.",
        -25: "A problem has occurred in the initialization of the BLACS. This may be because you are using a vendor's BLACS. Try using a BLACS version from netlib instead.",
        -26: "LRHS is out of range. INFO(2)=LRHS.",
        -27: "NZ RHS and IRHS PTR(NRHS+1) do not match. INFO(2) = IRHS PTR(NRHS+1).",
        -28: "IRHS PTR(1) is not equal to 1. INFO(2) = IRHS PTR(1).",
        -29: "LSOL loc is smaller than INFO(23). INFO(2)=LSOL loc.",
        -30: "SCHUR LLD is out of range. INFO(2) = SCHUR LLD.",
        -31: "A 2D block cyclic symmetric (SYM=1 or 2) Schur complement is required with the option ICNTL(19)=3, but the user has provided a process grid that does not satisfy the constraint MBLOCK=NBLOCK. INFO(2)=MBLOCK-NBLOCK.",
        -32: "Incompatible values of NRHS and ICNTL(25). Either ICNTL(25) was set to -1 and NRHS is different from INFOG(28); or ICNTL(25) was set to i, 1 <= i <= INFOG(28) and NRHS is different from 1. Value of NRHS is stored in INFO(2).",
        -33: "ICNTL(26) was asked for during solve phase (or during the factorization - see ICNTL(32)) but the Schur complement was not asked for at the analysis phase (ICNTL(19)). INFO(2)=ICNTL(26).",
        -34: "LREDRHS is out of range. INFO(2)=LREDRHS.",
        -35: "This error is raised when the expansion phase is called (ICNTL(26) = 2) but reduction phase (ICNTL(26)=1) was not called before. This error also occurs in case the reduction phase (ICNTL(26) = 1) is asked for at the solution phase (JOB=3) but the forward elimination was already performed during the factorization phase (JOB=2 and ICNTL(32)=1). INFO(2) contains the value of ICNTL(26).",
        -36: "Incompatible values of ICNTL(25) and INFOG(28). The value of ICNTL(25) is stored in INFO(2).",
        -37: "Value of ICNTL(25) incompatible with some other parameter. If ICNTL(25) is incompatible with ICNTL(xx), the index xx is stored in INFO(2).",
        -38: "Parallel analysis was set (i.e., ICNTL(28)=2) but PT-SCOTCH or ParMetis were not provided.",
        -39: "Incompatible values for ICNTL(28) and ICNTL(5) and/or ICNTL(19) and/or ICNTL(6). Parallel analysis is not possible in the cases where the matrix is unassembled and/or a Schur complement is requested and/or a maximum transversal is requested on the matrix.",
        -40: "The matrix was indicated to be positive definite (SYM=1) by the user but a negative or null pivot was encountered during the processing of the root by ScaLAPACK. SYM=2 should be used.",
        -41: "Incompatible value of LWK USER between factorization and solution phases. This error may only occur when the factorization is in-core (ICNTL(22)=1), in which case both the contents of WK USER and LWK USER should be passed unchanged between the factorization (JOB=2) and solution (JOB=3) phases.",
        -42: "ICNTL(32) was set to 1 (forward during factorization), but the value of NRHS on the host processor is incorrect: either the value of NRHS provided at analysis is negative or zero, or the value provided at factorization or solve is different from the value provided at analysis. INFO(2) holds the value of id%NRHS that was provided at analysis.",
        -43: "Incompatible values of ICNTL(32) and ICNTL(xx). The index xx is stored in INFO(2).",
        -44: "The solve phase (JOB=3) cannot be performed because the factors or part of the factors are not available. INFO(2) contains the value of ICNTL(31).",
        -45: "NRHS <= 0. INFO(2) contains the value of NRHS.",
        -46: "NZ RHS <= 0. This is currently not allowed in case of reduced right-hand-side (ICNTL(26)=1) and in case entries of A-1 are requested (ICNTL(30)=1). INFO(2) contains the value of NZ RHS.",
        -47: "Entries of A-1 were requested during the solve phase (JOB=3, ICNTL(30)=1) but the constraint NRHS=N is not respected. The value of NRHS is provided in INFO(2).",
        -48: "A-1 Incompatible values of ICNTL(30) and ICNTL(xx). xx is stored in INFO(2).",
        -49: "SIZE SCHUR has an incorrect value: SIZE SCHUR < 0 or SIZE SCHUR >= N, or SIZE SCHUR was modified on the host since the analysis phase. The value of SIZE SCHUR is provided in INFO(2).",
        -50: "An error occurred while computing the fill-reducing ordering during the analysis phase. This commonly happens when an (external) ordering tool returns an error code or a wrong result.",
        -51: "An external ordering (Metis/ParMetis, SCOTCH/PT-SCOTCH, PORD), with 32-bit default integers, is invoked to processing a graph of size larger than 231 - 1. INFO(2) holds the size required to store the graph as a number of integer values; it is negative and its absolute value should be multiplied by 1 million.",
        -52: "When default Fortran integers are 64 bit (e.g. Fortran compiler flag -i8 -fdefault-integer-8 or something equivalent depending on your compiler) then external ordering libraries (Metis/ParMetis, SCOTCH/PT-SCOTCH, PORD) should also have 64-bit default integers. INFO(2) = 1, 2, 3 means that respectively Metis/ParMetis, SCOTCH/PT-SCOTCH or PORD were invoked and were not generated with 64-bit default integers.",
        -53: "Internal error that could be due to inconsistent input data between two consecutive calls.",
        -54: "The analysis phase (JOB=1) was called with ICNTL(35)=0 but the factorization phase was called with ICNTL(35)=1, 2 or 3. In order to perform the factorization with BLR compression, please perform the analysis phase again using ICNTL(35)=1, 2 or 3 (see the documentation of ICNTL(35)).",
        -55: "During a call to MUMPS including the solve phase with distributed right-hand side, LRHS loc was detected to be smaller than Nloc RHS. INFO(2)=LRHS loc.",
        -56: "During a call to MUMPS including the solve phase with distributed right-hand side and distributed solution, RHS loc and SOL loc point to the same workarray but LRHS loc < LSOL loc. INFO(2)=LRHS loc.",
        -57: "During a call to MUMPS analysis phase with a block format (ICNTL(15) != 0), an error in the interface provided by the user was detected. INFO(2) holds additional information about the issue.",
        -70: "During a call to MUMPS with JOB=7, the file specified to save the current instance, as derived from SAVE DIR and/or SAVE PREFIX, already exists. Before saving an instance into this file, it should be first suppressed (see JOB=-3). Otherwise, a different file should be specified by changing the values of SAVE DIR and/or SAVE PREFIX.",
        -71: "An error has occured during the creation of one of the files needed to save MUMPS data (JOB=7).",
        -72: "Error while saving data (JOB=7); a write operation did not succeed (e.g., disk full, I/O error, . . . ). INFO(2) is the size that should have been written during that operation. If INFO(2) is negative, then its absolute value should be multiplied by 1 million.",
        -73: "During a call to MUMPS with JOB=8, one parameter of the current instance is not compatible with the corresponding one in the saved instance.",
        -74: "The file resulting from the setting of SAVE DIR and SAVE PREFIX could not be opened for restoring data (JOB=8). INFO(2) is the rank of the process (in the communicator COMM) on which the error was detected.",
        -75: "Error while restoring data (JOB=8); a read operation did not succeed (e.g., end of file reached, I/O error, . . . ). INFO(2) is the size still to be read. If INFO(2) is negative, then the size that the package requested is obtained by multiplying the absolute value of INFO(2) by 1 million.",
        -76: "Error while deleting the files (JOB=-3); some files to be erased were not found or could not be suppressed. INFO(2) is the rank of the process (in the communicator COMM) on which the error was detected.",
        -77: "Neither SAVE DIR nor the environment variable MUMPS SAVE DIR are defined.",
        -78: "Problem of workspace allocation during the restore step. The size still to be allocated is available in INFO(2). If INFO(2) is negative, then the size that the package requested is obtained by multiplying the absolute value of INFO(2) by 1 million.",
        -79: "MUMPS could not find a Fortran file unit to perform I/O's.",
        -89: "Internal error during SCOTCH kway-partitioning in SCOTCHFGRAPHPART. METIS package should be made available to MUMPS.",
        -90: "Error in out-of-core management. See the error message returned on output unit ICNTL(1) for more information.",
        -800: "Temporary error associated to the current MUMPS release, subject to change or disappearance in the future. If INFO(2)=5, then this error is due to the fact that the elemental matrix format (ICNTL(5)=1) is currently incompatible with a BLR factorization (ICNTL(35) != 0).",
    }

    warning_messages = {
        1: "Index (in IRN or JCN) out of range. Action taken by subroutine is to ignore any such entries and continue. INFO(2) is set to the number of faulty entries. Details of the first ten are printed on unit ICNTL(2).",
        2: "During error analysis the max-norm of the computed solution was found to be zero.",
        4: "User data JCN has been modified (internally) by the solver.",
        8: "Warning return from the iterative refinement routine. More than ICNTL(10) iterations are required.",
    }

    if ierr == 0:
        return "No error occurred."
    elif ierr < 0:
        return error_messages.get(ierr, "An unknown error occurred.")
    else:
        # The warning values are actually bitwise flags.
        messages = []
        for bit in warning_messages.keys():
            if ierr & bit != 0:
                messages.append(warning_messages[bit])

        if not messages:
            messages.append("An unknown warning occurred.")

        return "\n".join(messages)


class _Pointer:
    """Store a pointer with a destroy function that gets called on garbage collection.

    Multiple Mumps solver instances can point to the same factor in memory.
    This class ensures that the factor is garbage collected only when all references are removed.

    You can always clean the Solver, and it will explicitly call the destroy method.
    """
    def __init__(self, pointerINT, dtype):
        self.INT = pointerINT
        # The destroy function should only be called from this class upon deallocation.
        if dtype == float:
            self._destroy_func = _MUMPSINT.destroy_mumps
        elif dtype == complex:
            self._destroy_func = _MUMPSINT.destroy_mumps_cmplx
        else:
            raise ValueError(f"Attempted to use an invalid data type ({dtype})")

    def __del__(self):
        self._destroy_func(self.INT)


class Mumps(Base):
    """

    documentation::

        https://mumps-solver.org/doc/userguide_5.6.1.pdf

    """

    _transpose = False
    _conjugate = False

    @property
    def T(self):
        """Transpose operator.

        Allows solving A^T * x = b without needing to factor again.
        """
        return self._make_copy(transpose=True)

    def conjugate(self):
        """Conjugate operator.

        Allows solving \\bar(A) * x = b without needing to factor again.
        """
        return self._make_copy(conjugate=True)

    def _make_copy(self, *, transpose=False, conjugate=False):
        properties_with_setters = []
        for a in dir(self.__class__):
            attr = getattr(self.__class__, a)
            if hasattr(attr, "fset") and attr.fset is not None:
                properties_with_setters.append(a)
        kwargs = {attr: getattr(self, attr) for attr in properties_with_setters}

        copy = self.__class__(
            self.A,
            from_pointer=self.pointer,
            **kwargs,
        )
        copy._transpose = (not self._transpose) if transpose else self._transpose
        copy._conjugate = (not self._conjugate) if conjugate else self._conjugate
        return copy

    def __init__(self, A, from_pointer=None, **kwargs):
        self.A = A.tocsc()
        # As of 5.6.1, MUMPS has no optimizations for Hermitian matrices
        self.set_kwargs(**kwargs, ignore="is_hermitian")

        if from_pointer is None:
            self.factor()
        elif isinstance(from_pointer, _Pointer):
            self.pointer = from_pointer
        else:
            raise Exception('Unknown pointer for construction.')

    @property
    def _is_factored(self):
        return getattr(self, 'pointer', None) is not None

    @property
    def _matrix_type(self):
        if self.is_symmetric:
            if self.is_positive_definite:
                return 1  # symmetric, positive definite
            return 2  # general symmetric
        return 0  # unsymmetric

    def _funhandle(self, function):
        """Switch the function handle between real and complex."""
        if self.A.dtype == float:
            return {
                'factor': _MUMPSINT.factor_mumps,
                'solve': _MUMPSINT.solve_mumps
            }[function]
        elif self.A.dtype == complex:
            return {
                'factor': _MUMPSINT.factor_mumps_cmplx,
                'solve': _MUMPSINT.solve_mumps_cmplx
            }[function]

        raise ValueError(f"Attempted to use an invalid data type ({self.A.dtype})")

    def factor(self):
        if self._is_factored:
            return

        ierr, p = self._funhandle('factor')(
            self._matrix_type,
            self.A.data,
            self.A.indices+1,
            self.A.indptr+1
        )
        if ierr < 0:
            raise Exception(f"Mumps Exception [{ierr}] - {_mumps_message_from_exit_code(ierr)}")
        elif ierr > 0:
            warnings.warn(f"Mumps Warning [{ierr}] - {_mumps_message_from_exit_code(ierr)}")

        self.pointer = _Pointer(p, self.A.dtype)

    def _solveM(self, rhs):
        self.factor()
        rhs = rhs.flatten(order='F')
        n = self.A.shape[0]
        nrhs = rhs.size // n
        T = 1 if self._transpose else 0
        sol = self._funhandle('solve')(
            self.pointer.INT,
            nrhs,
            np.conjugate(rhs) if self._conjugate else rhs,
            T
        )
        return np.conjugate(sol) if self._conjugate else sol

    _solve1 = _solveM

    def clean(self):
        del self.pointer
        gc.collect()
