from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
try:
    import MumpsInterface as _MUMPSINT
    _mumpsExists = True
except ImportError:
    _mumpsExists = False
import gc
from pymatsolver.solvers import Base

_mumpsErrors = {
    -1:  "An error occurred on processor INFO(2).",
    -2:  "NZ is out of range. INFO(2)=NZ.",
    -3:  "MUMPS was called with an invalid value for JOB. This may happen for example if the analysis (JOB=1) was not performed before the factorization (JOB=2), or the factorization was not performed before the solve (JOB=3), or the initialization phase (JOB=-1) was performed a second time on an instance not freed (JOB=-2). See description of JOB in Section 3. This error also occurs if JOB does not contain the same value on all processes on entry to MUMPS.",
    -4:  "Error in user-provided permutation array PERM IN at position INFO(2). This error may only occuron the host.",
    -5:  "Problem of REAL or COMPLEX workspace allocation of size INFO(2) during analysis.",
    -6:  "Matrix is singular in structure. INFO(2) holds the structural rank.",
    -7:  "Problem of INTEGER workspace allocation of size INFO(2) during analysis.",
    -8:  "Main internal integer workarray IS too small for factorization. This may happen, for example, if numerical pivoting leads to significantly more fill-in than was predicted by the analysis. The user should increase the value of ICNTL(14) before calling the factorization again (JOB=2).",
    -9:  "Main internal real/complex workarray S too small. If INFO(2) is positive, then the number of entries that are missing in S at the moment when the error is raised is available in INFO(2). If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error -9 occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=2) again, except if ICNTL(23) is provided, in which case ICNTL(23) should be increased.",
    -10: "Numerically singular matrix.",
    -11: "Internal real/complex workarray S too small for solution. Please contact us. If INFO(2) is positive, then the number of entries that are missing in S at the moment when the error is raised is available in INFO(2).",
    -12: "Internal real/complex workarray S too small for iterative refinement. Please contact us.",
    -13: "An error occurred in a Fortran ALLOCATE statement. The size that the package requested is available in INFO(2). If INFO(2) is negative, then the size that the package requested is obtained by multiplying the absolute value of INFO(2) by 1 million.",
    -14: "Internal integer workarray IS too small for solution. See error INFO(1) = -8.",
    -15: "Integer workarray IS too small for iterative refinement and/or error analysis. See error INFO(1) = -8.",
    -16: "N is out of range. INFO(2)=N.",
    -17: "The internal send buffer that was allocated dynamically by MUMPS on the processor is too small. The user should increase the value of ICNTL(14) before calling MUMPS again.",
    -20: "The internal reception buffer that was allocated dynamically by MUMPS is too small. INFO(2) holds the minimum size of the reception buffer required (in bytes). The user should increase the value of ICNTL(14) before calling MUMPS again.",
    -21: "Value of PAR=0 is not allowed because only one processor is available; Running node mode (the host is not a slave processor itself) requires at least two processors. The user should either set PAR to 1 or increase the number of processors.",
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
    -33: "ICNTL(26) was asked for during solve phase but the Schur complement was not asked for at the analysis phase (ICNTL(19)). INFO(2)=ICNTL(26).",
    -34: "LREDRHS is out of range. INFO(2)=LREDRHS.",
    -35: "This error is raised when the expansion phase is called (ICNTL(26) = 2) but reduction phase (ICNTL(26)=1) was not called before. INFO(2) contains the value of ICNTL(26).",
    -36: "Incompatible values of ICNTL(25) and INFOG(28). The value of ICNTL(25) is stored in INFO(2).",
    -37: "Value of ICNTL(25) incompatible with some other parameter. with SYM or ICNTL(xx). If INFO(2)=0 then ICNTL(25) is incompatible with SYM: in current version, the null space basis functionality is not available for unsymmetric matrices (SYM=0). Otherwise, ICNTL(25) is incompatible with ICNTL(xx), and the index xx is stored in INFO(2).",
    -38: "Parallel analysis was set (i.e., ICNTL(28)=2) but PT-SCOTCH or ParMetis were not provided.",
    -39: "Incompatible values for ICNTL(28) and ICNTL(5) and/or ICNTL(19) and/or ICNTL(6). Parallel analysis is not possible in the cases where the matrix is unassembled and/or a Schur complement is requested and/or a maximum transversal is requested on the matrix.",
    -40: "The matrix was indicated to be positive definite (SYM=1) by the user but a negative or null pivot was encountered during the processing of the root by ScaLAPACK. SYM=2 should be used.",
    -44: "The solve phase (JOB=3) cannot be performed because the factors or part of the factors are not available. INFO(2) contains the value of ICNTL(31).",
    -45: "NRHS <= 0. INFO(2) contains the value of NRHS.",
    -46: "NZ RHS <= 0. This is currently not allowed in case of reduced right-hand-side (ICNTL(26)=1) and in case entries of A-1 are requested (ICNTL(30)=1). INFO(2) contains the value of NZ RHS.",
    -47: "Entries of A-1 were requested during the solve phase (JOB=3, ICNTL(30)=1) but the constraint NRHS=N is not respected. The value of NRHS is provided in INFO(2).",
    -48: "A-1 Incompatible values of ICNTL(30) and ICNTL(xx). xx is stored in INFO(2).",
    -49: "SIZE SCHUR has an incorrect value (SIZE SCHUR < 0 or SIZE SCHUR >=N, or SIZE SCHUR was modified on the host since the analysis phase. The value of SIZE SCHUR is provided in INFO(2).",
    -90: "Error in out-of-core management. See the error message returned on output unit ICNTL(1) for more information.",
    +1:  "Index (in IRN or JCN) out of range. Action taken by subroutine is to ignore any such entries and continue. INFO(2) is set to the number of faulty entries. Details of the first ten are printed on unit ICNTL(2).",
    +2:  "During error analysis the max-norm of the computed solution was found to be zero.",
    +4:  "User data JCN has been modified (internally) by the solver.",
    +8:  "Warning return from the iterative refinement routine. More than ICNTL(10) iterations are required.",
}


if _mumpsExists:

    class _Pointer(object):
        """Gets an int and a destroy call that gets called on garbage collection.

            There can be multiple Solvers around the place that are pointing to the same factor in memory.

            This ensures that when all references are removed, there is automatic garbage collection.

            You can always clean the Solver, and it will explicitly call the destroy method.
        """
        def __init__(self, pointerINT, destroyCall):
            self.INT = pointerINT
            self.destroyCall = destroyCall

        def __del__(self):
            self.destroyCall(self.INT)

    class Mumps(Base):
        """

        documentation::

            http://mumps.enseeiht.fr/doc/userguide_4.10.0.pdf

        """

        transpose = False
        symmetric = False

        @property
        def T(self):
            newMS = self.__class__(
                self.A,
                symmetric=self.symmetric,
                fromPointer=self.pointer
            )
            newMS.transpose = not self.transpose
            return newMS

        def __init__(self, A, symmetric=False, fromPointer=None):
            self.A = A.tocsc()
            self.symmetric = symmetric

            if fromPointer is None:
                self.factor()
            elif isinstance(fromPointer, _Pointer):
                self.pointer = fromPointer
            else:
                raise Exception('Unknown pointer for construction.')

        @property
        def isfactored(self):
            return getattr(self, 'pointer', None) is not None

        def _funhandle(self, ftype):
            """
            switches the function handle between real and complex.

            ftype in ['F','S','D']

            means factor, solve, destroy
            """
            if self.A.dtype == float:
                return {'F': _MUMPSINT.factor_mumps,
                        'S': _MUMPSINT.solve_mumps,
                        'D': _MUMPSINT.destroy_mumps}[ftype]
            elif self.A.dtype == complex:
                return {'F': _MUMPSINT.factor_mumps_cmplx,
                        'S': _MUMPSINT.solve_mumps_cmplx,
                        'D': _MUMPSINT.destroy_mumps_cmplx}[ftype]

        def factor(self):
            if self.isfactored:
                return

            sym = 1 if self.symmetric else 0
            ierr, p = self._funhandle('F')(
                sym,
                self.A.data,
                self.A.indices+1,
                self.A.indptr+1
            )
            if ierr < 0:
                raise Exception("Mumps Exception [{}] - {}".format(ierr, _mumpsErrors[ierr]))
            elif ierr > 0:
                print("Mumps Warning [{}] - {}".format(ierr, _mumpsErrors[ierr]))

            self.pointer = _Pointer(p, self._funhandle('D'))

        def _solveM(self, rhs):
            self.factor()
            rhs = rhs.flatten(order='F')
            n = self.A.shape[0]
            nrhs = rhs.size // n
            T = 1 if self.transpose else 0
            sol = self._funhandle('S')(self.pointer.INT, nrhs, rhs, T)
            return sol

        _solve1 = _solveM

        def clean(self):
            self._funhandle('D')(self.pointer.INT)
            del self.pointer
            gc.collect()


del _mumpsExists
