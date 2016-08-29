INCMUMPS  = -I/usr/include -L/usr/lib -L/usr/local/opt/libevent/lib /usr/local/Cellar/mumps/5.0.1_2/libexec/include
# INCMAC    = -L/usr/local/Cellar/scotch5/5.1.12b/lib -L/usr/local/Cellar/open-mpi/1.10.2/lib -I/usr/local/Cellar/mumps/5.0.1_2 -I/usr/local/Cellar/mumps/5.0.1_2/include -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include -I/usr/local/Cellar/scotch5/5.1.12b/include -I/usr/local/Cellar/open-mpi/1.10.2/include /usr/local/Cellar/mumps/5.0.1_2/libexec/include
INCMAC    = -L/usr/local/lib -L/usr/local/lib/i386 -L/usr/local/Cellar/scotch5/5.1.12b/lib -L/usr/local/Cellar/open-mpi/1.10.2/lib -L/usr/local/Cellar/mumps/5.0.1_2/lib -I/usr/local/Cellar/mumps/5.0.1_2/include -I/usr/local/Cellar/scotch5/5.1.12b/include -I/usr/local/Cellar/open-mpi/1.10.2/include

# brew install mumps --with-scotch5 --without-mpi
# Helpful: mpicc --showme

LIB = -lmpi -lblas -lpord -lcmumps
LIBSEQ = -lmpiseq_seq -lzmumps_seq -ldmumps_seq -lsmumps_seq -lmumps_common_seq
LIBPAR = -lmpiseq -lzmumps -ldmumps -lsmumps -lmumps_common
# LIBSCOTCH = -lesmumps -lptscotch -lptscotcherrexit -lscotch -lscotcherrexit -lptesmumps -lptscotcherr -lptscotchparmetis -lscotcherr -lscotchmetis
LIBSCOTCH = -lptscotch -lptscotcherrexit -lscotch -lscotcherrexit -lptesmumps -lptscotcherr -lptscotchparmetis -lscotcherr -lscotchmetis


all: clean build build_mac

build_mac:
	gfortran -c -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include mumps_p.f90 -fPIC
	gfortran -c -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include mumps_cmplx_p.f90 -fPIC

	f2py -c mumps_interface.f90 $(INCMAC) -m MumpsInterface --f90flags='-fcray-pointer' $(LIB) $(LIBPAR) mumps_p.o mumps_cmplx_p.o
	# My comment

build_mac_old:
	# gfortran -c $(INCMAC) mumps_p.f90 -fPIC
	# gfortran -fsyntax-only $(INCMAC) mumps_p.f90 -fPIC
	gfortran -c -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include mumps_p.f90 -fPIC
	# gfortran -c $(INCMAC) mumps_cmplx_p.f90 -fPIC
	gfortran -c -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include mumps_cmplx_p.f90 -fPIC


	# f2py -c mumps_interface.f90 -m MumpsInterface $(INCMAC) --f90flags='-fcray-pointer' $(LIB) $(LIBPAR) $(LIBSCOTCH) mumps_p.o mumps_cmplx_p.o
	# f2py -c mumps_interface.f90 -m MumpsInterface -I/usr/local/Cellar/mumps/5.0.1_2/libexec/include --f90flags='-fcray-pointer' $(LIB) $(LIBPAR) $(LIBSCOTCH) mumps_p.o mumps_cmplx_p.o
	# rm -f *.o *.mod

build:
	gfortran -c $(INCMUMPS) mumps_p.f90 -fPIC
	gfortran -c $(INCMUMPS) mumps_cmplx_p.f90 -fPIC
	f2py -c mumps_interface.f90 -m MumpsInterface \
        $(INCMUMPS) \
        --f90flags='-fcray-pointer' \
        $(LIB) $(LIBSEQ)\
        mumps_p.o mumps_cmplx_p.o
	rm -f *.o *.mod

clean:
	rm -f *.o *.mod *.so

test:
	cd ..;nosetests
