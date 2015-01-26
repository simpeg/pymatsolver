INCMUMPS  = -I/usr/local/Cellar/mumps/4.10.0/include -L/usr/local/Cellar/scotch5/5.1.12b/lib -I/usr/local/Cellar/scotch5/5.1.12b/include -I/usr/local/Cellar/open-mpi/1.8.3/include -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/1.8.3/lib -lmpi

# Helpful: mpicc --showme

LIB = -lmpiseq -lzmumps -ldmumps -lsmumps -lmumps_common -lblas -lpord -lcmumps
LIBSCOTCH = -lesmumps -lptscotch -lptscotcherrexit -lscotch -lscotcherrexit -lptesmumps -lptscotcherr -lptscotchparmetis -lscotcherr -lscotchmetis

all: clean build

build:
	gfortran -c $(INCMUMPS) mumps_p.f90 -fPIC
	gfortran -c $(INCMUMPS) mumps_cmplx_p.f90 -fPIC
	f2py -c mumps_interface.f90 -m MumpsInterface \
        $(INCMUMPS) \
        --f90flags='-fcray-pointer' \
        $(LIB) $(LIBSCOTCH)\
        mumps_p.o mumps_cmplx_p.o
	rm -f *.o *.mod

clean:
	rm -f *.o *.mod *.so

test:
	cd ..;nosetests
