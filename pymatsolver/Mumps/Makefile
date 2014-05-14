INCMUMPS = -I/usr/local/Cellar/mumps/4.10.0/include

LIB = -lmpiseq -lzmumps -ldmumps -lmumps_common -lblas -lpord

all: clean build

build:
	gfortran -c $(INCMUMPS) mumps_p.f90 -fPIC
	gfortran -c $(INCMUMPS) mumps_cmplx_p.f90 -fPIC
	f2py -c mumps_interface.f90 -m MumpsInterface \
        $(INCMUMPS) \
        --f90flags='-fcray-pointer' \
        $(LIB) \
        mumps_p.o mumps_cmplx_p.o
	rm -f *.o *.mod

clean:
	rm -f *.o *.mod *.so
