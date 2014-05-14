all: clean build

build:
	f2py -c TriSolve.f -m TriSolve
	rm -f *.o *.mod

clean:
	rm -f *.o *.mod *.so
