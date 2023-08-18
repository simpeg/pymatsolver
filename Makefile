.PHONY: build coverage lint graphs tests docs mumps mumps_mac mumps_install_mac

build:
	python setup.py build_ext --inplace

coverage:
	pytest --cov-config=.coveragerc --cov-report=xml --cov=pymatsolver -s -v

lint:
	pylint --output-format=html pymatsolver > pylint.html

graphs:
	pyreverse -my -A -o pdf -p pymatsolver pymatsolver/**.py pymatsolver/**/**.py

tests:
	pytest

docs:
	cd docs;make html

mumps:
	cd pymatsolver/mumps;make build

mumps_mac:
	cd pymatsolver/mumps;make build_mac

mumps_install_mac:
	brew install mumps --with-scotch5 --without-mpi
