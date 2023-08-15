.PHONY: build coverage lint graphs tests docs mumps mumps_mac mumps_install_mac

build:
	python setup.py build_ext --inplace

coverage:
	coverage run --source pymatsolver -m pytest
	coverage report -m
	coverage html

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
