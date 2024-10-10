.PHONY: build coverage lint graphs tests docs mumps mumps_mac mumps_install_mac

coverage:
	pytest --cov-config=pyproject.toml -s -v

tests:
	pytest

docs:
	cd docs;make html
