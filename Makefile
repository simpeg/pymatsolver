.PHONY: coverage tests docs

coverage:
	pytest --cov --cov-config=pyproject.toml -s -v
	coverage xml

tests:
	pytest

docs:
	cd docs;make html
