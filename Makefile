.PHONY: test lint docs

test:
	pytest

lint:
	flake8 src tests

docs:
	(cd docs && make html)