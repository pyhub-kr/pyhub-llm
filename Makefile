.PHONY: install test format lint clean build docs

install:
	pip install -e ".[dev,all]"

test:
	pytest tests/ -v

format:
	black src/ tests/
	ruff check src/ tests/ --fix

lint:
	black src/ tests/ --check
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html

publish: build
	python -m twine upload dist/*