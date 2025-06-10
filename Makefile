.PHONY: install test test-cov cov test-cov-report format lint clean build docs

install:
	uv pip install -e ".[dev,all]"

test:
	uv pip install -e ".[dev]"
	uv run pytest $(filter-out $@,$(MAKECMDGOALS))

test-cov:
	uv pip install -e ".[dev]"
	uv run pytest --cov=src/pyhub/llm --cov-report=term --cov-report=html $(filter-out $@,$(MAKECMDGOALS))

cov:
	uv pip install -e ".[dev]"
	uv run pytest --cov=src/pyhub/llm --cov-report=term --cov-report=html $(filter-out $@,$(MAKECMDGOALS))

test-cov-report: test-cov
	@if [ "$$(uname)" = "Darwin" ]; then \
		open htmlcov/index.html; \
	elif [ "$$(uname)" = "Linux" ]; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Please open htmlcov/index.html manually"; \
	fi

format:
	uv pip install -e ".[dev]"
	$(eval PATHS := $(if $(filter-out $@,$(MAKECMDGOALS)),$(filter-out $@,$(MAKECMDGOALS)),./src ./tests))
	uv run black $(PATHS)
	uv run isort $(PATHS)
	uv run ruff check $(PATHS) --fix
	find $(PATHS) -name "*.html" -type f | xargs -r uv run djlint --reformat

lint:
	uv pip install -e ".[dev]"
	$(eval PATHS := $(if $(filter-out $@,$(MAKECMDGOALS)),$(filter-out $@,$(MAKECMDGOALS)),./src ./tests))
	uv run black $(PATHS) --check
	uv run isort $(PATHS) --check
	uv run ruff check $(PATHS)
	find $(PATHS) -name "*.html" -type f | xargs -r uv run djlint --check

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv pip install -e ".[build]"
	uv run -m build --wheel

publish: build
	uv run -m twine upload dist/*


docs:
	uv pip install -e ".[docs]"
	uv run mkdocs serve --dev-addr localhost:8080

docs-build:
	uv pip install -e ".[docs]"
	uv run mkdocs build --clean --site-dir docs-build

