.PHONY: install test format lint clean build docs

install:
	uv pip install -e ".[dev,all]"

test:
	uv pip install -e ".[dev]"
	uv run pytest $(filter-out $@,$(MAKECMDGOALS))

format:
	uv pip install -e ".[dev]"
	$(eval PATHS := $(if $(filter-out $@,$(MAKECMDGOALS)),$(filter-out $@,$(MAKECMDGOALS)),./src ./tests))
	uv run black $(PATHS)
	uv run isort $(PATHS)
	uv run ruff check $(PATHS) --fix
	uv run djlint $(PATHS) --reformat

lint:
	uv pip install -e ".[dev]"
	$(eval PATHS := $(if $(filter-out $@,$(MAKECMDGOALS)),$(filter-out $@,$(MAKECMDGOALS)),./src ./tests))
	uv run black $(PATHS) --check
	uv run isort $(PATHS) --check
	uv run ruff check $(PATHS)
	uv run djlint $(PATHS) --check

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

