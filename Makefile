# Makefile
SHELL := /bin/bash
DEVICE ?= cpu

.PHONY: help
help:
	@echo "Commands:"
	@echo "clean				: cleans all unnecessary files."
	@echo "docs-serve			: serves the documentation."
	@echo "docs-build			: builds the documentation."
	@echo "style				: runs pre-commit."
	@echo "unit-tests:			: runs unit tests."
	@echo "integration-tests:	: runs integration tests."

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".egg-info" | xargs rm -rf
	rm -f .coverage

# Styling
.PHONY: style
style:
	pre-commit install && \
	pre-commit autoupdate
	pre-commit run --all --verbose
.PHONY: docs-build
docs-build:
	mkdocs build -d ./site

.PHONY: docs-serve
docs-serve:
	mkdocs serve

.PHONY: unit-tests
unit-tests:
	@python -m pytest -v --disable-pytest-warnings --strict-markers --color=yes --device $(DEVICE)
	
