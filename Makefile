dev:
	uv sync --all-groups
	uv pip install -e .
test:
	uv run pytest --cov
sync:
	uv sync
build:
	uv run pydoc-markdown
	@echo "Documentation generated to /docs/content folder."
	uv build
