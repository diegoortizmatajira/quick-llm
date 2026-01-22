generate:
	uv run pydoc-markdown
	@echo "Documentation generated to /docs/content folder."
test:
	uv run pytest --cov
sync:
	uv sync
build:
	uv build
