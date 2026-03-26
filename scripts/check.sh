#!/bin/sh
uv run ruff check . --fix
uv run ruff format .
uv run ty check
uv run pytest