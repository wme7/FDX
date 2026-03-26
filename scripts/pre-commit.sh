#!/bin/sh

# Run against staged files
# git add .
# uv run pre-commit run 

# Run against all files
uv run pre-commit run --all-files

# Run specific hooks
# uv run pre-commit run ruff-check --all-files
# uv run pre-commit run ruff-format --all-files
# uv run pre-commit run ty-check --all-files
