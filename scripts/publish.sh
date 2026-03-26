#!/bin/sh
# uv publish  # publishes dist/* to PyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token $PYPI_TOKEN