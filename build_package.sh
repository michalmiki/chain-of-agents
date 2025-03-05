#!/bin/bash

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python setup.py sdist bdist_wheel

# Optional: Upload to PyPI
# python -m twine upload dist/*

# Optional: Upload to TestPyPI
# python -m twine upload --repository testpypi dist/*

echo "Package built successfully. Distribution files are in the 'dist' directory."
echo "To upload to PyPI, run: python -m twine upload dist/*"
echo "To upload to TestPyPI, run: python -m twine upload --repository testpypi dist/*"
