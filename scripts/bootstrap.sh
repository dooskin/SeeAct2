#!/usr/bin/env bash
set -euo pipefail

# Use active environment; do not create/activate here.
python -m pip install --upgrade pip
python -m pip install -e ./seeact_package
python -m pip install playwright
playwright install

echo "Bootstrap complete. Try:\n  cd src && python seeact.py"
