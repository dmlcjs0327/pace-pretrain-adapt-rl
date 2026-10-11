#!/bin/bash
# Clone Aerial Gym Simulator v2.0.0 and copy our overlay onto it.
# Usage: bash setup/apply_overlay.sh <target directory>      (default: ./aerial_gym_simulator)
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
TARGET=${1:-$PWD/aerial_gym_simulator}
UPSTREAM=https://github.com/ntnu-arl/aerial_gym_simulator.git
COMMIT=198ebc12198f24c2f2a8c043139679c1d8b58485   # tag v2.0.0
git clone "$UPSTREAM" "$TARGET"
git -C "$TARGET" checkout -q "$COMMIT"
cp -r "$HERE/aerial_gym_overlay/." "$TARGET/"   # aerial_gym/ (code) and resources/robots/x500 (the X500 model)
sed -i 's/warp-lang==1.0.0b5/warp-lang==1.0.0/' "$TARGET/requirements.txt" "$TARGET/setup.py"
pip install -e "$TARGET"
echo "Aerial Gym v2.0.0 with the overlay is installed at $TARGET"
echo "Set AERIAL_GYM_ROOT=$TARGET. Isaac Gym Preview 4 must be installed separately (see README)."
