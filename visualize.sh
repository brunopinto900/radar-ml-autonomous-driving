#!/usr/bin/env bash
# Load a RadarScenes sequence and open it in the official rad_viewer GUI.
# Usage: ./visualize.sh [sequence_number]   (default: 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ_NUM="${1:-1}"
SCENES_JSON="$SCRIPT_DIR/data/RadarScenes/RadarScenes/data/sequence_${SEQ_NUM}/scenes.json"

if [[ ! -f "$SCENES_JSON" ]]; then
    echo "No such sequence: $SCENES_JSON" >&2
    exit 1
fi

source "$SCRIPT_DIR/.radar_ml/bin/activate"

# Needed on WSL2 without WSLg: point at the third-party X server on the Windows host.
export DISPLAY="${DISPLAY:-172.28.176.1:0}"

exec rad_viewer "$SCENES_JSON"