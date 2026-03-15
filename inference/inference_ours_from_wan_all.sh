#!/usr/bin/env bash
# Run SS4D inference on all WAN-derived samples (run create_ss4d_from_wan_* first).
set -euo pipefail

{
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/inference_ours_from_wan_dahlia.sh"
bash "${SCRIPT_DIR}/inference_ours_from_wan_daisy.sh"
bash "${SCRIPT_DIR}/inference_ours_from_wan_hibiscus.sh"
bash "${SCRIPT_DIR}/inference_ours_from_wan_lily.sh"
bash "${SCRIPT_DIR}/inference_ours_from_wan_rose.sh"

echo "All from_wan SS4D inference done."

exit 0
}
