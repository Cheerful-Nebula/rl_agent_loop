#!/bin/bash
set -euo pipefail

# 1. Capture arguments from outer_loop.sh
# MAX_LOOPS comes from the first argument passed by outer_loop.sh
MAX_LOOPS=${1:-10}
# 2. Capture the controller script name (default to standard.py)
CONTROLLER_SCRIPT=${2:-"controllers/standard.py"}


GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}Starting Inner Loop for $MAX_LOOPS iterations...${NC}"

# We use a standard Bash C-style loop for cleanliness
for (( i=1; i<=MAX_LOOPS; i++ ))
do
    echo -e "\n${BLUE}=========================================="
    echo -e "      ITERATION $i / $MAX_LOOPS"
    echo -e "==========================================${NC}"

    # 1. Train the Agent
    echo -e "${GREEN}[Step 1] Training Agent (Iteration $i)...${NC}"
    python3 train.py --iteration "$i"

    # 2. Improve the Code
    echo -e "${GREEN}[Step 2] Designing New Reward Function (Iteration $i)...${NC}"
    MODULE_PATH=$(echo "$CONTROLLER_SCRIPT" | sed 's/\//./g' | sed 's/\.py//g')
    python3 -m "$MODULE_PATH" --iteration "$i"

    # Optional: Short sleep to let file system buffers flush/sync
    sleep 2
done

echo -e "${GREEN}Inner Loop Complete.${NC}"