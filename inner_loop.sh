#!/bin/bash
set -euo pipefail

# Capture arguments from outer_loop.sh
# MAX_LOOPS comes from the first argument passed by outer_loop.sh
MAX_LOOPS=${1:-10}

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
    # We pass the iteration explicitly. No more guessing or reading state.json.
    echo -e "${GREEN}[Step 1] Training Agent (Iteration $i)...${NC}"
    python3 train.py --iteration "$i"

    # 2. Improve the Code
    # The brain also gets the specific iteration so it knows where to look.
    echo -e "${GREEN}[Step 2] Designing New Reward Function (Iteration $i)...${NC}"
    python3 agent_controller.py --iteration "$i"

    # Optional: Short sleep to let file system buffers flush/sync
    sleep 2
done

echo -e "${GREEN}Inner Loop Complete.${NC}"