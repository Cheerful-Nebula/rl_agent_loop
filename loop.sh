#!/bin/bash

MAX_LOOPS=${1:-10}
COUNTER=1
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}Starting Automated Improvement Loop for $MAX_LOOPS iterations...${NC}"

while [ $COUNTER -le $MAX_LOOPS ]
do
    echo -e "\n${BLUE}=========================================="
    echo -e "      ITERATION $COUNTER / $MAX_LOOPS"
    echo -e "==========================================${NC}"

    # 0. TICK THE CLOCK (Single Source of Truth)
    # We use python to increment the state.json file
    python3 -c "from config import Config; i = Config.increment_iteration(); print(f'🕒 State updated to Iteration {i}')"

    # 1. Train the Agent (It will read the NEW iteration number)
    echo -e "${GREEN}[Step 1] Training Agent...${NC}"
    python3 train.py

    # 2. Improve the Code (It will read the SAME iteration number)
    echo -e "${GREEN}[Step 2] Designing New Reward Function...${NC}"
    python3 agent_controller.py

    ((COUNTER++))
    sleep 5
done