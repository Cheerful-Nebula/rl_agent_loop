#!/bin/bash

# 1. Define colors for pretty output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Project Bootstrap...${NC}"

# 2. Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install it."
    exit
fi

# 3. Create a Virtual Environment (The Sandbox)
# This keeps your project libraries separate from your system libraries.
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 4. Activate the Virtual Environment
source venv/bin/activate

# 5. Upgrade pip (Good practice)
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# 6. Install Dependencies
echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# 7. Create Project Directories
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p logs
mkdir -p models
mkdir -p plots

echo -e "${GREEN}Bootstrap Complete! To start working, run:${NC}"
echo -e "source venv/bin/activate"