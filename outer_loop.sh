#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
# Define the contenders
#MODELS=("codegemma:7b" "llama3.1:8b" "qwen2.5-coder:7b" "dolphin-llama3:8b" "wizardlm2:7b" "deepseek-coder:6.7b" ) # Add your models here
MODELS=("gpt-oss:20b")
# Capture arguments
ITERATIONS=${1:-5}     # required-ish: default 5
TIMESTEPS=${2:-50000}  # required-ish: default 50000
TAG="${3:-}"           # optional experiment tag (e.g., testingPrompts, compareLLMs)

# Export global settings for Python scripts
export TOTAL_TIMESTEPS=$TIMESTEPS
# Add the project root to the Python search path
export PYTHONPATH="${PYTHONPATH:-}:."

# Validate input for iterations
if ! [[ "${ITERATIONS}" =~ ^[0-9]+$ ]]; then
  echo "Error: Please provide a valid integer for iterations."
  exit 1
fi
# 🚀 LOGIC SWITCH: Select Controller based on Tag
case "$TAG" in
  *"vision"*)
    echo "👁️  MODE DETECTED: Vision Language Experiment"
    SELECTED_CONTROLLER="controllers/vision.py"
    ;;
  *"agentic"*)
    echo "🤖 MODE DETECTED: Agentic Tools Experiment"
    SELECTED_CONTROLLER="controllers/agentic.py"
    ;;
  *"think"*)
  echo "🤖 MODE DETECTED: Thinking Experiment"
  SELECTED_CONTROLLER="controllers/thinking.py"
  ;;
  *)
    echo "📉 MODE: Standard Text Analysis"
    SELECTED_CONTROLLER="controllers/standard.py"
    ;;
esac
# 🚀 LOGIC SWITCH 2: Select Training Engine based on Tag
# We check if the tag contains the string "remote"
if [[ "$TAG" == *"remote"* ]]; then
    echo "📡 ENGINE: Distributed Training (Mac -> Linux)"
    # This script runs on Mac but talks to Linux
    TRAINING_SCRIPT="train_remote.py"
else
    echo "💻 ENGINE: Local Training (Mac Only)"
    # This script runs the PPO math locally
    TRAINING_SCRIPT="train.py"
fi

# Format steps for directory naming (e.g. 500000 -> 500k, 1200000 -> 1.2M)
format_steps() {
  local n="$1"
  local suffix value

  if (( n >= 1000000 )); then
    # Millions: 1 decimal place, strip trailing .0
    value=$(awk -v n="$n" 'BEGIN { printf "%.1f", n/1000000 }')
    value="${value%\.0}"
    suffix="M"
  elif (( n >= 1000 )); then
    # Thousands: integer k
    value=$(( n / 1000 ))
    suffix="k"
  else
    value="$n"
    suffix=""
  fi

  printf "%s%s" "$value" "$suffix"
}

for model in "${MODELS[@]}"; do
  echo -e "\n\n"
  echo "============================================"
  echo " 🥊 BENCHMARKING MODEL: $model"
  echo "============================================"

  # 1. DEFINE THE CAMPAIGN (The "Outer Loop" Context)
  TIMESTAMP=$(date +%Y-%m-%d)

  # Sanitize model name for filesystem (replace ':' with '-')
  SANITIZED_MODEL=$(echo "$model" | tr ':' '-')

  # Human-readable steps component
  STEP_STR=$(format_steps "$TIMESTEPS")

  # Optional tag suffix
  if [[ -n "$TAG" ]]; then
    TAG_PART="_${TAG}"
  else
    TAG_PART=""
  fi

  # Example final pattern:
  # 2025-12-20_10cycles_500kSteps_LLMcomparison
  CAMPAIGN_TAG="${TIMESTAMP}_${ITERATIONS}cycles_${STEP_STR}Steps${TAG_PART}"

  export CAMPAIGN_TAG
  export LLM_MODEL="$model"

  echo "📂 Target Directory: experiments/$CAMPAIGN_TAG"
  # 2. Generate Initial Reward Function
  echo -e "${GREEN}[Step 0] Generating Base Reward Function...${NC}"
  python3 controllers/initial_shaping.py 

  # 3. RUN THE INNER LOOP
  ./inner_loop.sh "$ITERATIONS" "$SELECTED_CONTROLLER" "$TRAINING_SCRIPT"

  # 4. (Optional) GENERATE SUMMARY PLOT
  # python3 src/plot_campaign_summary.py

  echo "✅ Campaign Complete for $model."
done

echo -e "\n🎉 BENCHMARK SUITE COMPLETE!"
echo "All data is organized in the 'experiments/' directory."
