#!/bin/bash

# Define the contenders
MODELS=("dolphincoder:7b" )
# Capture two arguments: iterations and timesteps
ITERATIONS=${1:-5}
TIMESTEPS=${2:-20000}

# Export the variable so ALL child scripts (loop.sh and python scripts) see it
export TOTAL_TIMESTEPS=$TIMESTEPS

if ! [[ "$1" =~ ^[0-9]+$ ]] && [ ! -z "$1" ]; then
    echo "Error: Please provide a valid integer for iterations."
    exit 1
fi

# Create the results directory if it doesn't exist
mkdir -p benchmark_results

for model in "${MODELS[@]}"; do
    echo -e "\n\n"
    echo "============================================"
    echo "   🥊  BENCHMARKING MODEL: $model"
    echo "============================================"
    
    # 1. CLEAN SLATE
    # We pipe 'yes' to confirm the reset
    echo "yes" | python3 reset_experiment.py
    
    # 2. SET ENVIRONMENT VARIABLE
    # Config.py will read this
    export LLM_MODEL=$model
    
    # 3. RUN THE LOOP
    # We assume loop.sh runs for a set number of times or we control it here.
    # Ideally, modify loop.sh to take an argument, or just let it run its default.
    # For now, let's assume loop.sh runs 10 times or whatever is hardcoded.
    ./loop.sh "$ITERATIONS"
    
    # 4. GENERATE THE PLOT
    # We must run this explicitly because loop.sh doesn't do it!
    echo "📊 Generating progress report..."
    python3 plot_progress.py
    
    # 5. ARCHIVE RESULTS
    echo "💾 Saving results for $model..."
    
    # Move the plot
    if [ -f "progress_report.png" ]; then
        mv progress_report.png "benchmark_results/progress_$model.png"
    else
        echo "⚠️  Warning: No progress_report.png found."
    fi
    
    # Move the videos folder
    if [ -d "logs/videos" ]; then
        # We rename the folder to include the model name
        mv logs/videos "benchmark_results/videos_$model"
    else
        echo "⚠️  Warning: No video logs found."
    fi

    # Move the Reasoning History
    # This preserves every thought process for side-by-side comparison
    if [ -d "logs/reasoning_history" ]; then
        mv logs/reasoning_history "benchmark_results/reasoning_$model"
    else
        echo "⚠️  Warning: No reasoning logs found."
    fi

    # Move the Code History
    # This preserves code generation steps for side-by-side comparison
    if [ -d "logs/code_history" ]; then
        mv logs/code_history "benchmark_results/code_$model"
    else
        echo "⚠️  Warning: No coding logs found."
    fi

    # Move the metrics History
    if [ -d "logs/metrics_history" ]; then
        mv logs/metrics_history "benchmark_results/metrics_$model"
    else
        echo "⚠️  Warning: No metrics logs found."
    fi

    # Move the SB3 log History
    if [ -d "logs/sb3_log_history" ]; then
        mv logs/sb3_log_history "benchmark_results/sb3_log_$model"
    else
        echo "⚠️  Warning: No SB3 logs found."
    fi

    echo "✅ Completed run for $model"
done

echo -e "\n🎉 BENCHMARK SUITE COMPLETE!"
echo "Results are stored in the 'benchmark_results' folder."