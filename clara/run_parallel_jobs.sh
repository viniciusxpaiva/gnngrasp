#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
ENV_NAME="deep-grasp"      # Name of the conda environment to use
PREFIX="teste"          # Prefix for the experiment output name
START=1                    # Start of the experiment index range
END=3                  # End of the experiment index range
SCRIPT="deep-grasp.py"     # Script to run (make sure the name is correct)
# ==========================

echo "Conda environment: $ENV_NAME"
echo "Running script: $SCRIPT"
echo "Jobs to run: ${PREFIX}${START} .. ${PREFIX}${END}"

# Launch experiments in parallel
for i in $(seq "$START" "$END"); do
  name="${PREFIX}${i}"
  echo ">>> Starting ${name}"
  # Run the Python script inside the specified conda environment
  conda run -n "$ENV_NAME" python "$SCRIPT" -o "$name" &
done

# Wait for all background jobs to finish before exiting
wait
echo "All experiments have finished!"
