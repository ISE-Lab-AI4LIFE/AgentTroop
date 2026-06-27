#!/bin/bash

SEEDS=(1 10 15 20 25 30)

for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Running experiment with num-seeds=${seed}"
    echo "========================================"

    python3 experiments/run_experiment.py \
        --backend openrouter \
        --config experiments/configs/openrouter_experiment_config.yaml \
        --num-seeds "${seed}" \
        --model-name "deepseek/deepseek-v3.2"

    echo
done

echo "All experiments completed."========================================

