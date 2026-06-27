#!/bin/bash

SEEDS=(15 30)

for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Running experiment with num-seeds=${seed}"
    echo "========================================"

    python3 experiments/run_experiment.py \
        --backend openrouter \
        --config experiments/configs/openrouter_experiment_config.yaml \
        --num-seeds "${seed}" \
        --model-name "qwen/qwen-2.5-72b-instruct"

    echo
done

echo "All experiments completed."========================================

