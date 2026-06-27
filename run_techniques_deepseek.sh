#!/bin/bash

# Danh sách các giá trị max-techniques cần chạy
for tech in 10
do
    echo "--------------------------------------------------"
    echo "Đang chạy thử nghiệm với --max-techniques $tech..."
    echo "--------------------------------------------------"
    
    python3 experiments/run_experiment.py \
      --backend openrouter \
      --agentic-backend openrouter \
      --model-name deepseek/deepseek-v3.2 \
      --judge-backend openrouter \
      --judge-model deepseek/deepseek-v3.2 \
      --num-seeds 5 \
      --max-techniques $tech
done

echo "Tất cả thử nghiệm đã hoàn thành!"
