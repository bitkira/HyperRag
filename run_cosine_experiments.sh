#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

echo "Running experiment with 'cosine' similarity and 'codebert' model..."
uv run python run_repobench_r.py -l python -s cosine -m microsoft/codebert-base -k 3 

# echo "Running experiment with 'cosine' similarity and 'unixcoder' model..."
# uv run python run_repobench_r.py -l python -s cosine -m microsoft/unixcoder-base -k 3 5 10 20 30 60 120

# echo "All cosine experiments finished."
