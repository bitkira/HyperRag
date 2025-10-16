#!/bin/bash

# Run RepoBench-R experiments
# Total: 4 experiments
# - Setting: cross_file_random
# - Difficulty: easy
# - Language: python
# - Similarity: edit, jaccard
# - Keep lines: 3, 5

echo "Starting RepoBench-R experiments..."
echo "Total experiments: 4"
echo ""

# Experiment 1: random, easy, python, edit, k=3
echo "[1/4] Running: random, easy, python, edit, k=3"
uv run python run_repobench_r.py \
    -l python \
    -s edit \
    -k 3 \
    --setting cross_file_random \
    --difficulty easy

echo ""

# Experiment 2: random, easy, python, edit, k=5
echo "[2/4] Running: random, easy, python, edit, k=5"
uv run python run_repobench_r.py \
    -l python \
    -s edit \
    -k 5 \
    --setting cross_file_random \
    --difficulty easy

echo ""

# Experiment 3: random, easy, python, jaccard, k=3
echo "[3/4] Running: random, easy, python, jaccard, k=3"
uv run python run_repobench_r.py \
    -l python \
    -s jaccard \
    -k 3 \
    --setting cross_file_random \
    --difficulty easy

echo ""

# Experiment 4: random, easy, python, jaccard, k=5
echo "[4/4] Running: random, easy, python, jaccard, k=5"
uv run python run_repobench_r.py \
    -l python \
    -s jaccard \
    -k 5 \
    --setting cross_file_random \
    --difficulty easy

echo ""
echo "All experiments completed!"
echo ""
echo "Results saved in:"
echo "  - results/retrieval/edit/python_random_easy_k3.json"
echo "  - results/retrieval/edit/python_random_easy_k5.json"
echo "  - results/retrieval/jaccard/python_random_easy_k3.json"
echo "  - results/retrieval/jaccard/python_random_easy_k5.json"
