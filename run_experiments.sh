#!/bin/bash

echo "Running experiment with 'edit' similarity..."
uv run python run_repobench_r.py -l python -s edit -k 3 5 10 20 30 60 120

echo "Running experiment with 'jaccard' similarity..."
uv run python run_repobench_r.py -l python -s jaccard -k 3 5 10 20 30 60 120

echo "All experiments finished."
