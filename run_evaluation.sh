#!/bin/bash

# Run evaluation for all retrieval results
# Total: 2 directories (edit, jaccard)
# Settings: pass@1, pass@3, pass@5 with random baseline

echo "Starting retrieval evaluation..."
echo "Pass@k values: 1, 3"
echo "Random baseline: enabled (100 trials)"
echo ""

# Evaluate edit similarity results
echo "[1/2] Evaluating edit similarity results..."
uv run python evaluate_retrieval.py \
    --dir results/retrieval/edit \
    --k-values 1 3

echo ""

# Evaluate jaccard similarity results
echo "[2/2] Evaluating jaccard similarity results..."
uv run python evaluate_retrieval.py \
    --dir results/retrieval/jaccard \
    --k-values 1 3

echo ""
echo "All evaluations completed!"
echo ""
echo "Results saved in:"
echo "  - evalresults/retrieval/edit/"
echo "  - evalresults/retrieval/jaccard/"
