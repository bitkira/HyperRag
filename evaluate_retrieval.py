import json
import random
import os
import argparse
from collections import defaultdict

random.seed(42)


def evaluate(results, k):
    """
    Calculate pass@k for retrieval results.

    Args:
        results: list of dicts with 'ranks' and 'ground_truth'
        k: int, pass@k position to evaluate

    Returns:
        float, pass@k accuracy
    """
    correct = 0
    for item in results:
        if item['ground_truth'] in item['ranks'][:k]:
            correct += 1
    return correct / len(results) if len(results) > 0 else 0.0


def compute_random_baseline(results, k, num_trials=100):
    """
    Compute random baseline: shuffle candidates for each sample, repeat 100 times and average.

    Args:
        results: list of dicts with 'ranks' and 'ground_truth'
        k: int, pass@k position to evaluate
        num_trials: int, number of random trials (default 100)

    Returns:
        float, average pass@k over random trials
    """
    acc_list = []
    for trial in range(num_trials):
        correct = 0
        for item in results:
            num_candidates = len(item['ranks'])
            random_order = list(range(num_candidates))
            random.shuffle(random_order)
            if item['ground_truth'] in random_order[:k]:
                correct += 1
        acc_list.append(correct / len(results) if len(results) > 0 else 0.0)
    return sum(acc_list) / len(acc_list)


def parse_filename(filename):
    """
    Parse filename to extract metadata.
    Expected format: {language}_{setting}_{difficulty}_k{keep_line}.json

    Example: python_random_hard_k3.json -> (python, random, hard, 3)

    Args:
        filename: str, filename to parse

    Returns:
        tuple: (language, setting, difficulty, keep_line) or None if parse fails
    """
    if not filename.endswith('.json'):
        return None

    parts = filename[:-5].split('_')  # Remove .json and split
    if len(parts) < 4:
        return None

    try:
        # Expected format: language_setting_difficulty_kN
        language = parts[0]
        setting = parts[1]
        difficulty = parts[2]
        keep_line_str = parts[3]

        if not keep_line_str.startswith('k'):
            return None

        keep_line = int(keep_line_str[1:])

        return (language, setting, difficulty, keep_line)
    except (ValueError, IndexError):
        return None


def main(
    dir="results/retrieval/edit",
    k_values=[1, 3, 5],
    random_baseline=True,
    output_dir=None
):
    """
    Evaluate retrieval results from JSON files.

    Args:
        dir: str, directory containing result JSON files
        k_values: list of int, pass@k values to evaluate (e.g., [1, 3, 5] for pass@1, pass@3, pass@5)
        random_baseline: bool, whether to compute random baseline
        output_dir: str, output directory for markdown files (default: replace 'results' with 'evalresults')
    """
    if not os.path.exists(dir):
        print(f"Error: Directory {dir} does not exist")
        return

    # Get all JSON files
    json_files = [f for f in os.listdir(dir) if f.endswith('.json')]

    if len(json_files) == 0:
        print(f"No JSON files found in {dir}")
        return

    print(f"Found {len(json_files)} JSON files in {dir}")

    # Determine output directory
    if output_dir is None:
        # Replace 'results' with 'evalresults' in the path
        output_dir = dir.replace('results', 'evalresults', 1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process each JSON file individually
    for filename in json_files:
        parsed = parse_filename(filename)
        if parsed is None:
            print(f"Warning: Could not parse filename {filename}, skipping")
            continue

        language, setting, difficulty, keep_line = parsed

        filepath = os.path.join(dir, filename)
        with open(filepath, 'r') as f:
            results = json.load(f)

        print(f"Processing {filename}: {len(results)} samples")

        # Compute metrics
        metrics = {}
        for k in k_values:
            acc = evaluate(results, k)
            metrics[f'pass@{k}'] = acc * 100  # Convert to percentage

            if random_baseline:
                print(f"  Computing random baseline for pass@{k}...")
                random_acc = compute_random_baseline(results, k)
                metrics[f'random@{k}'] = random_acc * 100

        # Generate markdown output for this file
        output_filename = filename.replace('.json', '.md')
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, 'w') as f:
            f.write(f"# Retrieval Evaluation: {filename}\n\n")
            f.write(f"**Language**: {language.capitalize()}\n\n")
            f.write(f"**Setting**: {setting}\n\n")
            f.write(f"**Difficulty**: {difficulty}\n\n")
            f.write(f"**Keep lines**: {keep_line}\n\n")
            f.write(f"**Number of samples**: {len(results)}\n\n")
            f.write(f"**Evaluated pass@k values**: {k_values}\n\n")
            f.write(f"**Random baseline**: {'Yes (100 trials)' if random_baseline else 'No'}\n\n")
            f.write("---\n\n")

            # Create table
            f.write("## Results\n\n")
            header = "| Metric |"
            separator = "|--------|"

            for k in k_values:
                header += f" Pass@{k} (%) |"
                separator += "-------------|"

            f.write(header + "\n")
            f.write(separator + "\n")

            # Write pass@k results
            row = "| **Retrieval** |"
            for k in k_values:
                pass_k = metrics.get(f'pass@{k}', 0.0)
                row += f" {pass_k:.2f} |"
            f.write(row + "\n")

            # Write random baseline results
            if random_baseline:
                row = "| **Random Baseline** |"
                for k in k_values:
                    random_acc = metrics.get(f'random@{k}', 0.0)
                    row += f" {random_acc:.2f} |"
                f.write(row + "\n")

            f.write("\n")

        print(f"  ✓ Written to: {output_filepath}")

    print(f"\nAll evaluation results written to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate retrieval results")
    parser.add_argument("--dir", "-d", type=str, default="results/retrieval/edit",
                        help="Directory containing result JSON files")
    parser.add_argument("--k-values", "-k", type=int, nargs='+', default=[1, 3, 5],
                        help="Pass@k values to evaluate (e.g., --k-values 1 3 5 for pass@1, pass@3, pass@5)")
    parser.add_argument("--no-random-baseline", action='store_true',
                        help="Disable random baseline computation")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for markdown files (default: replace 'results' with 'evalresults')")

    args = parser.parse_args()

    main(
        dir=args.dir,
        k_values=args.k_values,
        random_baseline=not args.no_random_baseline,
        output_dir=args.output_dir
    )
