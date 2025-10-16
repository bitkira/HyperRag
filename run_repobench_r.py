from archive_data.utils import load_data, crop_code_lines
from retriever.retriever import retrieve
import json
import os
import argparse

# Optional/soft deps: tqdm, transformers, unixcoder.
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore

try:
    from model.unixcoder import UniXcoder  # type: ignore
except Exception:
    UniXcoder = None  # type: ignore

def main(
        language: str, # language of the data, python or java
        similarity: str, # the similarity used to retrieve, e.g., cosine, edit, jaccard
        keep_line: int, # the number of lines to keep, e.g., 3
        setting: str, # cross_file_first or cross_file_random
        difficulty: str, # easy or hard
        model_name: str = "", # the model used to encode the code, e.g., microsoft/unixcoder-base
        max_length: int = 512, # max length of the code
    ):
    # load the data for the specified setting only
    data = load_data(split="test", task="retrieval", language=language, settings=setting)

    # Extract the test split and specified difficulty
    dic_list = data["test"][difficulty]

    # Default lexical retrieval
    class _SimpleTokenizer:
        """Lightweight tokenizer to avoid heavy deps for lexical similarity.
        Splits on word characters and punctuation; keeps simple alphanumerics.
        """
        def tokenize(self, text, max_length=None, truncation=False):
            import re
            # basic word split; lowercased to make Jaccard/Edit more stable
            tokens = re.findall(r"[A-Za-z0-9_]+|\S", text.lower())
            if max_length is not None and truncation:
                tokens = tokens[:max_length]
            return tokens

    tokenizer = _SimpleTokenizer()
    model = None

    # if semantic retrieval
    if model_name or similarity == "cosine":
        # load the tokenizer and model
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers is required for semantic retrieval but is not installed")
        if not model_name:
            raise ValueError("--model-name must be provided when using cosine similarity")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        if "codegen" in model_name:
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            max_length = 2048
        elif "CodeGPT" in model_name:
            max_length = 512
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "codebert" in model_name:
            max_length = 512
        elif "unixcoder" in model_name:
            max_length = 512
        
        if "unixcoder" in model_name:
            if UniXcoder is None:
                raise ImportError("UniXcoder class not available. Ensure model/unixcoder.py exists or choose another model.")
            model = UniXcoder(model_name)
        else:
            model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        # Device selection: prioritize CUDA, then MPS (Apple Silicon), then CPU
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            model.to(device)
        except Exception:
            pass


    # Process the specified setting and difficulty
    results = []
    for dic in tqdm(dic_list, desc=f"running {setting}/{difficulty}"):
        code = crop_code_lines(dic['code'], keep_line)
        candidates = dic['context']
        ranks = retrieve(
            code=code,
            candidates=candidates,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            similarity=similarity)

        # In data the field name is 'golden_snippet_index'.
        # Fall back to 'gold_snippet_index' if present to be robust to older dumps.
        res_dic = {
            'ranks': ranks,
            'ground_truth': dic.get('golden_snippet_index', dic.get('gold_snippet_index'))
        }
        results.append(res_dic)

    # Write results
    # Construct filename: python_first_easy_k3.json
    setting_short = "first" if "first" in setting else "random"
    filename = f"{language}_{setting_short}_{difficulty}_k{keep_line}.json"

    if model_name:
        os.makedirs(f'results/retrieval/{model_name.split("/")[-1]}', exist_ok=True)
        filepath = f"results/retrieval/{model_name.split('/')[-1]}/{filename}"
    else:
        os.makedirs(f'results/retrieval/{similarity}', exist_ok=True)
        filepath = f"results/retrieval/{similarity}/{filename}"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RepoBench-R retrieval baseline")
    parser.add_argument("--language", "-l", type=str, choices=["python", "java", "py"], required=True,
                        help="Language of the data")
    parser.add_argument("--similarity", "-s", type=str, choices=["edit", "jaccard", "cosine"], required=True,
                        help="Similarity metric to use")
    parser.add_argument("--keep-line", "-k", type=int, required=True,
                        help="Number of lines to keep; e.g., --keep-line 3")
    parser.add_argument("--setting", type=str, choices=["cross_file_first", "cross_file_random"], required=True,
                        help="Setting: cross_file_first or cross_file_random")
    parser.add_argument("--difficulty", "-d", type=str, choices=["easy", "hard"], required=True,
                        help="Difficulty level: easy or hard")
    parser.add_argument("--model-name", "-m", type=str, default="",
                        help="HF model name for semantic retrieval (e.g., microsoft/unixcoder-base)")
    parser.add_argument("--max-length", "-M", type=int, default=512,
                        help="Max tokenized length for encoding")

    args = parser.parse_args()

    main(
        language=args.language,
        similarity=args.similarity,
        keep_line=args.keep_line,
        setting=args.setting,
        difficulty=args.difficulty,
        model_name=args.model_name,
        max_length=args.max_length,
    )

    
