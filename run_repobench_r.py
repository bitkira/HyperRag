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
        keep_lines: list, # the lines to keep, e.g., [3, 10]
        model_name: str = "", # the model used to encode the code, e.g., microsoft/unixcoder-base
        max_length: int = 512, # max length of the code
    ):
    # load the data
    settings = ["cross_file_first", "cross_file_random"]
    data_first, data_random = load_data(split="test", task="retrieval", language=language, settings=settings)

    data_first = data_first["test"]
    data_random = data_random["test"]

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
        # keep original behavior; cosine mode requires device, but we do not
        # force CUDA if not available. Fallback to CPU.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        except Exception:
            pass

    
    
    mapping = {
        "first": data_first,
        "random": data_random
    }

    for setting, dataset in mapping.items():
        res = {}
        i = 0
        for key, dic_list in dataset.items():
            res[key] = []
            for dic in tqdm(dic_list, desc=f"running {key}"):
                res_dic = {}
                for i in keep_lines:
                    code = crop_code_lines(dic['code'], i)
                    candidates = dic['context']
                    res_dic[i] = retrieve(
                        code=code,
                        candidates=candidates, 
                        tokenizer=tokenizer,
                        model=model, 
                        max_length=max_length,
                        similarity=similarity)
                
                # In data the field name is 'golden_snippet_index'.
                # Fall back to 'gold_snippet_index' if present to be robust to older dumps.
                res_dic['ground_truth'] = dic.get('golden_snippet_index', dic.get('gold_snippet_index'))
                res[key].append(res_dic)
        
        # write
        if model_name:
            os.makedirs(f'results/retrieval/{model_name.split("/")[-1]}', exist_ok=True)
            with open(f"results/retrieval/{model_name.split('/')[-1]}/{language}_{setting}.json", "w") as f:
                json.dump(res, f, indent=4)
        else:
            os.makedirs(f'results/retrieval/{similarity}', exist_ok=True)
            with open(f"results/retrieval/{similarity}/{language}_{setting}.json", "w") as f:
                json.dump(res, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RepoBench-R retrieval baseline")
    parser.add_argument("--language", "-l", type=str, choices=["python", "java", "py"], required=True,
                        help="Language of the data")
    parser.add_argument("--similarity", "-s", type=str, choices=["edit", "jaccard", "cosine"], required=True,
                        help="Similarity metric to use")
    parser.add_argument("--keep-lines", "-k", type=int, nargs='+', required=True,
                        help="Lines to keep; e.g., --keep-lines 3 10")
    parser.add_argument("--model-name", "-m", type=str, default="",
                        help="HF model name for semantic retrieval (e.g., microsoft/unixcoder-base)")
    parser.add_argument("--max-length", "-M", type=int, default=512,
                        help="Max tokenized length for encoding")

    args = parser.parse_args()

    main(
        language=args.language,
        similarity=args.similarity,
        keep_lines=args.keep_lines,
        model_name=args.model_name,
        max_length=args.max_length,
    )

    
