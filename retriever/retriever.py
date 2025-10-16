from transformers import AutoTokenizer, AutoModel
from retriever.similarity import edit_similarity, jaccard_similarity, cosine_similarity
import torch
from model.unixcoder import UniXcoder


def retrieve(
    code,
    candidates: list,
    tokenizer = None,
    model = None,
    max_length: int = None,
    similarity: str = "jaccard"
    ):
    
    # check if the similarity is valid
    assert similarity in ["edit", "jaccard", "cosine"], "similarity must be one of edit, jaccard, cosine"

    if similarity == "cosine":
        assert model, "model must be provided if similarity is cosine"

        def get_embeddings_batch(codes):
            """Batch encode multiple code snippets for better GPU utilization"""
            if isinstance(model, UniXcoder):
                device = model.model.device
                hidden_size = model.config.hidden_size
            else:
                device = model.device
                hidden_size = model.config.hidden_size

            # Handle empty codes
            non_empty_codes = [(i, c) for i, c in enumerate(codes) if c]
            if not non_empty_codes:
                return [torch.zeros(hidden_size).to(device) for _ in codes]

            indices, valid_codes = zip(*non_empty_codes)

            if isinstance(model, UniXcoder):
                # UniXcoder batch processing
                tokens_ids = model.tokenize(list(valid_codes), max_length=max_length, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(device)
                with torch.no_grad():
                    _, batch_embeddings = model(source_ids)  # [batch_size, hidden_size]
            else:
                # Standard transformers batch processing
                encoded = tokenizer(
                    list(valid_codes),
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    if attention_mask is not None:
                        outputs = model(input_ids, attention_mask=attention_mask)
                    else:
                        outputs = model(input_ids)

                    # Get last hidden state and mean pool
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    else:
                        hidden_states = outputs[0]

                    # Mean pooling with attention mask
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = sum_embeddings / sum_mask
                    else:
                        batch_embeddings = torch.mean(hidden_states, dim=1)

            # Reconstruct full embedding list with zeros for empty codes
            all_embeddings = []
            valid_idx = 0
            for i, c in enumerate(codes):
                if c:
                    all_embeddings.append(batch_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    all_embeddings.append(torch.zeros(hidden_size).to(device))

            return all_embeddings

        # Batch encode all codes (query + candidates) together for maximum efficiency
        all_codes = [code] + candidates
        all_embeddings = get_embeddings_batch(all_codes)

        code_embedding = all_embeddings[0]
        candidates_embeddings = all_embeddings[1:]

        # calculate the cosine similarity between the code and the candidates
        sim_scores = []
        for i, candidate_embedding in enumerate(candidates_embeddings):
            sim_scores.append((i, cosine_similarity(code_embedding, candidate_embedding)))
    else:
        # candidates is a list of code strings
        # we need to sort the candidate index based on the edit similarity in a descending order
        sim_scores = []
        for i, candidate in enumerate(candidates):
            if similarity == "edit":
                sim_scores.append((i, edit_similarity(code, candidate, tokenizer)))
            elif similarity == "jaccard":
                sim_scores.append((i, jaccard_similarity(code, candidate, tokenizer)))
        
    
    # sort the candidate index based on the edit similarity in a descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # only return the index
    ranks = [ index for index, score in sim_scores ]

    return ranks


def retrieve_batch(
    codes: list,
    candidates_list: list,
    tokenizer = None,
    model = None,
    max_length: int = None,
    similarity: str = "jaccard"
    ):
    """
    Batch version of retrieve function for processing multiple queries at once.

    Args:
        codes: list of code strings (queries)
        candidates_list: list of candidate lists, one for each query
        tokenizer: tokenizer for encoding
        model: model for encoding (required for cosine similarity)
        max_length: maximum sequence length
        similarity: similarity metric (edit, jaccard, cosine)

    Returns:
        list of ranks, one for each query
    """

    # check if the similarity is valid
    assert similarity in ["edit", "jaccard", "cosine"], "similarity must be one of edit, jaccard, cosine"
    assert len(codes) == len(candidates_list), "codes and candidates_list must have the same length"

    if similarity == "cosine":
        assert model, "model must be provided if similarity is cosine"

        # Prepare all codes for batch encoding
        all_codes_flat = []
        code_indices = []  # Track which code each embedding belongs to
        candidate_counts = []  # Track number of candidates per query

        for query_idx, (code, candidates) in enumerate(zip(codes, candidates_list)):
            all_codes_flat.append(code)
            code_indices.append(query_idx)

            for candidate in candidates:
                all_codes_flat.append(candidate)
                code_indices.append(query_idx)

            candidate_counts.append(len(candidates))

        # Batch encode everything at once
        if isinstance(model, UniXcoder):
            device = model.model.device
            hidden_size = model.config.hidden_size
        else:
            device = model.device
            hidden_size = model.config.hidden_size

        # Handle empty codes
        non_empty_codes = [(i, c) for i, c in enumerate(all_codes_flat) if c]
        if not non_empty_codes:
            # If all codes are empty, return empty ranks
            return [[] for _ in codes]

        indices, valid_codes = zip(*non_empty_codes)

        if isinstance(model, UniXcoder):
            # UniXcoder batch processing
            tokens_ids = model.tokenize(list(valid_codes), max_length=max_length, mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(device)
            with torch.no_grad():
                _, batch_embeddings = model(source_ids)  # [batch_size, hidden_size]
        else:
            # Standard transformers batch processing
            encoded = tokenizer(
                list(valid_codes),
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                if attention_mask is not None:
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)

                # Get last hidden state and mean pool
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0]

                # Mean pooling with attention mask
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                else:
                    batch_embeddings = torch.mean(hidden_states, dim=1)

        # Reconstruct full embedding list with zeros for empty codes
        all_embeddings = []
        valid_idx = 0
        for i, c in enumerate(all_codes_flat):
            if c:
                all_embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                all_embeddings.append(torch.zeros(hidden_size).to(device))

        # Split embeddings back to queries and candidates
        results = []
        embed_idx = 0

        for query_idx, num_candidates in enumerate(candidate_counts):
            query_embedding = all_embeddings[embed_idx]
            embed_idx += 1

            candidate_embeddings = all_embeddings[embed_idx:embed_idx + num_candidates]
            embed_idx += num_candidates

            # Calculate similarities for this query
            sim_scores = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                sim_scores.append((i, cosine_similarity(query_embedding, candidate_embedding)))

            # Sort and extract ranks
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            ranks = [index for index, score in sim_scores]
            results.append(ranks)

        return results

    else:
        # For lexical similarity, still process individually (already fast)
        results = []
        for code, candidates in zip(codes, candidates_list):
            ranks = retrieve(code, candidates, tokenizer, model, max_length, similarity)
            results.append(ranks)
        return results