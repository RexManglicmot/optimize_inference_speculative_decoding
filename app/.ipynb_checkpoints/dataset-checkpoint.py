# app/dataset.py

from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer

from app.config import cfg
from app.prompt import render_prompt

def _get_field_keys():
    ds = cfg["dataset"]
    f = ds.get("text_fields", {})
    return (
        f.get("id", "id"),
        f.get("question", "question"),
        f.get("contexts", "contexts"),  # set to "context" in config if needed
    )

def _maybe_chunk(ids: list[int], window: int, overlap: int) -> list[list[int]]:
    if len(ids) <= window:
        return [ids]
    out, start = [], 0
    while start < len(ids):
        end = min(start + window, len(ids))
        out.append(ids[start:end])
        if end == len(ids):
            break
        start = max(end - overlap, 0)
    return out

def load_prompts() -> List[Dict]:
    """
    Loads ./data/processed/my_data.csv (path from config),
    builds prompts via app/prompt.py, tokenizes with GPT-2 tokenizer,
    applies truncation or optional sliding-window chunking.
    Returns a list of dicts:
      - id: str
      - prompt_text: str
      - input_ids: torch.LongTensor (CPU)
      - attention_mask: torch.LongTensor (CPU)
    """
    ds = cfg["dataset"]
    assert ds.get("source", "csv") == "csv", "Set dataset.source: 'csv' in config.yaml"
    path = ds.get("path", "./data/processed/my_data.csv")

    id_key, q_key, a_key = _get_field_keys()
    max_tokens = int(ds.get("max_input_tokens", 512))
    sample_size = int(ds.get("sample_size", 200))

    chunk_cfg = ds.get("chunking", {"enabled": False})
    chunk_enabled = bool(chunk_cfg.get("enabled", False))
    window_tokens = int(chunk_cfg.get("window_tokens", 896))
    overlap_tokens = int(chunk_cfg.get("overlap_tokens", 128))

    # Read CSV
    df = pd.read_csv(path)
    # Keep only rows with both fields present
    df = df.dropna(subset=[q_key, a_key])

    # Sample (deterministic)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=cfg["project"].get("seed", 42)).reset_index(drop=True)

    # Single tokenizer for alignment across all models (GPT-2 BPE)
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    outputs: List[Dict] = []
    for _, row in df.iterrows():
        ex_id = str(row.get(id_key, ""))
        question = str(row[q_key]).strip()
        abstract = str(row[a_key]).strip()

        prompt_text = render_prompt(question=question, abstract=abstract)

        enc = tok(prompt_text, add_special_tokens=False)
        ids: list[int] = enc["input_ids"]

        if chunk_enabled:
            for idx, chunk_ids in enumerate(_maybe_chunk(ids, window_tokens, overlap_tokens)):
                chunk_ids = chunk_ids[:max_tokens]
                attn = [1] * len(chunk_ids)
                outputs.append({
                    "id": f"{ex_id}::chunk{idx}" if ex_id else f"row{_}::chunk{idx}",
                    "prompt_text": prompt_text,
                    "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                })
        else:
            ids = ids[:max_tokens]
            attn = [1] * len(ids)
            outputs.append({
                "id": ex_id or f"row{_}",
                "prompt_text": prompt_text,
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            })

    return outputs

"""
dataset.py and returns a Python list of dictionaries.

Looks like this:
{
    "id": "12345",                   # str, unique id from your CSV
    "prompt_text": "Q: ... A:",      # str, full rendered prompt (for debugging/logging)
    "input_ids": torch.LongTensor,   # 1D tensor of token IDs (on CPU)
    "attention_mask": torch.LongTensor  # 1D tensor of 1s, same length as input_ids
}

Overall return is:
List[Dict[str, Any]]

The list is returned to whoever calls load_prompts() (e.g., in benchmarks.py).

dataset.py runs entirely on CPU.
- CSV reading: pandas → CPU.
- Prompt rendering: Python strings → CPU.
- Tokenization: transformers.AutoTokenizer → CPU.
- Tensors created: torch.LongTensor → CPU.

This is intentional:
- Tokenization is fast on CPU and doesn’t benefit much from GPU.
- Keeping them on CPU avoids filling GPU memory before you actually run inference.

In benchmarks.py, right before you feed a batch into the draft/verifier models, you move them to GPU:
batch_input_ids = batch_input_ids.to("cuda", non_blocking=True)
batch_attention_mask = batch_attention_mask.to("cuda", non_blocking=True)
"""