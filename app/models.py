# app/models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import cfg

# --- GPU & precision settings ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = cfg["models"].get("device", "cuda")
DTYPE = torch.float16 if cfg["models"]["verifier"].get("precision", "fp16") == "fp16" else torch.float32

def _load_model(model_id: str, dtype=DTYPE, device=DEVICE):
    """
    Load a causal LM + tokenizer. Keeps tokenizer consistent (GPT-2 family).
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if torch.__version__ >= "2.0.0" else None,
    )
    model.eval()
    return model, tok

def load_models():
    """
    Returns:
      drafts: list of (id, model, tokenizer)
      verifier: (id, model, tokenizer)
    """
    drafts_cfg = cfg["models"].get("drafts", [])
    verifier_cfg = cfg["models"]["verifier"]

    drafts = []
    for draft in drafts_cfg:
        mid = draft["id"]
        print(f" Loading draft: {mid}")
        m, t = _load_model(mid, dtype=DTYPE, device=DEVICE)
        drafts.append((mid, m, t))

    v_id = verifier_cfg["id"]
    print(f" Loading verifier: {v_id}")
    v_model, v_tok = _load_model(v_id, dtype=DTYPE, device=DEVICE)

    return drafts, (v_id, v_model, v_tok)

if __name__ == "__main__":
    drafts, verifier = load_models()
    print("Drafts loaded:", [d[0] for d in drafts])
    print(" Verifier loaded:", verifier[0])
