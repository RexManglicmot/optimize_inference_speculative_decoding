# app/models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import cfg

# -------- Device / Precision ----------
DEVICE = cfg["models"].get("device", "cuda")

def _cfg_dtype() -> torch.dtype:
    prec = str(cfg["models"]["verifier"].get("precision", "fp16")).lower()
    if prec in ("bf16", "bfloat16"):
        return torch.bfloat16
    if prec in ("fp32", "float32"):
        return torch.float32
    return torch.float16

DTYPE = _cfg_dtype()

# Optional: allow TF32 on Ampere+ (safe speed)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -------- Helpers ----------
def _resolve_id(entry) -> str:
    """Accept either 'gpt2' or {'id': 'gpt2'}."""
    if isinstance(entry, dict):
        return str(entry.get("id"))
    return str(entry)

def _load_model(model_id: str, dtype: torch.dtype, device: str):
    print(f"🔹 Loading {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    # GPT-2 family often lacks pad token; make it explicit
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Force stable attention path to avoid CUDA asserts with past_key_values
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    # Ensure generate() has a valid pad id
    try:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    except Exception:
        pass

    model.to(device)
    model.eval()
    return model, tok

# -------- Public API ----------
def load_models():
    drafts_cfg = cfg["models"].get("drafts", [])
    drafts = []
    for entry in drafts_cfg:
        mid = _resolve_id(entry)
        model, tok = _load_model(mid, dtype=DTYPE, device=DEVICE)
        # benchmarks.py expects (draft_id, model, tokenizer)
        drafts.append((mid, model, tok))

    v_id = _resolve_id(cfg["models"]["verifier"])
    v_model, v_tok = _load_model(v_id, dtype=DTYPE, device=DEVICE)
    verifier = (v_id, v_model, v_tok)
    return drafts, verifier

if __name__ == "__main__":
    ds, ver = load_models()
    print(f"✅ Loaded {len(ds)} drafts; verifier: {ver[0]}")



# # app/models.py

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from app.config import cfg

# # Speed opts (safe)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.set_float32_matmul_precision("high")

# DEVICE = cfg["models"].get("device", "cuda")
# DTYPE = torch.float16 if cfg["models"]["verifier"].get("precision", "fp16") == "fp16" else torch.float32

# def _attn_impl_for(model_id: str) -> str | None:
#     """Choose an attention implementation per architecture."""
#     lid = model_id.lower()
#     # GPT-Neo and GPT-J: use eager (SDPA not supported yet)
#     if "gpt-neo" in lid or "gpt-j" in lid:
#         return "eager"
#     # GPT-2 family: SDPA is fine on recent PyTorch/Transformers
#     if "gpt2" in lid or "openai-community/gpt2" in lid or "distilgpt2" in lid:
#         # only request SDPA if torch supports it
#         return "sdpa" if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 0) else None
#     # Fallback
#     return None

# def _load_model(model_id: str, dtype=DTYPE, device=DEVICE):
#     tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token

#     attn_impl = _attn_impl_for(model_id)

#     kwargs = dict(
#         torch_dtype=dtype,
#         device_map=device,              # e.g., "cuda" or "cuda:0"
#         low_cpu_mem_usage=True,
#     )
#     if attn_impl is not None:
#         kwargs["attn_implementation"] = attn_impl

#     model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
#     model.eval()
#     return model, tok

# def load_models():
#     drafts_cfg = cfg["models"].get("drafts", [])
#     verifier_cfg = cfg["models"]["verifier"]

#     drafts = []
#     for draft in drafts_cfg:
#         mid = draft["id"]
#         print(f"🔹 Loading draft: {mid}")
#         m, t = _load_model(mid)
#         drafts.append((mid, m, t))

#     v_id = verifier_cfg["id"]
#     print(f"🔹 Loading verifier: {v_id}")
#     v_model, v_tok = _load_model(v_id)

#     return drafts, (v_id, v_model, v_tok)

# if __name__ == "__main__":
#     drafts, verifier = load_models()
#     print("Drafts loaded:", [d[0] for d in drafts])
#     print("Verifier loaded:", verifier[0])
