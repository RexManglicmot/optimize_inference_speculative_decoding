# app/acceptance_audit_2.py
import argparse, json, os, random
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app.config import cfg
from app.dataset import load_prompts
from app.models import load_models  # returns drafts, verifier


# -------------------------
# Tuple normalization
# -------------------------
def _is_tokenizer(x: Any) -> bool:
    return hasattr(x, "pad_token_id") and hasattr(x, "decode")

def _is_model(x: Any) -> bool:
    return hasattr(x, "generate") or hasattr(x, "forward")

def _norm_tuple(entry: Any, default_id: str) -> Tuple[Any, Any, str]:
    if not isinstance(entry, (list, tuple)):
        raise ValueError(f"Model entry must be tuple-like, got: {type(entry)}")
    elems = list(entry)
    tok = next((e for e in elems if _is_tokenizer(e)), None)
    mdl = next((e for e in elems if _is_model(e)), None)
    mid = next((e for e in elems if isinstance(e, str)), None)
    if mid is None and mdl is not None:
        mid = getattr(mdl, "name_or_path", default_id)
    if tok is None or mdl is None:
        raise ValueError(f"Could not parse tokenizer/model from tuple: {entry}")
    return tok, mdl, str(mid)

def _norm_verifier_tuple(v: Any) -> Tuple[Any, Any, str]:
    return _norm_tuple(v, default_id="verifier")

def _norm_draft_tuple(d: Any) -> Tuple[Any, Any, str]:
    return _norm_tuple(d, default_id="draft")


# -------------------------
# Helpers
# -------------------------
def _set_pad_eos(tok, model):
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tok))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id

def _to_device(t: torch.Tensor, device: torch.device):
    return t.to(device, non_blocking=True)

@torch.inference_mode()
def _verifier_greedy_step(v_model, input_ids, attn_mask, past_key_values=None):
    out = v_model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=True,
        past_key_values=past_key_values,
    )
    logits = out.logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_id, out.past_key_values

@torch.inference_mode()
def _draft_next_token(d_model, input_ids, attn_mask, past_key_values=None, temperature=0.0, top_p=1.0):
    out = d_model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=True,
        past_key_values=past_key_values,
    )
    logits = out.logits[:, -1, :]
    if temperature and temperature > 0.0:
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[..., 0] = True
            filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(filtered, num_samples=1)
            next_id = sorted_idx.gather(1, choice)
        else:
            next_id = torch.multinomial(probs, num_samples=1)
    else:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_id, out.past_key_values

@torch.inference_mode()
def specdec_audit_one(
    v_tok, v_model, d_tok, d_model,
    context_ids: torch.Tensor, context_mask: torch.Tensor,
    max_new_tokens: int, temperature: float, top_p: float
):
    # Verifier-only baseline text (kept only for text artifact parity)
    base_out = v_model.generate(
        input_ids=context_ids,
        attention_mask=context_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=v_tok.pad_token_id,
        eos_token_id=v_tok.eos_token_id,
        use_cache=True,
    )
    baseline_text = v_tok.decode(base_out[0], skip_special_tokens=True)

    # Warm caches on context
    v_out = v_model(input_ids=context_ids, attention_mask=context_mask, use_cache=True)
    v_past = v_out.past_key_values
    d_out = d_model(input_ids=context_ids, attention_mask=context_mask, use_cache=True)
    d_past = d_out.past_key_values

    gen_ids = context_ids.clone()
    gen_mask = context_mask.clone()
    accepts, total = 0, 0

    for _ in range(max_new_tokens):
        d_next, d_past = _draft_next_token(
            d_model, input_ids=gen_ids[:, -1:], attn_mask=gen_mask,
            past_key_values=d_past, temperature=temperature, top_p=top_p
        )
        v_next, v_past_cand = _verifier_greedy_step(
            v_model, input_ids=gen_ids[:, -1:], attn_mask=gen_mask, past_key_values=v_past
        )
        total += 1

        if torch.equal(d_next, v_next):
            accepts += 1
            next_id = d_next
            v_past = v_past_cand
        else:
            next_id = v_next
            v_past = v_past_cand

        gen_ids = torch.cat([gen_ids, next_id], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_id)], dim=1)

        eos_id = v_tok.eos_token_id
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    spec_text = v_tok.decode(gen_ids[0], skip_special_tokens=True)
    acc_rate = accepts / max(1, total)
    return spec_text, baseline_text, acc_rate


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Audit SD: acceptance only (trimmed)")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    device = torch.device(cfg["models"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    temperature = float(cfg["decoding"].get("temperature", 0.7))
    top_p = float(cfg["decoding"].get("top_p", 0.9))
    max_new = int(cfg["decoding"].get("max_new_tokens", 256))

    print("🔎 Loading dataset...")
    prompts = load_prompts()
    if args.shuffle:
        random.Random(args.seed).shuffle(prompts)
    subset_idx = list(range(min(args.sample, len(prompts))))

    print("🧠 Loading models (drafts + verifier)...")
    drafts, verifier = load_models()

    v_tok, v_model, _ = _norm_verifier_tuple(verifier)
    v_model.eval().to(device)
    _set_pad_eos(v_tok, v_model)

    spec_texts_path = "outputs/texts_specdec_2.jsonl"
    base_texts_path = "outputs/texts_baseline_2.jsonl"
    open(spec_texts_path, "w").close()
    open(base_texts_path, "w").close()

    rows = []

    # OUTER progress bar over drafts
    with tqdm(total=len(drafts), desc="Drafts", unit="draft", ncols=100) as outer_bar:
        for draft_entry in drafts:
            d_tok, d_model, d_id = _norm_draft_tuple(draft_entry)
            outer_bar.set_postfix_str(d_id[:40])

            d_model.eval().to(device)
            _set_pad_eos(d_tok, d_model)

            acc_rates = []

            # INNER progress bar over prompts for this draft
            for i in tqdm(subset_idx, desc=f"{d_id}", leave=False, ncols=100):
                item = prompts[i]
                pid = item["id"]
                context_ids = _to_device(item["input_ids"].unsqueeze(0), device)
                context_mask = _to_device(item["attention_mask"].unsqueeze(0), device)

                spec_text, base_text, acc = specdec_audit_one(
                    v_tok, v_model, d_tok, d_model,
                    context_ids, context_mask, max_new, temperature, top_p
                )

                acc_rates.append(acc)

                # keep text artifacts (unchanged)
                with open(spec_texts_path, "a") as fo:
                    fo.write(json.dumps({"id": pid, "draft_id": d_id, "text": spec_text}) + "\n")
                with open(base_texts_path, "a") as fo:
                    fo.write(json.dumps({"id": pid, "draft_id": "BASELINE(verifier_only)", "text": base_text}) + "\n")

            acc_arr = np.array(acc_rates, dtype=float)
            rows.append({
                "draft_id": d_id,
                "samples": len(subset_idx),
                "acceptance_rate_mean": float(np.nanmean(acc_arr)),
                "acceptance_rate_p50": float(np.nanpercentile(acc_arr, 50)),
                "acceptance_rate_p95": float(np.nanpercentile(acc_arr, 95)),
            })

            # free memory between drafts
            if device.type == "cuda":
                d_model.to("cpu")
                torch.cuda.empty_cache()

            outer_bar.update(1)

    out_csv = "outputs/acceptance_audit.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}")
    print(f"📝 Texts: {spec_texts_path}, {base_texts_path}")

if __name__ == "__main__":
    main()
