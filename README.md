## Inspiration for this Project
Inference takes a long time, and when I first started I ran LLMs locally on my MacOS which had a MPS and CPU. Because of such, generating the results took such a long time, and I wanted a learn a way to speed up inference and thus this project. Learning about speculative decoding made me push myself to work with GPUs via the cloud on Vast Ai. Now, I know both about speeding up inference and am comfortable using third party GPUs platforms in projects. 

---

## Introduction
Large Language Models (LLMs) enable powerful applications in customer support, healthcare Q&A, and enterprise search. However, they are **slow and expensive to run in production**: **each generated token requires a full forward pass through a large model**, leading to high latency for users and high compute costs for companies. This makes it difficult to deliver real-time interactions at scale without overspending on infrastructure.

---

## Speculative Decoding
Speculative decoding is a technique to accelerate LLM inference while preserving output quality. Instead of generating tokens one by one with a large and costly model, several smaller draft models propose multiple tokens in parallel. A larger verifier model then checks these tokens in a single pass:

1) If valid → the tokens are committed all at once.
2) If not → the system falls back to standard decoding.

This mechanism allows speculative decoding to achieve significant latency reductions and lower cost per query, without sacrificing the correctness of the verifier’s outputs.

Business & Technical Benefits:
1) **Lower latency** → real-time responsiveness improves user experience.
2) **Reduced cost per query** → fewer verifier passes cut GPU/cloud spend.
3) **Higher throughput** → serve more users with the same infrastructure. 
4) **Preserved quality** → outputs remain as accurate as the verifier baseline, maintaining trust in critical domains like healthcare and finance.

---

## Key Terms

**Draft models (small and fast):**  
  Lightweight language models that quickly generate short blocks of candidate tokens.  
  They are cheaper to run and act as “speculators,” proposing possible continuations of the answer.

**Verifier model (large and accurate):**  
  A larger language model that validates the draft tokens against its own probability distribution.  
  It ensures that the final output matches the quality of the baseline verifier-only system.

**Token block:**  
  A short sequence of tokens (e.g., 4–8) proposed by a draft model at once.  
  Blocks allow the verifier to check multiple tokens in a single pass, rather than generating them one by one.

Intuition:
Draft models = **interns** quickly proposing ideas.  
Verifier = **senior expert** who approves or corrects them.  

The system saves effort because the expert doesn’t need to write every word, just approve or fix the interns’ suggestions.

---

## Example: PubMed QA

**Question:**  
*What are common first-line treatments for non-small cell lung cancer (NSCLC)?*  

**Abstract (excerpt):**  
*Non-small cell lung cancer (NSCLC) is typically managed with a combination of surgery, radiation therapy, and systemic treatments. For resectable early-stage NSCLC, surgical resection remains standard, often followed by adjuvant chemotherapy. In locally advanced disease, concurrent chemoradiation is frequently used. Targeted therapies (e.g., EGFR, ALK inhibitors) and immunotherapies have become first-line options in selected molecularly defined or PD-L1-expressing tumors.*  


### How it works in practice
1. **Draft stage (small models):**  
- We run **four small draft models sequentially** (e.g., `distilgpt2`, `gpt2`, `gpt-neo-125M`, `gpt2-medium`).  
- Each draft is **much cheaper** per forward pass than the large verifier. Their job is to **speculate** and quickly propose a short **block of k tokens** (e.g., k=4–8) that might plausibly continue the answer given the question + abstract.  
- Because the abstract already constrains content (mentions surgery, chemotherapy, radiation, targeted therapy, immunotherapy), the drafts tend to propose medically plausible continuations.  
   Drafts quickly guess a short block of candidate tokens to answer the question, e.g.:  
   Block 1 → `["First-line", "treatment", "often", "includes", "surgery"]`

2. **Verifier stage (large model):**  
   The verifier checks this block in a **single forward pass**:  
   - If the block is **consistent** with its own probability distribution → accept all tokens at once.  
   - If **inconsistent** (e.g., draft says “includes aspirin” which isn’t correct) → reject from that token onward and correct it.  [See below for explanation]

3. **Commit or fallback:**  
   - Accepted → multiple tokens are added at once (*fast*).  
   - Rejected → only the mismatched token is recomputed (*minimal wasted work*).  

4. **Continue with the same question (next block):**  
   After Block 1 is accepted, the system proposes another block, e.g.:  
   Block 2 → `["and", "chemotherapy", "in", "early-stage", "disease"]`  
   If accepted, the answer grows to: *“First-line treatment often includes surgery and chemotherapy in early-stage disease…”*  

   The cycle repeats:  
   - Drafts propose → Verifier checks → Accept or Reject → Commit  
   - Until a stop token, end-of-sequence (EOS), or maximum token limit is reached.  

   Only when the answer is complete does the system move on to the **next Question + Abstract** in the PubMed dataset.  
   

### Example of an **inconsistent** draft block

Continuing the same NSCLC example, suppose **Block 1** is accepted:

> Current answer so far:  
> *“First-line treatment often includes surgery …”*

Now a draft proposes **Block 2**:
- Proposed: `["and", "aspirin", "for", "early-stage", "disease"]`

**Verifier check (single forward pass over the block):**
- Token 1: **“and”** → **OK** (high probability under verifier)  
- Token 2: **“aspirin”** → **MISMATCH** (very low probability; not a first-line NSCLC therapy)

**Decision & correction:**
- Accept the **prefix** up to the last correct token (“and”).  
- **Reject from the first mismatch** (“aspirin”) onward.  
- Verifier **replaces** the wrong token with its own next token: **“chemotherapy”**.  
- The remainder of the draft block is **discarded**.

**Resulting answer after correction:**
> *“First-line treatment often includes surgery **and chemotherapy** …”*

The system then immediately continues **with the same question**, generating the **next block** (e.g., a draft might propose `["and","radiation","therapy","in","locally","advanced","disease"]`).  
This **draft → verify → accept/reject → commit** cycle repeats until the answer is complete (EOS/stop).


### Why Not Let the Verifier check it all?

The large verifier model **can** generate the full answer on its own, but doing so is **slow and costly**. Because it must run a **full forward pass for every single token**. If an answer is ~100 tokens long, that means ~100 expensive passes. **Speculative decoding makes this faster.**


### Caveat

Most of the work is done in `benchmark.py` with the help the Hugging Face `assistant_model` argument path thats provides a fast way to run speculative decoding, but it comes with key limitations:

- **Batch size fixed to 1:** The API forces `batch_size=1`, so it can’t measure acceptance across multiple queries in parallel.  
- **No token-level fidelity info:** It only returns final decoded text. Detailed agreement between drafts and the verifier (needed for acceptance rate) is hidden.  
- **Metrics gap:** Core evaluation metrics like Acceptance Rate (%) require token-by-token verifier probabilities, which the fast path (benchmark.py) doesn’t expose.

**Solution:**  
Built a **separate audit script** `audit_benchmark.py` that replays prompts through the drafts and verifier explicitly. This slower path captures per-token acceptance/disagreement and logs full fidelity metrics, while the main speculative loop remains optimized for generation speed.


---

## Dataset
PubMedQA

Columns
-`id`: PMID number
- `question`:
-`abstract`: Abstract text to generate over

Baseline-Fideliuty Evaluation. No Gold labels

---

## Models

All draft and verifier models used in this project are part of the **GPT-2 family**, which are transformer-based, decoder-only language models trained on large internet text corpora.

They share the **GPT-2 tokenizer**, ensuring consistent tokenization across models of different sizes. Because all models belong to the **GPT-2 family** and use the **same GPT-2 tokenizer**, draft tokens align naturally with the verifier’s token space. This compatibility makes speculative decoding possible without needing additional alignment steps.

### Draft Models (small and fast)
- **distilgpt2 (~82M params)**  
  A distilled, lightweight version of GPT-2; faster and cheaper, with reduced size while retaining much of GPT-2’s performance.  
- **openai-community/gpt2 (~124M params)**  
  The original small GPT-2 model; serves as a baseline draft generator.  
- **openai-community/gpt2-large (~774M params)**  
  A larger GPT-2 variant; still cheaper than verifier but produces more fluent candidate blocks.  
- **openai-community/gpt2-medium (~355M params)**  
  Mid-sized GPT-2; balances speed and quality among the draft set.

### Verifier Model (large and accurate)
- **openai-community/gpt2-xl (~1.5B params)**  
  The largest released GPT-2 model, serving as the **verifier**. It checks draft proposals against its own probability distribution to ensure quality.

---


## Metrics 
1) **Latency** (at p50 and p95, sec/token) 
- Time per generated token/request
- Shows accelearation relative to baseline (verifier only).
- How slow/fast each token comes
- Lower = faster response

2) **Throughout** (at p50 and p95, token/sec)
- How many tokens per second (or queries per second)
- Higher = more overall capacity

3) **Speedup vs baseline** (%)
- It is a ratio of draft to basdline
- How much faster (or slower) you are compared to the baseline.
- Higher leads to faster turnaround

4) **Acceptance Rate** (at p50 and 95, %)
- What percentage of draft tokens were accepted by the verifier
- Higher leads to more efficiency and less recomputation

## Tech Stack
Python
YAML
HuggingFace
numpy
pandas
time
matplotlib
PyYAML
logging

## Workflow
```text
                ┌─────────────────────────────────┐
                │   Dataset: PubMed QA            │
                │  (Questions + Abstract context) │
                └──────────────┬──────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │  Draft Models (Local, 4 small LLMs)         │
        │  - distilgpt2                               │
        │  - gpt2 (~124M params)                      │
        │  - gpt2-medium (~355M params)               │
        │  - gpt2-large (~774M params)                │
        └──────────────┬──────────────────────────────┘
                       │
     Drafts propose k tokens (e.g., ["cancer", "is", "often", "treated"])
                       │
                       ▼
        ┌─────────────────────────────────────────────┐
        │  Verifier Model (GPU container)             │
        │  - gpt2-xl (~1.5B params)                   │
        │  Checks draft tokens against its own logits │
        └──────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴───────────────────┐
         │                                 │
   Tokens accepted ✔                  Tokens rejected ✘
   (verifier agrees)                  (verifier replaces with its own)
         │                                 │
         └─────────────┬───────────────────┘
                       ▼
        ┌─────────────────────────────────────────────┐
        │         Metrics Logging + Evaluation        │
        │  - Latency (p50/p95)                        │
        │  - Throughput (tokens/sec)                  │
        │  - Speedup vs baseline                      │
        │  - Acceptance Rate (%)                      │
        └─────────────────────────────────────────────┘

```



## Results


### Tables
Performance with full 712 rows

| model | latency_p50 (sec/token) | latency_p95 (sec/token) | throughput_p50 (token/sec) | throughput_p95 (token/sec) | speedup (×) |
| --- | --- | --- | --- | --- | --- |
| distilgpt2 | 0.0084 | 0.0112 | 118.9 | 139.2 | 2.44 |
| openai-community/gpt2-medium | 0.0134 | 0.0158 | 74.8 | 82.0 | 1.54 |
| openai-community/gpt2-large | 0.0172 | 0.0195 | 58.0 | 62.3 | 1.19 |
| openai-community/gpt2 | 0.0173 | 0.0218 | 57.7 | 113.8 | 1.19 |
| BASELINE(verifier_only) | 0.0205 | 0.0210 | 48.7 | 49.4 | 1.00 |

Insights:
- `distilgpt`2 is the clear winner: lowest latency (0.0084 s/token), highest throughput (~119 tok/s), and ~2.4× speedup over baseline.
- Bigger draft models bring diminishing returns: `gpt2-medium` (1.54×) and `gpt2-large/gpt2` (~1.2×) are only modestly faster than baseline despite heavier compute.

Performance with half, 356 rows (due to HF)
| draft_id | samples | acceptance_mean | acceptance_rate_p50 | acceptance_rate_p95 |
| --- | --- | --- | --- | --- |
| distilgpt2 | 356 | 80.5% | 84.8% | 93.4% |
| openai-community/gpt2 | 356 | 91.8% | 90.6% | 98.8% |
| openai-community/gpt2-large | 356 | 96.0% | 98.0% | 99.6% |
| openai-community/gpt2-medium | 356 | 95.4% | 96.9% | 99.2% |


Insights:
- Larger draft models align more closely with the verifier, with acceptance rates climb from ~80% (`distilgpt2`) up to ~96% (gpt2-large).
- There is a Trade-off. While small drafts (`distilgpt2`) yield bigger speedups, they come with lower acceptance, meaning the verifier rejects more tokens (less efficient).


### Plots
![Latency p50 vs p95](outputs/latency_grouped_p50_p95.png). How fast responses are (median and tail).

Insights:

- The baseline verifier-only (`gpt2-xl`) is ~0.0205 sec/token and the `distilgpt2` draft, latency drops to ~0.0084 sec/token (≈2.5× faster).
- `distilgpt2` provides the lowest p50 and p95 latencies.
- As draft size increases (`gpt2` → `gpt2-medium` → `gpt2-large`), latency climbs closer to baseline, reducing net benefit.

**Main takeaway**: speculative decoding substantially reduces latency — especially with small, fast draft models like `distilgpt2`, achieving ~2.5× faster token generation compared to verifier-only decoding.

![Throughput p50 vs p95](outputs/throughput_grouped_p50_p95.png). How many tokens generated per second (median and tail).
Insights:

- Baseline (`gpt2-xl`) runs at ~49 tokens/sec (p50), the slowest among all.
- `distilgpt2` achieves the highest throughput (~119 tokens/sec p50, ~139 tokens/sec p95), >2× faster than baseline
- `gpt2-medium` performs at ~75–82 tokens/sec and `gpt2-large` and `gpt2` both hover ~58–62 tokens/sec, both of which are modestly slight better than the baseline. 

**Main takeaway**: Throughput ranking mirrors latency results, smaller drafts deliver more benefit, larger drafts offer diminishing returns. In particular, speculative decoding dramatically boosts throughput when paired with lightweight drafts like `distilgpt2` (≈2.5–3× improvement). Further, higher throughput means more efficient GPU use and more tokens produced per second and a small draft models like `distilgpt2` allow the verifier to accept many tokens in batches which boosted throughput significantly compared to verifier-only runs.


![Speedup vs baseline](outputs/speedup_bar.png)
Insights:

- `distilgpt2` achieves the highest speedup (~2.44×) over the baseline (`gpt2-xl` verifier only).
- `gpt2-medium` gives ~1.54× speedup, still meaningfully faster than baseline, but less than `distilgpt2`.
- `gpt2-large` and `gpt2` drafts show marginal gains (~1.19×) — only ~19% faster than baseline.

**Main takeaway**: As draft size increases, speedup diminishes because larger drafts themselves take longer to propose tokens, eroding speculative decoding’s advantage.


![Acceptance rate](outputs/acceptance_bar.png). How often draft proposals are accepted.
Insights:


- `distilgpt2` has the lowest acceptance rate (~80%) → meaning the verifier rejects ~20% of its draft tokens.
- `GPT2-medium` and `GPT2-large` have the highest acceptance rates (~95–96%), showing they propose tokens very close to what the verifier would generate.
- `GPT2-small` (openai-community/gpt2) sits in between at ~92%

**Main takeaway**: Small drafts like `distilgpt2` give big latency/throughput gains, but the verifier discards more tokens (lower acceptance). Larger drafts (`gpt2-medium` / `gpt2-large`) are slower but achieve very high acceptance, minimizing wasted computation. In sum, choose the draft size depending on outcome: maximum speedup (small draft) or maximum efficiency/accuracy (larger draft).

---

## Conclusion
This project demonstrates that speculative decoding can significantly reduce latency and cost per query while preserving the quality of verifier outputs. By using small draft models to propose candidate continuations and a larger verifier model to validate them, we achieve faster generation without sacrificing correctness. Applied to PubMed QA, this approach highlights how efficiency gains can translate into real-world benefits for biomedical question answering.

---

## Limitations
- **Verifier dependency:** Speculative decoding cannot correct verifier errors; it only speeds up generation while preserving the verifier’s output quality. The accuracy is just as good at the verifier. For the purpse of this experiment, accuracy was omitted.   
- **Hugging Face API constraints:** The `assistant_model` path fixes batch size to 1 and hides per-token details, requiring separate scripts for auditing acceptance and disagreement rates.  

---

## Next Steps

- **Scale to larger datasets:** Move beyond PubMed QA to a bigger biomedical corpus (e.g., full PubMed abstracts or CORD-19) to test performance at scale.  
- **Use larger verifier models:** Replace `GPT-2 XL` with a modern LLM (e.g., `LLaMA-2`, `GPT-J-6B`, or `Falcon`) to evaluate how speculative decoding performs with stronger baselines.  
- **Stress-test real-world applications:** Apply speculative decoding in latency-sensitive domains such as clinical decision support, healthcare chatbots, or biomedical literature search.   

---

## AI/ML End-to-End Build Order