## Status

1 Folders. DONE


2 Env and dependencies. DONE


3 Prepare Data


4 Config data.

- config.py....SEMI DONE
- dataset.py
- models.py
- benchmarks.py
- eval.py
- viz.py



5 Build App files


6 Outputs generated


7 Documentation


## Inspiration for this Project



## Introduction
Large Language Models (LLMs) enable powerful applications in customer support, healthcare Q&A, and enterprise search. However, they are **slow and expensive to run in production**: each generated token requires a full forward pass through a large model, leading to high latency for users and high compute costs for companies. This makes it difficult to deliver real-time interactions at scale without overspending on infrastructure.

## Speculative Decoding
Speculative decoding is a technique to accelerate LLM inference while preserving output quality. Instead of generating tokens one by one with a large, costly model, several smaller draft models propose multiple tokens in parallel. A larger verifier model then checks these tokens in a single pass:

1) If valid → the tokens are committed all at once.
2) If not → the system falls back to standard decoding.

This mechanism allows speculative decoding to achieve significant latency reductions and lower cost per query, without sacrificing the correctness of the verifier’s outputs.

Business & Technical Benefits
1) Lower latency → real-time responsiveness improves user experience.

2) Reduced cost per query → fewer verifier passes cut GPU/cloud spend.

3) Higher throughput → serve more users with the same infrastructure. 

4) Preserved quality → outputs remain as accurate as the verifier baseline, maintaining trust in critical domains like healthcare and finance.

```text
                           ┌────────────────────────┐
                           │        Prompt          │
                           │ (Question + Abstract)  │
                           └───────────┬────────────┘
                                       │
                                       ▼
        ┌───────────────────────────────────────────────────────────┐
        │   Draft Stage (Sequential, 4 small models, MIN LOC)       │
        │ - Loop over draft models one by one                       │
        │ - Collect a candidate block of k tokens                   │
        │   e.g., ["the","patient","was","treated"]                 │
        └───────────┬───────────────────────────────────────────────┘
                    │ proposals
                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │              Verifier Stage (1 large model)               │
        │ - Single forward pass with verifier                       │
        │ - Check draft block against verifier logits               │
        │   → If consistent → ACCEPT                                │
        │   → If inconsistent → REJECT (fallback decode)            │
        └───────────┬───────────────────────────┬───────────────────┘
                    │                           │
            Accepted path ✔               Rejected path ✘
                    │                           │
     commit k tokens at once                   │ decode next token
     (bulk step forward)                       │ directly with verifier
                    │                           │
                    └──────────────┬────────────┘
                                   ▼
                      ┌───────────────────────────┐
                      │   Updated context/state   │
                      │  (longer generated text)  │
                      └──────────────┬────────────┘
                                     │
                           loop until stop
                        (max tokens or EOS)
                                     ▼
                        ┌────────────────────────┐
                        │        Output          │
                        │  Final generated text  │
                        └──────────┬─────────────┘
                                   ▼
         ┌───────────────────────────────────────────────────────┐
         │               Logging & Metrics (per prompt)          │
         │ - Latency (p50/p95)  - Throughput (tokens/sec)        │
         │ - Speedup vs baseline  - Acceptance rate (%)          │
         │ - Disagreement rate (%) - Cost per Query              │
         └───────────────────────────────────────────────────────┘


```

## Dataset
PubMedQA

Columns
-`id`: PMID number
- `question`:
-`abstract`: Abstract text to generate over

Baseline-Fideliuty Evaluation. No Gold labels

## Models
Verifier Model (large and accurate)
`EleutherAI/gpt-j-6B` a 6B, GPT-2 Tokenizer -- same alignment with smaller models

Draft Models (small and fast)
1)`distilgpt2` (~82M)
2)`gpt2` (~124M)
3)`EleutherAI/gpt-neo-125M`
4)`EleutherAI/gpt-neo-350M`
5)`gpt2-medium` (~355M)


## Metrics (DONE)
1) **Latency** (p50 and p95) 
- Time per generated token/request
- shows accelearation relative to baseline (verifier only


2) **Throughout** (token/sec)
- How many tokens per second (or queries per second) under SD

3) ** VerifierAcceptance Rate (%)**
- What percentage of draft tokens were accepted by the verifier
- Higher leads to more efficiency and less recomputation

4) **Quality / Accuracy Gap**
- Compare outputs vs verifier-only baseline
- Use perplexity, VLEU,ROUGE, or semantic similiarity
Goal is to confirm that speedup does not degrade text quality

5) **Cost/Query**
- Convert throughout gains into dollars per token saved

## Plots
1) Latency Reduction (p50/p95). How fast responses are (median and tail).
2) Throughput (tokens/sec or QPS) How many tokens generated per second.
3) Acceptance Rate (%). How often draft proposals are accepted.
4) Disagreement rate(%). How often SpecDec output ≠ Verifier-only output
5) Speedup Factor (× vs baseline)
6) Summary table showing latencyp50, latencyp95, tokens/sex, speedup, acceptance rate, disagreement rate, Cost/Query

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
        │  - gpt2                                     │
        │  - EleutherAI/gpt-neo-125M                  │
        │  - gpt2-medium                              │
        └──────────────┬──────────────────────────────┘
                       │
     Drafts propose k tokens (e.g., ["cancer", "is", "often", "treated"])
                       │
                       ▼
        ┌─────────────────────────────────────────────┐
        │  Verifier Model (Remote or GPU container)   │
        │  - gpt2-large (example)                     │
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
        │  - Disagreement Rate (%)                    │
        │  - Cost per Query                           │
        └──────────────┬─────────────────



```

## Results



## Limitations



## Next Steps



# AI/ML End-to-End Build Order