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


## Speculative Decoding



## Business Case / So What?



## Dataset
From Kaggle website, CORD-19 metadata.csv

Columns
-`title` 
-`abstract`, text to generate over


Preprocessing
- Drop missing rows with missing abstracts
- Sample 100-500 entries


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


2) **Throughout** (token/sec or QPS)
- How many tokens per second (or queries per second) under SD

3) **Acceptance Rate (%)**
- What percentage of draft tokens were accepted by the verifier
- Higher leads to more efficiency and less recomputation

4) **Quality / Accuracy Gap**
- Compare outputs vs verifier-only baseline
- Use perplexity, VLEU,ROUGE, or semantic similiarity
Goal is to confirm that speedup does not degrade text quality

5) **Cost Estimate**
- Convert throughout gains into dollars per token saved

## Plots
1)Latency Reduction (p50/p95)
2) Throughput (tokens/sec or QPS)
3) Acceptance Rate (%)
4) Scatter plot (draft models as points).
5) Table

Optional
5) Quality Gap (Perplexity / BLEU)
6) Quality Gap (Perplexity / BLEU)

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
Local Drafts: Run locally (no issue).

Local drafts: run in your machine’s RAM/VRAM → results get exported to `outputs/`.

HF verifier: runs remotely on Hugging Face’s Inference Serverless → responses come back over API → you aggregate and save them in outputs/ too.


Verifier: Use HF Inference Serverless — this avoids the hardware bottleneck, keeps your project reproducible on any laptop, and looks resume-ready (“Benchmarked speculative decoding pipelines using Hugging Face Inference API verifiers…”).
HF verifier: runs remotely on Hugging Face’s Inference Serverless → responses come back over API → you aggregate and save them in `outputs/` too.

This hybrid setup is common in production: cheap/light components run locally or on low-tier hardware, heavy inference is offloaded to a hosted API.

We run draft models locally, but the verifier model (gpt-j-6B) via Hugging Face Inference Serverless, for speed, reproducibility, and minimal LOC.
## Results



## Limitations



## Next Steps



# AI/ML End-to-End Build Order