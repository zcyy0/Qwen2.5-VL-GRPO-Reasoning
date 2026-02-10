# Geometry-GRPO
This repo studies how **verifiable RL (GRPO)** can improve **visual geometry reasoning** in **Qwen2.5-VL-3B-Instruct**. This project use the following setup:
**Benchmark:** Zebra-CoT (Geometry subset)  
**Model:** Qwen2.5-VL-3B-Instruct 
**Training dataset:** VLAA-Thinking GeoQA170k subset
**Metric:** open-form answer accuracy (final expression/number)



## Roadmap 
- **[x] Stage 0 — Baseline accuracy on the benchmark**
  - Accuracy: 29.1%. Refer to `results/zebra_geometry_baseline_summary.json` for the evaluation summary.
  - scripts/eval_baseline_benchmark.py contains the evaluation code
    
- **[ ]Stage 1 — Open-form targets + verifiable scoring**
  - Convert VLAA-GeoQA ground truth "(A/B/C/D)" to open-form expressions
  - Implement `math_verify`-based equivalence reward

- **[ ]Stage 2 — GRPO training**
  - GRPO on GeoQA-openform (pilot: 1–2k prompts, then full ~6.5k)
  - Track: reward stats, parse rate, length, KL, held-out eval

- **[ ]Stage 3 — Denser reward**
  - Extract 1–3 intermediate checkpoints from `ds_answer`
  - Add partial-credit reward via math_verify
