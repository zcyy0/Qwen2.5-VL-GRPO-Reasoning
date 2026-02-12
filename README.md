# Geometry-GRPO
This repo studies how **verifiable RL (GRPO)** can improve **visual geometry reasoning** in **Qwen2.5-VL-3B-Instruct**. This project use the following setup:

- **Benchmark:** MathVision (Geometry Subsets)  
- **Model:** Qwen2.5-VL-3B-Instruct 
- **Training dataset:** VLAA-Thinking GeoQA170k and Synthesis subset + Zebra COT Geometry subset
- **Metric:** open-form answer accuracy (final expression/number)

## Roadmap 
- **[X]Stage 1 — Data Processing + verifiable scoring**
  - Build data processing pipelines that convert VLAA-GeoQA ground truth "(A/B/C/D)" to open-form expressions and normalize ground truth.
  - Implement `math_verify`-based equivalence reward.
  - Classify dataset into 3 different difficulty levels (difficulty=1,2, 3) based on the reasoning length and answer complexity.
  - Set up train/dev/test splits and introduce curriculum learning. 

- **[In Progress ]Stage 2 — GRPO training**
  - GRPO on training data:
    5000 examples, train in three phases. Phase 1 trains on difficulty=1 examples, phase 2 difficulty=2, phase 3 difficulty=3.
    300 dev examples for frequent evaluation during training.
    1000 test examples for evaluation at the end.
  - Reward design: Outcome based reward
  - Track: mean reward per source per difficulty level, accuracy, pass@k rate

- **[ ]Stage 3 — Evaluation on Benchmark**
- **[ ]Stage 4 — Denser reward**
  - Explore partial-credit reward
