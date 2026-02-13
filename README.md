# Visual Chain-of-Thought Reasoning with Qwen2.5-VL and GRPO

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Stack](https://img.shields.io/badge/Tech-TRL_%7C_VLLM_%7C_LoRA-blue)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)

## 📌 Project Overview
This project implements **GRPO (Group Relative Policy Optimization)** with verifiable reward to improve **visual geometry reasoning** in **Qwen2.5-VL-3B-Instruct**. The training pipeline is built using **HuggingFace TRL**, **LoRA** for efficient parameter tuning, and **VLLM** for high-throughput rollout generation.

**Target Benchmark:** [MathVision](https://huggingface.co/datasets/mathvision/mathvision)
**Training Data:** [VLAA-Thinking GeoQA and Synthesis subset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [Zebra CoT Geometry subset](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)

## 🔬 Methodology: Structure-Aware Reward Modeling
The goal is to bootstrap the model's reasoning capabilities by strictly enforcing a `<think>` ... `<answer>` structure via Reinforcement Learning (GRPO). The reward function $R(y)$ is designed to penalize "shortcut learning" (guessing the answer without reasoning).
### Reward Function Logic
The reward is assigned based on a hierarchy of constraints:

$$
R(y) = 
\begin{cases} 
1.0 & \text{if Correct Answer AND Strict Format } (\texttt{<think>...<answer>}) \\
0.5 & \text{if Correct Answer BUT Format Violation} \\
0.0 & \text{if Incorrect Answer}
\end{cases}
$$
## 🔮 Future Work & Roadmap
* **Transition to Dense Process Rewards:** Move from format-checking to logic-checking by implementing a heuristic that verifies intermediate steps inside the `<think>` block.



- **Benchmark:** MathVision
- **Model:** Qwen2.5-VL-3B-Instruct 
- **Training dataset:**
  - VLAA-Thinking GeoQA170k and Synthesis subset
  - Zebra COT Geometry subset
- **Metric:** match accuracy + pass@k rate



## Roadmap 
- **[X]Stage 1 — Process data and build verifiable scoring**
  - Build data processing pipelines that convert VLAA-GeoQA multiple choice ground truth "(A/B/C/D)" to open-form expressions, and normalize the datasets' ground truths to standard latext math expressions.
  - Implement `math_verify`-based equivalence reward.
  - Classify dataset into 3 different difficulty levels (difficulty=1,2,3) based on the reference reasoning length and ground truth complexity.
  - Set up train/dev/test splits: training set: 5000 examples; dev set: 300 examples; test set: 1000 examples. Difficulty ratio: 20% difficulty= 3; 60% difficulty=2; 20% difficulty=1.

- **[In Progress ]Stage 2 — GRPO training**
  - Introduce curriculum learning: divide the training to 3 phases. Phase 1 trains on difficulty=1 data; phase 2 trains on difficulty=2 data; phase 3 trains on difficulty=3 data.
  - Prompt requires the model output to follow a strict format: <think>...</think><answer>...</answer>
  - Reward design: Outcome based reward. reward = 1 if the output follows strict format and the answer is correct; reward = 0.5 if the output only has answer tags and the answer is correct; reward = 0 for other cases.
  - Track: mean reward per source per difficulty level, accuracy, pass@k rate

- **[ ]Stage 3 — Evaluation on Benchmark**
- **[ ]Stage 4 — Denser reward**
  - Explore partial-credit reward
