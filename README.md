# 📐 Visual Geometry Reasoning with Qwen2.5-VL & GRPO

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)
![Tech](https://img.shields.io/badge/Stack-TRL_%7C_VLLM_%7C_LoRA-blue)

## 📌 Project Overview
This project implements **Group Relative Policy Optimization (GRPO)** to enhance **visual geometry reasoning** in the **Qwen2.5-VL-3B-Instruct** model. 

Unlike standard fine-tuning, this pipeline uses Reinforcement Learning (RL) to enforce verifiable "Chain of Thought" (CoT) reasoning. The training system leverages **HuggingFace TRL** for the RL loop, **LoRA** for parameter-efficient tuning, and **VLLM** for high-throughput generation during the exploration phase.

**Target Benchmark:** [MathVision](https://huggingface.co/datasets/mathvision/mathvision)  
**Training Data:** [VLAA-Thinking (GeoQA/Synthesis)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) & [Zebra CoT Geometry](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)

---

## 🔬 Methodology: Structure-Aware Reward Modeling
The goal is to bootstrap the model's reasoning capabilities by strictly enforcing a structured output format:
`<think>...reasoning steps...</think><answer>...final answer...</answer>`

### The Reward Function
To penalize "shortcut learning" (guessing without reasoning) while maintaining training stability, I implemented a hierarchical reward function.

*(Note: If LaTeX does not render, please view in a compatible markdown viewer)*

$$
R(y) = 
\begin{cases} 
1.0 & \text{if Correct Answer AND Strict Format (<think> tags present)} \\
0.5 & \text{if Correct Answer BUT missing <think> tags} \\
0.0 & \text{if Incorrect Answer}
\end{cases}
$$

**Why this matters:**
* **Partial Credit (0.5):** Prevents reward collapse early in training when the model has not yet learned the XML tag structure but gets the answer right.
* **Strict Format (1.0):** Incentivizes the model to shift probability mass toward explicit reasoning chains.

---

## 🛠️ Data Engineering & Curriculum
This project moves beyond simple dataset loading by implementing a **Curriculum Learning** strategy based on problem complexity.

### Data Processing Pipeline
* **Normalization:** Converted VLAA-GeoQA's multiple-choice format `(A/B/C/D)` into open-form expressions.
* **Math Standardization:** Normalized all ground truth values to standard LaTeX math expressions using `math_verify` equivalence checks.
* **Difficulty Stratification:** Classified samples into 3 tiers based on reasoning length and ground-truth complexity:
    * **Tier 1 (Easy):** Direct application of formulas.
    * **Tier 2 (Medium):** Multi-step deduction.
    * **Tier 3 (Hard):** Complex visual grounding required.

### Project Structure
```bash
├── train/
│   └── train_grpo.py           # Main RL training loop (TRL + VLLM integration)
├── scripts/
│   ├── build_splits.py         # Stratified splitting (Train: 5k, Dev: 300, Test: 1k)
│   ├── process_geoqa_data.py   # Multiple-choice to Open-ended conversion
│   └── process_zebra_cot.py    # Geometry dataset cleaning
├── utils/
│   └── extract_answer.py       # Regex logic for answer extraction & normalization
└── README.md
