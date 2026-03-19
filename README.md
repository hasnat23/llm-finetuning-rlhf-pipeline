# LLM Fine-Tuning & RLHF Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-TRL-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready **end-to-end pipeline** for fine-tuning Large Language Models (LLMs) using **RLHF (Reinforcement Learning from Human Feedback)**, **LoRA**, and **QLoRA**. Supports LLaMA-2, Mistral-7B, and Falcon models for instruction following and domain-specific tasks.

## Overview

This project implements the full RLHF training stack:
1. **Supervised Fine-Tuning (SFT)** — Instruction tuning with LoRA/QLoRA
2. **Reward Model Training** — Learn human preferences from comparison data
3. **PPO Training** — Optimize LLM with proximal policy optimization
4. **DPO (Direct Preference Optimization)** — Simpler alternative to PPO

## Supported Models

| Model | Size | Method | Task |
|---|---|---|---|
| LLaMA-2 | 7B, 13B | QLoRA | Instruction Following |
| Mistral | 7B | LoRA | Code Generation |
| Falcon | 7B | QLoRA | Summarization |
| GPT-2 | 124M | Full Fine-tune | Text Classification |

## Tech Stack

| Component | Tools |
|---|---|
| Base Framework | PyTorch, Transformers |
| PEFT | LoRA, QLoRA (bitsandbytes) |
| RLHF | TRL (SFT Trainer, PPO Trainer, DPO Trainer) |
| Data | Hugging Face Datasets, Alpaca, OpenHermes |
| Tracking | Weights & Biases, TensorBoard |
| Deployment | vLLM, FastAPI, Gradio |

## Project Structure

```
llm-finetuning-rlhf-pipeline/
├── configs/
│   ├── sft_config.yaml          # SFT training config
│   ├── reward_model_config.yaml  # Reward model config
│   └── ppo_config.yaml           # PPO training config
├── data/
│   ├── prepare_dataset.py        # Dataset preprocessing
│   └── preference_data.py        # Human preference data
├── models/
│   ├── reward_model.py           # Reward model architecture
│   └── policy_model.py           # Policy model (LLM)
├── training/
│   ├── sft_trainer.py            # Supervised fine-tuning
│   ├── reward_trainer.py         # Reward model training
│   ├── ppo_trainer.py            # PPO reinforcement learning
│   └── dpo_trainer.py            # DPO training
├── evaluation/
│   ├── benchmark.py              # MT-Bench, HumanEval
│   └── human_eval.py             # Human evaluation scripts
├── deployment/
│   ├── vllm_server.py            # High-throughput inference
│   └── gradio_demo.py            # Interactive demo
├── notebooks/
│   ├── 01_SFT_LLaMA2.ipynb
│   ├── 02_Reward_Model.ipynb
│   └── 03_PPO_Training.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Supervised Fine-Tuning with QLoRA

```bash
python training/sft_trainer.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name tatsu-lab/alpaca \
  --output_dir ./models/llama2-sft \
  --num_epochs 3 \
  --use_4bit True \
  --lora_r 64 \
  --lora_alpha 16
```

### 2. Train Reward Model

```bash
python training/reward_trainer.py \
  --model_name ./models/llama2-sft \
  --dataset_name Anthropic/hh-rlhf \
  --output_dir ./models/reward_model
```

### 3. PPO Training

```bash
python training/ppo_trainer.py \
  --sft_model ./models/llama2-sft \
  --reward_model ./models/reward_model \
  --output_dir ./models/llama2-rlhf
```

## LoRA Configuration

```python
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

## Results

| Model | Method | MT-Bench Score | HumanEval Pass@1 |
|---|---|---|---|
| LLaMA-2-7B Base | None | 4.2 | 12.8% |
| LLaMA-2-7B SFT | QLoRA | 6.1 | 28.4% |
| LLaMA-2-7B RLHF | PPO | 6.8 | 31.2% |
| LLaMA-2-7B RLHF | DPO | 6.9 | 32.1% |

## Author

**Muhammad Hasnat**  
ML & AI Engineer | LLM Fine-Tuning Specialist  
[LinkedIn](https://linkedin.com/in/hasnat23) | [GitHub](https://github.com/hasnat23)

## License

MIT License
