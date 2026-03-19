import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, SFTTrainer
from peft import LoraConfig

from data_loader import RLHFDataLoader
from model import LLMModel, RewardModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    reward_model_name: str = "meta-llama/Llama-2-7b-hf"
    dataset_name: str = "tatsu-lab/alpaca"
    output_dir: str = "./outputs"
    max_length: int = 512
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    ppo_epochs: int = 4
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_samples: Optional[int] = 10000


def sft_train(config: RLHFConfig):
    """Step 1: Supervised Fine-Tuning (SFT)."""
    logger.info("Starting SFT training...")

    llm = LLMModel(
        model_name=config.model_name,
        use_4bit=config.use_4bit,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    model, tokenizer = llm.load_model()

    data_loader = RLHFDataLoader(config.model_name, config.max_length)
    dataset = data_loader.load_instruction_dataset(
        config.dataset_name, max_samples=config.max_samples
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "sft"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="output",
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, "sft_final"))
    logger.info("SFT training complete.")
    return trainer.model


def ppo_train(config: RLHFConfig, sft_model_path: str):
    """Step 2: PPO RLHF training."""
    logger.info("Starting PPO RLHF training...")

    ppo_config = PPOConfig(
        model_name=sft_model_path,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        log_with="wandb",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
    tokenizer_loader = RLHFDataLoader(config.model_name, config.max_length)
    tokenizer = tokenizer_loader.tokenizer
    reward_model = RewardModel(config.reward_model_name)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    dataset = tokenizer_loader.load_instruction_dataset(
        config.dataset_name, max_samples=config.max_samples
    )

    for epoch in range(config.num_epochs):
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors)
            rewards = [reward_model(r.unsqueeze(0)) for r in response_tensors]
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
        logger.info(f"Epoch {epoch + 1} complete")

    ppo_trainer.save_pretrained(os.path.join(config.output_dir, "rlhf_final"))
    logger.info("PPO RLHF training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--stage", choices=["sft", "ppo", "all"], default="all")
    args = parser.parse_args()

    config = RLHFConfig(model_name=args.model_name, output_dir=args.output_dir)

    if args.stage in ["sft", "all"]:
        sft_model = sft_train(config)

    if args.stage in ["ppo", "all"]:
        ppo_train(config, os.path.join(config.output_dir, "sft_final"))
