import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class RLHFDataLoader:
    """Data loader for RLHF fine-tuning pipeline."""

    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded: {model_name}")

    def load_instruction_dataset(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """Load instruction-following dataset."""
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    def format_prompt(self, example: Dict) -> str:
        """Format example into instruction-following prompt."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""

        def tokenize_fn(examples):
            prompts = [
                self.format_prompt(
                    {
                        "instruction": inst,
                        "input": inp,
                        "output": out,
                    }
                )
                for inst, inp, out in zip(
                    examples["instruction"],
                    examples.get("input", [""] * len(examples["instruction"])),
                    examples["output"],
                )
            ]
            return self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        logger.info(f"Tokenized {len(tokenized)} examples")
        return tokenized

    def load_preference_dataset(self, path: str) -> Dataset:
        """Load human preference dataset for reward model training."""
        with open(path, "r") as f:
            data = json.load(f)
        return Dataset.from_list(data)


if __name__ == "__main__":
    loader = RLHFDataLoader("meta-llama/Llama-2-7b-hf", max_length=512)
    dataset = loader.load_instruction_dataset(max_samples=1000)
    print(f"Dataset loaded: {len(dataset)} examples")
    print(dataset[0])
