import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate fine-tuned LLM models on instruction-following tasks."""

    def __init__(self, model_path: str, device: str = "auto"):
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores."""
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = self.rouge.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)
        return {k: float(np.mean(v)) for k, v in scores.items()}

    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute corpus BLEU score."""
        smoothing = SmoothingFunction().method1
        refs = [[r.split()] for r in references]
        hyps = [p.split() for p in predictions]
        return corpus_bleu(refs, hyps, smoothing_function=smoothing)

    def evaluate_dataset(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        split: str = "test",
        max_samples: int = 200,
    ) -> Dict:
        """Evaluate model on a dataset split."""
        logger.info(f"Evaluating on {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        predictions, references = [], []
        for example in dataset:
            instruction = example["instruction"]
            input_text = example.get("input", "")
            reference = example["output"]

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            pred = self.generate(prompt)
            predictions.append(pred)
            references.append(reference)

        rouge_scores = self.compute_rouge(predictions, references)
        bleu = self.compute_bleu(predictions, references)

        results = {
            "num_samples": len(predictions),
            "rouge": rouge_scores,
            "bleu": bleu,
        }
        logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_file", default="eval_results.json")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path)
    results = evaluator.evaluate_dataset(args.dataset, args.split, args.max_samples)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")
