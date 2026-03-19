import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
import logging

logger = logging.getLogger(__name__)


class LLMModel:
    """Wrapper for LLM with LoRA/QLoRA support."""

    def __init__(
        self,
        model_name: str,
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device_map = device_map
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load base model with optional quantization."""
        logger.info(f"Loading model: {self.model_name}")

        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Base model loaded successfully")
        return self.model, self.tokenizer

    def apply_lora(self, target_modules=None):
        """Apply LoRA adapters to the model."""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA adapters applied")
        return self.model

    def get_model_info(self):
        """Return model parameter info."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 4),
        }


class RewardModel(nn.Module):
    """Reward model for RLHF training."""

    def __init__(self, base_model_name: str):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto"
        )
        hidden_size = self.base.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
