from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./outputs/rlhf_final")
tokenizer = None
model = None


def load_model():
    global tokenizer, model
    logger.info(f"Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded successfully")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "instruction" not in data:
        return jsonify({"error": "Missing 'instruction' field"}), 400

    instruction = data["instruction"]
    input_text = data.get("input", "")
    max_new_tokens = data.get("max_new_tokens", 256)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return jsonify({
        "instruction": instruction,
        "input": input_text,
        "response": response,
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
