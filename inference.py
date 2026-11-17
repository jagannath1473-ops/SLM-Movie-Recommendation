import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from modeling_netflix_slm import Qwen3Model

MODEL_DIR = Path(__file__).resolve().parent

with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
    config = json.load(f)

architecture_keys = [
    "vocab_size",
    "d_model",
    "n_heads",
    "n_kv_groups",
    "n_layers",
    "d_ff",
    "max_seq_len"
]

model_config = {
    key: config[key] for key in architecture_keys
}

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True
)

model = Qwen3Model(**model_config)

weights = torch.load(
    MODEL_DIR / "pytorch_model.bin",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights, strict=True)
model.eval()


@torch.inference_mode()
def generate_response(prompt, max_new_tokens=80):

    prompt_text = (
        f"<|user|>\n{prompt}<|end|>\n"
        "<|assistant|>\n"
    )

    input_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=False
    )

    context_length = config["max_seq_len"]

    tokens = torch.tensor(
        [input_ids[-context_length:]],
        dtype=torch.long
    )

    eos_id = tokenizer.convert_tokens_to_ids(
        "<|eos|>"
    )

    generated = []

    for _ in range(max_new_tokens):

        context = tokens[:, -context_length:]

        logits, _ = model(context)

        next_id = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        if next_id == eos_id:
            break

        generated.append(next_id)

        next_token = torch.tensor(
            [[next_id]],
            dtype=torch.long
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Recommend a Netflix comedy movie."
    )
    args = parser.parse_args()

    print("Question:", args.prompt)
    print(
        "Model response:",
        generate_response(args.prompt)
    )
