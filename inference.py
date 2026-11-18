"""Run the trained Netflix SLM without changing its checkpoint or architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from transformers import PreTrainedTokenizerFast

from modeling_netflix_slm import Qwen3Model


MODEL_DIR = Path(__file__).resolve().parent
ARCHITECTURE_KEYS = (
    "vocab_size",
    "d_model",
    "n_heads",
    "n_kv_groups",
    "n_layers",
    "d_ff",
    "max_seq_len",
)
SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
}
ROLE_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant providing details on movies and TV shows."
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def load_runtime(model_dir: Path, device: torch.device):
    with (model_dir / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)

    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise KeyError(f"Missing architecture settings: {', '.join(missing)}")

    # Load tokenizer.json directly. This preserves the trained vocabulary while
    # avoiding Transformers trying to resolve the custom model_type.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        **SPECIAL_TOKENS,
        additional_special_tokens=list(ROLE_TOKENS),
    )

    expected_ids = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    actual_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in expected_ids
    }
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Tokenizer special-token IDs do not match the trained checkpoint: "
            f"expected {expected_ids}, found {actual_ids}"
        )

    model = Qwen3Model(
        **{key: config[key] for key in ARCHITECTURE_KEYS}
    ).to(device)
    weights = torch.load(
        model_dir / "pytorch_model.bin",
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]

    model.load_state_dict(weights, strict=True)
    model.eval()
    return config, tokenizer, model


def build_prompt_ids(
    prompt: str,
    tokenizer: PreTrainedTokenizerFast,
    context_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[int]:
    """Match V2 training format while always retaining the role markers."""
    if system_prompt:
        prefix_text = (
            f"<|system|>\n{system_prompt.strip()}<|end|>\n"
            "<|user|>\n"
        )
    else:
        prefix_text = "<|user|>\n"

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(
        "<|end|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    content_ids = tokenizer.encode(prompt.strip(), add_special_tokens=False)
    content_budget = context_length - len(prefix_ids) - len(suffix_ids)
    if content_budget <= 0:
        raise RuntimeError("The configured context is too short for chat markers.")

    # Long prompts retain their ending, but the user/assistant structure is
    # never truncated away.
    content_ids = content_ids[-content_budget:]
    return prefix_ids + content_ids + suffix_ids


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Iterable[int],
    penalty: float,
) -> None:
    if penalty == 1.0:
        return
    for token_id in set(generated_ids):
        value = logits[token_id]
        logits[token_id] = value * penalty if value < 0 else value / penalty


def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_ids) + 1 < ngram_size:
        return set()
    if ngram_size == 1:
        return set(generated_ids)

    prefix_size = ngram_size - 1
    current_prefix = tuple(generated_ids[-prefix_size:])
    banned: set[int] = set()
    for start in range(len(generated_ids) - ngram_size + 1):
        ngram = generated_ids[start : start + ngram_size]
        if tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return banned


def filter_top_k_top_p(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        k = min(top_k, filtered.numel())
        cutoff = torch.topk(filtered, k).values[-1]
        filtered[filtered < cutoff] = -torch.inf

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_logits[remove] = -torch.inf
        filtered = torch.full_like(filtered, -torch.inf)
        filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


@torch.inference_mode()
def generate_response(
    prompt: str,
    *,
    config: dict,
    tokenizer: PreTrainedTokenizerFast,
    model: Qwen3Model,
    device: torch.device,
    max_new_tokens: int = 80,
    min_new_tokens: int = 2,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.12,
    no_repeat_ngram_size: int = 3,
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, str]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be between 0 and max_new_tokens.")
    if temperature < 0:
        raise ValueError("temperature cannot be negative; use 0 for greedy decoding.")
    if top_k < 0:
        raise ValueError("top_k cannot be negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1.0.")

    context_length = int(config["max_seq_len"])
    prompt_ids = build_prompt_ids(
        prompt,
        tokenizer,
        context_length,
        system_prompt=system_prompt,
    )
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []

    stop_tokens = (
        "<|eos|>",
        "<|end|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|pad|>",
    )
    stop_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in stop_tokens
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    stop_reason = "max_new_tokens"

    for step in range(max_new_tokens):
        context = tokens[:, -context_length:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :].float().clone()

        apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
        for token_id in banned_ngram_tokens(
            generated_ids,
            no_repeat_ngram_size,
        ):
            next_logits[token_id] = -torch.inf

        if step < min_new_tokens:
            for token_id in stop_ids:
                next_logits[token_id] = -torch.inf

        if temperature == 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            filtered = filter_top_k_top_p(next_logits, top_k, top_p)
            probabilities = torch.softmax(filtered, dim=-1)
            if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_id = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    ).item()
                )

        if next_id in stop_ids:
            token_text = tokenizer.convert_ids_to_tokens(next_id)
            stop_reason = f"stop_token:{token_text}"
            break

        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        tokens = torch.cat((tokens, next_token), dim=1)

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return response, stop_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a response with the trained Netflix SLM checkpoint."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Recommend a Netflix comedy movie.",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0 selects greedy decoding; values above 0 enable sampling.",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Defaults to the most common system prompt in the training data.",
    )
    parser.add_argument("--show-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    config, tokenizer, model = load_runtime(args.model_dir, device)
    response, stop_reason = generate_response(
        args.prompt,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )

    print("Question:", args.prompt)
    print("Model response:", response or "[empty response]")
    if args.show_details:
        mode = "greedy" if args.temperature == 0 else "sampling"
        print(
            "Generation:",
            f"mode={mode}, temperature={args.temperature}, top_k={args.top_k}, "
            f"top_p={args.top_p}, repetition_penalty={args.repetition_penalty}, "
            f"no_repeat_ngram_size={args.no_repeat_ngram_size}, seed={args.seed}",
        )
        print("Device:", device)
        print("Stop reason:", stop_reason)


if __name__ == "__main__":
    main()
