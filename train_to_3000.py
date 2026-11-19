#!/usr/bin/env python3
"""Continue Netflix-SLM-V2 from step 1000 with a boundary-safe SFT objective."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

TEST_PROMPTS = [
    "Recommend a Netflix comedy movie.",
    "What is the plot of Stranger Things?",
    "Is The Crown a movie or a TV show?",
    "Recommend a romantic TV series on Netflix.",
    "Recommend an Indian thriller movie.",
    "What genre is Money Heist?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"C:\Users\jagan\SLM-Movie-Recommendation"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "run_3000")
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--target-step", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--correction-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_conversation(example: dict) -> bool:
    text = " ".join(str(message.get("content", "")) for message in example["messages"])
    if "\ufffd" in text or "????" in text:
        return False
    lowered = text.lower()
    conflict_titles = ("stranger things", "the crown", "money heist")
    conflict_phrases = ("isn't in my data", "couldn't find", "don't have any record", "not in my netflix dataset")
    if any(title in lowered for title in conflict_titles) and any(phrase in lowered for phrase in conflict_phrases):
        return False
    return True


def format_message(role: str, content: str) -> str:
    ending = "<|eos|>" if role == "assistant" else "<|end|>"
    return f"<|{role}|>\n{content or ''}{ending}\n"


def make_items(examples: list[dict], tokenizer, max_seq_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    items: list[tuple[torch.Tensor, torch.Tensor]] = []
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    pad_id = tokenizer.pad_token_id
    full_limit = max_seq_len + 1
    max_response = min(96, full_limit - 24)

    for example in examples:
        messages = example["messages"]
        for assistant_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix = "".join(format_message(m["role"], m.get("content", "")) for m in messages[:assistant_index])
            prefix += "<|assistant|>\n"
            prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
            response_ids = tokenizer.encode(message.get("content", ""), add_special_tokens=False) + [eos_id]
            if len(response_ids) > max_response:
                response_ids = response_ids[: max_response - 1] + [eos_id]
            prompt_budget = full_limit - len(response_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
            sequence = prompt_ids + response_ids
            response_start = len(prompt_ids)

            x = sequence[:-1]
            y = sequence[1:]
            labels = [token if index + 1 >= response_start else -100 for index, token in enumerate(y)]
            padding = max_seq_len - len(x)
            x += [pad_id] * padding
            labels += [-100] * padding
            items.append((torch.tensor(x, dtype=torch.int32), torch.tensor(labels, dtype=torch.int32)))
    return items


def import_model_class(repo: Path):
    source = repo / "huggingface_netflix_slm_v2" / "modeling_netflix_slm.py"
    spec = importlib.util.spec_from_file_location("modeling_netflix_slm", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.Qwen3Model


def sample_batch(items, batch_size: int, rng: random.Random, device: torch.device):
    selected = [items[rng.randrange(len(items))] for _ in range(batch_size)]
    x = torch.stack([item[0] for item in selected]).to(device=device, dtype=torch.long, non_blocking=True)
    y = torch.stack([item[1] for item in selected]).to(device=device, dtype=torch.long, non_blocking=True)
    return x, y


@torch.inference_mode()
def evaluate_loss(model, items, batch_size: int, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for offset in range(0, len(items), batch_size):
        batch = items[offset: offset + batch_size]
        x = torch.stack([item[0] for item in batch]).to(device=device, dtype=torch.long)
        y = torch.stack([item[1] for item in batch]).to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")
        count = int((y != -100).sum().item())
        weighted_loss += float(loss_sum.item())
        token_count += count
    model.train()
    return weighted_loss / token_count


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, device: torch.device, max_seq_len: int, seed: int = 42) -> str:
    system = "You answer Netflix questions concisely and factually."
    text = f"<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    ids = tokenizer.encode(text, add_special_tokens=False)[-max_seq_len:]
    tokens = torch.tensor([ids], device=device, dtype=torch.long)
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    generator = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    model.eval()
    for _ in range(80):
        context = tokens[:, -max_seq_len:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(context)
        scores = logits[0, -1].float()
        for token_id in set(generated):
            scores[token_id] = scores[token_id] / 1.1 if scores[token_id] > 0 else scores[token_id] * 1.1
        scores = scores / 0.8
        top_values, top_indices = torch.topk(scores, 40)
        probs = torch.softmax(top_values, dim=-1)
        sorted_probs, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > 0.95] = 0
        sorted_probs /= sorted_probs.sum()
        chosen = torch.multinomial(sorted_probs, 1, generator=generator)
        next_id = int(top_indices[order[chosen]].item())
        if next_id in (eos_id, end_id):
            break
        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    model.train()
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def capture_responses(model, tokenizer, device, max_seq_len: int) -> list[dict]:
    return [{"prompt": prompt, "response": generate(model, tokenizer, prompt, device, max_seq_len)} for prompt in TEST_PROMPTS]


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA support.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_checkpoint = args.start_checkpoint or args.repo / "models" / "netflix_slm_v2_latest.pt"
    tokenizer_dir = args.repo / "netflix_tokenizer_8k"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    checkpoint = torch.load(start_checkpoint, map_location="cpu", weights_only=True)
    start_step = int(checkpoint["step"])
    if start_step >= args.target_step:
        raise ValueError(f"checkpoint is already at step {start_step}")

    Model = import_model_class(args.repo)
    model = Model(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    train_rows = [row for row in load_jsonl(args.repo / "data" / "train.jsonl") if clean_conversation(row)]
    validation_rows = [row for row in load_jsonl(args.repo / "data" / "validation.jsonl") if clean_conversation(row)]
    correction_rows = load_jsonl(Path(__file__).resolve().parent / "verified_corrections.jsonl")
    train_items = make_items(train_rows, tokenizer, checkpoint["config"]["max_seq_len"])
    validation_items = make_items(validation_rows, tokenizer, checkpoint["config"]["max_seq_len"])
    correction_items = make_items(correction_rows, tokenizer, checkpoint["config"]["max_seq_len"])
    eval_rng = random.Random(args.seed)
    eval_items = [validation_items[eval_rng.randrange(len(validation_items))] for _ in range(min(args.eval_samples, len(validation_items)))]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_lr = float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["initial_lr"] = args.learning_rate
    continuation_steps = args.target_step - start_step

    def lr_factor(relative_step: int) -> float:
        if relative_step < args.warmup_steps:
            start_factor = start_lr / args.learning_rate
            return start_factor + (1.0 - start_factor) * (relative_step / max(1, args.warmup_steps))
        progress = (relative_step - args.warmup_steps) / max(1, continuation_steps - args.warmup_steps)
        minimum_factor = args.minimum_learning_rate / args.learning_rate
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    rng = random.Random(args.seed)
    model.train()

    if args.benchmark_only:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.benchmark_steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.gradient_accumulation):
                x, y = sample_batch(train_items, args.batch_size, rng, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = model(x, y)
                (loss / args.gradient_accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_step = elapsed / args.benchmark_steps
        print(json.dumps({
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "batch_size": args.batch_size,
            "benchmark_steps": args.benchmark_steps, "seconds_per_step": seconds_per_step,
            "estimated_seconds_for_remaining_steps": seconds_per_step * continuation_steps,
            "train_items": len(train_items), "validation_items": len(validation_items),
            "correction_items": len(correction_items),
        }, indent=2))
        return 0

    source_info = {
        "path": str(start_checkpoint.resolve()), "sha256": sha256(start_checkpoint),
        "step": start_step, "legacy_best_validation_loss": checkpoint.get("best_val_loss"),
    }
    (args.output_dir / "source_checkpoint.json").write_text(json.dumps(source_info, indent=2), encoding="utf-8")
    shutil.copy2(start_checkpoint, args.output_dir / "original_step_1000_backup.pt")

    baseline_loss = evaluate_loss(model, eval_items, 64, device)
    baseline_responses = capture_responses(model, tokenizer, device, checkpoint["config"]["max_seq_len"])
    (args.output_dir / "baseline_responses.json").write_text(json.dumps(baseline_responses, indent=2), encoding="utf-8")

    history = []
    best_loss = baseline_loss
    latest_path = args.output_dir / "netflix_slm_v3_step3000_latest.pt"
    best_path = args.output_dir / "netflix_slm_v3_step3000_best.pt"
    started = time.perf_counter()

    for global_step in range(start_step + 1, args.target_step + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(args.gradient_accumulation):
            source = correction_items if rng.random() < args.correction_ratio else train_items
            x, y = sample_batch(source, args.batch_size, rng, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated += float(loss.item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()

        should_eval = global_step % args.eval_interval == 0 or global_step == args.target_step
        if should_eval:
            val_loss = evaluate_loss(model, eval_items, 64, device)
            record = {
                "step": global_step, "train_loss": accumulated / args.gradient_accumulation,
                "assistant_validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": grad_norm, "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            payload = {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "config": checkpoint["config"],
                "step": global_step, "best_val_loss": min(best_loss, val_loss), "history": history,
                "training_objective": "assistant_only_boundary_safe_sft",
                "source_checkpoint": source_info, "correction_ratio": args.correction_ratio,
            }
            torch.save(payload, latest_path)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(payload, best_path)
            print(json.dumps(record), flush=True)

    selected_path = best_path if best_path.exists() else latest_path
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    final_responses = capture_responses(model, tokenizer, device, checkpoint["config"]["max_seq_len"])
    (args.output_dir / "updated_responses.json").write_text(json.dumps(final_responses, indent=2), encoding="utf-8")
    summary = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "start_step": start_step, "target_step": args.target_step,
        "baseline_assistant_validation_loss": baseline_loss, "best_assistant_validation_loss": best_loss,
        "selected_checkpoint": str(selected_path), "elapsed_seconds": time.perf_counter() - started,
        "train_items": len(train_items), "validation_items": len(validation_items),
        "correction_items": len(correction_items), "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
