#!/usr/bin/env python3
"""Safely export a 512-token Netflix SLM continuation checkpoint to an HF-style folder.

This prepares the model files for the existing GGUF conversion workflow; it does
not convert to GGUF or change an existing Ollama model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-hf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    original_dir = args.original_hf_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        raise SystemExit(f"STOP: Output folder already exists; will not overwrite: {output_dir}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"STOP: Checkpoint not found: {checkpoint_path}")
    required = (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "modeling_netflix_slm.py",
    )
    missing = [name for name in required if not (original_dir / name).is_file()]
    if missing:
        raise SystemExit(f"STOP: Missing original Hugging Face files: {', '.join(missing)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise SystemExit("STOP: Checkpoint is not a dictionary")
    step = int(checkpoint.get("step", -1))
    weights = checkpoint.get("model_state_dict")
    ckpt_config = checkpoint.get("config")
    if step < 1 or not isinstance(weights, dict) or not weights or not isinstance(ckpt_config, dict):
        raise SystemExit("STOP: Checkpoint lacks valid step, model_state_dict, or config")
    if int(ckpt_config.get("max_seq_len", 0)) != 512:
        raise SystemExit(f"STOP: Expected max_seq_len=512, got {ckpt_config.get('max_seq_len')!r}")

    original_config = json.loads((original_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(original_config, dict):
        raise SystemExit("STOP: Original config.json is not an object")
    config = dict(original_config)
    config.update(ckpt_config)
    # Preserve the exact original model metadata while keeping any alternate
    # context-length fields synchronized with the trained 512-token setting.
    for key in ("max_position_embeddings", "context_length", "seq_length", "n_ctx"):
        if key in config:
            config[key] = 512
    evaluation = checkpoint.get("evaluation") or {}
    config.update({
        "training_steps": step,
        "best_validation_loss": evaluation.get("longchat_validation_loss"),
        "longchat_validation_loss": evaluation.get("longchat_validation_loss"),
        "original_validation_loss": evaluation.get("original_validation_loss"),
        "selection_score": evaluation.get("selection_score"),
        "training_objective": checkpoint.get("training_objective"),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "parameter_count": checkpoint.get("parameter_count"),
    })

    # Use a temporary sibling folder and rename it only after every file is ready.
    temp_dir = output_dir.with_name(output_dir.name + ".partial")
    if temp_dir.exists():
        raise SystemExit(f"STOP: Temporary folder already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    try:
        torch.save(weights, temp_dir / "pytorch_model.bin")
        (temp_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for filename in required:
            if filename != "config.json":
                shutil.copy2(original_dir / filename, temp_dir / filename)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print("EXPORT PREPARATION COMPLETE (GGUF NOT YET CREATED)")
    print("New export folder:", output_dir)
    print("Training step:", step)
    print("Context length:", config["max_seq_len"])
    print("Weight tensors:", len(weights))
    print("Multi-turn validation loss:", evaluation.get("longchat_validation_loss"))
    print("Original checkpoint and Ollama model: UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
