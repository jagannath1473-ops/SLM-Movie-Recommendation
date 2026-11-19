"""Convert the trained Netflix SLM checkpoint to standard Hugging Face Qwen3.

This is a key-for-key format conversion. It does not train, resize, initialize,
merge, quantize, or otherwise substitute model weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM


SPECIAL_TOKEN_IDS = {
    "pad_token_id": 0,
    "unk_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 3,
}


def expected_source_keys(n_layers: int) -> set[str]:
    keys = {"tok_emb.weight", "final_norm.weight", "out_head.weight"}
    suffixes = (
        "norm1.weight",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.o_proj.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
        "norm2.weight",
        "ffn.gate.weight",
        "ffn.up.weight",
        "ffn.down.weight",
    )
    for layer in range(n_layers):
        keys.update(f"blocks.{layer}.{suffix}" for suffix in suffixes)
    return keys


def build_name_map(n_layers: int) -> dict[str, str]:
    mapping = {
        "tok_emb.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "out_head.weight": "lm_head.weight",
    }
    layer_map = {
        "norm1.weight": "input_layernorm.weight",
        "attn.q_proj.weight": "self_attn.q_proj.weight",
        "attn.k_proj.weight": "self_attn.k_proj.weight",
        "attn.v_proj.weight": "self_attn.v_proj.weight",
        "attn.o_proj.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "norm2.weight": "post_attention_layernorm.weight",
        "ffn.gate.weight": "mlp.gate_proj.weight",
        "ffn.up.weight": "mlp.up_proj.weight",
        "ffn.down.weight": "mlp.down_proj.weight",
    }
    for layer in range(n_layers):
        for source_suffix, target_suffix in layer_map.items():
            mapping[f"blocks.{layer}.{source_suffix}"] = (
                f"model.layers.{layer}.{target_suffix}"
            )
    return mapping


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_source_config(config: dict) -> None:
    required = {
        "vocab_size",
        "d_model",
        "n_heads",
        "n_kv_groups",
        "n_layers",
        "d_ff",
        "max_seq_len",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Source config is missing: {sorted(missing)}")
    if config["d_model"] % config["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if config["n_heads"] % config["n_kv_groups"]:
        raise ValueError("n_heads must be divisible by n_kv_groups")


def make_qwen3_config(source: dict) -> Qwen3Config:
    head_dim = source["d_model"] // source["n_heads"]
    return Qwen3Config(
        vocab_size=source["vocab_size"],
        hidden_size=source["d_model"],
        intermediate_size=source["d_ff"],
        num_hidden_layers=source["n_layers"],
        num_attention_heads=source["n_heads"],
        num_key_value_heads=source["n_kv_groups"],
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=source["max_seq_len"],
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=SPECIAL_TOKEN_IDS["pad_token_id"],
        bos_token_id=SPECIAL_TOKEN_IDS["bos_token_id"],
        eos_token_id=SPECIAL_TOKEN_IDS["eos_token_id"],
    )


def copy_tokenizer(source_dir: Path, output_dir: Path) -> None:
    # tokenizer.json is copied byte-for-byte; vocabulary and IDs cannot drift.
    for filename in ("tokenizer.json", "chat_template.jinja"):
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / filename)

    source_tokenizer_config = load_json(source_dir / "tokenizer_config.json")
    tokenizer_config = dict(source_tokenizer_config)
    tokenizer_config.update(
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 512,
            "clean_up_tokenization_spaces": False,
        }
    )
    with (output_dir / "tokenizer_config.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(tokenizer_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
        ],
    }
    with (output_dir / "special_tokens_map.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(special_tokens_map, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def convert(source_dir: Path, output_dir: Path) -> None:
    source_config = load_json(source_dir / "config.json")
    validate_source_config(source_config)

    checkpoint = torch.load(
        source_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    expected = expected_source_keys(source_config["n_layers"])
    actual = set(checkpoint)
    if actual != expected:
        raise ValueError(
            "Checkpoint keys do not match the inspected architecture. "
            f"Missing={sorted(expected - actual)}; unexpected={sorted(actual - expected)}"
        )
    if not torch.equal(checkpoint["tok_emb.weight"], checkpoint["out_head.weight"]):
        raise ValueError("Source embedding and output-head weights are not tied/equal")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_qwen3_config(source_config)
    model = Qwen3ForCausalLM(config)
    name_map = build_name_map(source_config["n_layers"])
    converted = {name_map[name]: tensor for name, tensor in checkpoint.items()}

    result = model.load_state_dict(converted, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    model.tie_weights()
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("Target output head and embedding are not tied")

    # Save in float32 to preserve every source value exactly. GGUF conversion can
    # later choose f16/Q8 explicitly; this canonical HF export remains lossless.
    model.eval()
    model.save_pretrained(output_dir, safe_serialization=True)
    copy_tokenizer(source_dir, output_dir)
    GenerationConfig(
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
    ).save_pretrained(output_dir)

    manifest = {
        "source": str(source_dir.resolve()),
        "format": "standard Hugging Face Qwen3ForCausalLM",
        "source_tensor_count": len(checkpoint),
        "target_parameter_count": sum(p.numel() for p in model.parameters()),
        "tie_word_embeddings": True,
        "rope_theta": 10000.0,
        "dtype": "float32",
        "weight_transformations": "none; names remapped only",
    }
    with (output_dir / "conversion_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Converted model saved to: {output_dir.resolve()}")
    print(f"Parameters: {manifest['target_parameter_count']:,}")
    print("Weight operation: key rename only (no retraining or tensor transforms)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.source_dir.resolve(), args.output_dir.resolve())
