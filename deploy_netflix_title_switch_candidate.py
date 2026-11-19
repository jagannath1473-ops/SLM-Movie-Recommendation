#!/usr/bin/env python3
r"""Deploy the measured-improved title-switch 40.38M Netflix SLM candidate to Ollama.

One command on the user's Windows PC (no Work mode required):
  & "$HOME\SLM-Movie-Recommendation\.venv\Scripts\python.exe" "$HOME\SLM-Movie-Recommendation\deploy_netflix_title_switch_candidate.py"

Reads title_switch_repair_run/best.pt; reuses the ALREADY-CONVERTED Qwen3 HF
architecture/tokenizer from longchat_512_qwen3; writes a separate Qwen3 HF
folder and GGUF; registers a NEW Ollama model, preserving all existing models.
Runs identical multi-turn Ollama questions on OLD and NEW models and saves both responses.
No model training. No pretrained model substituted. No internet downloads.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from urllib import error, request

HOME = Path.home()
REPO = HOME / "SLM-Movie-Recommendation"
SOURCE = REPO / "title_switch_repair_run" / "best.pt"
BASE_HF = REPO / "longchat_512_qwen3"
OUTPUT = REPO / "title_switch_repair_qwen3"
GGUF = OUTPUT / "netflix-slm-40m-title-switch-f32.gguf"
MODELFILE = OUTPUT / "Modelfile.title-switch"
MODEL_NAME = "netflix-slm-40m-title-switch-candidate"
OLD_MODEL_NAME = "netflix-slm-40m-dialogue-v2-fixed"
CONVERTER = (HOME / "Documents" / "Codex" / "2026-11-18" / "new-chat"
             / "work" / "llama.cpp" / "convert_hf_to_gguf.py")
MANIFEST = OUTPUT / "deployment_manifest.json"
REPORT = OUTPUT / "ollama_candidate_comparison.json"
SYSTEM = "You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation."


def announce(message: str) -> None:
    print(message, flush=True)


def require_file(path: Path, explanation: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {explanation}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_key(key: str) -> str | None:
    if key == "tok_emb.weight":
        return "model.embed_tokens.weight"
    if key == "final_norm.weight":
        return "model.norm.weight"
    if key == "out_head.weight":
        return None  # tied to tok_emb.weight; reference file decides whether to include lm_head
    m = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
    if not m:
        raise RuntimeError(f"Unrecognized checkpoint tensor: {key}")
    layer, suffix = m.groups()
    prefix = f"model.layers.{layer}."
    direct = {
        "norm1.weight": "input_layernorm.weight",
        "norm2.weight": "post_attention_layernorm.weight",
        "attn.q_proj.weight": "self_attn.q_proj.weight",
        "attn.k_proj.weight": "self_attn.k_proj.weight",
        "attn.v_proj.weight": "self_attn.v_proj.weight",
        "attn.o_proj.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "ffn.gate.weight": "mlp.gate_proj.weight",
        "ffn.up.weight": "mlp.up_proj.weight",
        "ffn.down.weight": "mlp.down_proj.weight",
    }
    if suffix not in direct:
        raise RuntimeError(f"Unknown tensor suffix: {key}")
    return prefix + direct[suffix]


def export_checkpoint():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(f"Missing Python dependency: {exc}. Use your project's .venv Python.") from exc

    require_file(SOURCE, "fine-tuned checkpoint")
    require_file(BASE_HF / "model.safetensors", "previously working Qwen3 reference model")
    require_file(BASE_HF / "config.json", "reference Qwen3 config")
    require_file(BASE_HF / "tokenizer.json", "reference tokenizer")
    require_file(CONVERTER, "previously used llama.cpp converter")
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama command is not available. Start Ollama Desktop and check PATH.")

    source_hash = sha256(SOURCE)
    if OUTPUT.exists():
        if not MANIFEST.is_file():
            raise RuntimeError(f"Output folder already exists but has no manifest. To avoid overwriting, stopped: {OUTPUT}")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_hash:
            raise RuntimeError("Checkpoint differs from existing export. Refusing to mix checkpoints.")
        require_file(OUTPUT / "model.safetensors", "existing exported model")
        announce("Existing export matches checkpoint; reusing it.")
        return manifest

    announce("Loading saved title-switch BEST checkpoint (read only)...")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise RuntimeError("Checkpoint lacks model_state_dict/config.")
    if int(config.get("max_seq_len", -1)) != 512:
        raise RuntimeError(f"Unexpected context length: {config.get('max_seq_len')}")
    if int(checkpoint.get("parameter_count", 0)) != 40_384_512:
        raise RuntimeError("This is not the expected 40,384,512-parameter model.")
    if not torch.equal(state["tok_emb.weight"], state["out_head.weight"]):
        raise RuntimeError("Tied embedding weights do not match; stopping.")

    checkpoint_step = int(checkpoint.get("step", -1))
    if checkpoint_step <= 0:
        raise RuntimeError("Invalid checkpoint step; refusing export")

    mapped = {}
    for key, tensor in state.items():
        new_key = map_key(key)
        if new_key is not None:
            if new_key in mapped:
                raise RuntimeError(f"Duplicate mapped tensor: {new_key}")
            mapped[new_key] = tensor.detach().cpu().contiguous()
    del checkpoint, state

    with safe_open(str(BASE_HF / "model.safetensors"), framework="pt", device="cpu") as sf:
        reference_keys = set(sf.keys())
        reference_shapes = {key: tuple(sf.get_slice(key).get_shape()) for key in reference_keys}
    if "lm_head.weight" in reference_keys:
        mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"].clone()
    if set(mapped) != reference_keys:
        missing = sorted(reference_keys - set(mapped))
        extra = sorted(set(mapped) - reference_keys)
        raise RuntimeError(f"HF tensor-key mismatch: missing={missing[:10]}, unexpected={extra[:10]}")
    for key, tensor in mapped.items():
        if tuple(tensor.shape) != reference_shapes[key]:
            raise RuntimeError(f"Shape mismatch for {key}: {tuple(tensor.shape)} vs {reference_shapes[key]}")

    # Create in a temporary sibling directory, then atomically rename.
    temp = OUTPUT.with_name(OUTPUT.name + ".partial")
    if temp.exists():
        raise RuntimeError(f"Previous incomplete export exists: {temp}. Stop rather than overwrite.")
    temp.mkdir(parents=True)
    try:
        for item in BASE_HF.iterdir():
            if not item.is_file():
                continue
            if item.name.endswith((".safetensors", ".gguf", ".bin")) or item.name.startswith("Modelfile"):
                continue
            if item.name.endswith(".safetensors.index.json"):
                continue
            shutil.copy2(item, temp / item.name)
        config_path = temp / "config.json"
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        if hf_config.get("model_type") != "qwen3":
            raise RuntimeError("Reference is not a Qwen3-compatible export.")
        hf_config["max_position_embeddings"] = 512
        hf_config["tie_word_embeddings"] = True
        hf_config["finetuned_checkpoint_step"] = int(checkpoint_step)
        hf_config["finetuning_objective"] = "natural_title_switch_sft_experiment"
        config_path.write_text(json.dumps(hf_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        announce(f"Saving {len(mapped)} verified Qwen3 tensors...")
        # Explicitly clone tensors only if necessary for safetensors storage ownership.
        save_file({k: v.contiguous().clone() for k, v in mapped.items()}, str(temp / "model.safetensors"), metadata={"format": "pt"})
        manifest = {
            "source_checkpoint": str(SOURCE),
            "source_sha256": source_hash,
            "checkpoint_step": checkpoint_step,
            "parameter_count": 40_384_512,
            "context_length": 512,
            "hf_reference": str(BASE_HF),
            "mapped_tensor_count": len(mapped),
            "note": "Real user-trained 40.38M model, not a pretrained Qwen model; Qwen3 is only the compatible file architecture.",
        }
        (temp / MANIFEST.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    announce("Checkpoint safely mapped to Qwen3 HF format. Existing files unchanged.")
    return manifest


def run_logged(command: list[str], logfile: Path, label: str) -> None:
    announce(label)
    with logfile.open("w", encoding="utf-8") as out:
        result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = "\n".join(logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {result.returncode}).\nLast log lines:\n{tail}")


def convert_gguf():
    if GGUF.is_file() and GGUF.stat().st_size > 50_000_000:
        announce("Existing GGUF found; reusing it.")
        return
    if GGUF.exists():
        raise RuntimeError(f"Incomplete GGUF exists: {GGUF}; stopped to avoid overwriting.")
    run_logged([sys.executable, str(CONVERTER), str(OUTPUT), "--outfile", str(GGUF), "--outtype", "f32"],
               OUTPUT / "gguf_conversion.log", "Converting actual fine-tuned checkpoint to F32 GGUF...")
    if not GGUF.is_file() or GGUF.stat().st_size < 50_000_000:
        raise RuntimeError("GGUF converter did not produce a valid-size output.")
    announce(f"GGUF created: {GGUF} ({GGUF.stat().st_size / 1e6:.1f} MB)")


def register_ollama():
    # Match the exact custom role tokens used during SFT. Do not rely on the
    # stock Qwen3 template: the trained model uses <|system|>, <|user|>, etc.
    template = '''{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|user|>
{{ .Content }}<|end|>
{{ else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|eos|>
{{ end }}{{ end }}<|assistant|>
'''
    modelfile_text = (
        f"FROM ./{GGUF.name}\n"
        + 'TEMPLATE """' + template + '"""\n'
        + 'SYSTEM "' + SYSTEM + '"\n'
        + 'PARAMETER num_ctx 512\n'
        + 'PARAMETER num_predict 96\n'
        + 'PARAMETER temperature 0.15\n'
        + 'PARAMETER top_k 20\n'
        + 'PARAMETER top_p 0.85\n'
        + 'PARAMETER repeat_penalty 1.08\n'
        + 'PARAMETER stop "<|eos|>"\n'
        + 'PARAMETER stop "<|end|>"\n'
        + 'PARAMETER stop "<|user|>"\n'
        + 'PARAMETER stop "<|assistant|>"\n'
    )
    MODELFILE.write_text(modelfile_text, encoding="utf-8")
    run_logged(["ollama", "create", MODEL_NAME, "-f", str(MODELFILE)],
               OUTPUT / "ollama_create.log", f"Registering candidate as {MODEL_NAME}...")
    result = subprocess.run(["ollama", "show", MODEL_NAME, "--template"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Cannot verify registered Ollama chat template")
    (OUTPUT / "ollama_template.txt").write_text(result.stdout, encoding="utf-8")
    if "<|assistant|>" not in result.stdout or "<|user|>" not in result.stdout:
        raise RuntimeError("Registered Ollama template lacks expected custom role tokens")
    announce(f"Ollama candidate created: {MODEL_NAME} (original model unchanged)")


def chat_once(messages: list[dict], model_name: str) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 512, "num_predict": 80, "temperature": 0.15, "seed": 20261119},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request("http://127.0.0.1:11434/api/chat", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=180) as resp:
        result = json.load(resp)
    return result.get("message", {}).get("content", "").strip()


def test_ollama():
    # A deliberately small, repeatable *manual-review* test. We do not claim
    # statistical accuracy from these few examples. New and old receive the
    # same conversation and identical generation settings.
    cases = [
        ["Who directed Army of the Dead?",
         "Is Cobra Kai a movie or a TV series?",
         "What year was Cobra Kai released?",
         "Is Grey's Anatomy a TV show?",
         "Who directed Army of the Dead?",
         "How many seasons does Cobra Kai have?"],
        ["Who directed Army of the Dead?",
         "How long is Army of the Dead?",
         "Now switch to Cobra Kai. Is it a TV show?",
         "Back to Army of the Dead. Who directed it?"],
    ]
    comparison = {}
    for model_name in (OLD_MODEL_NAME, MODEL_NAME):
        announce("\n=== TESTING " + model_name + " ===")
        model_results = []
        try:
            for case in cases:
                messages = [{"role": "system", "content": SYSTEM}]
                transcript = []
                for question in case:
                    messages.append({"role": "user", "content": question})
                    reply = chat_once(messages, model_name)
                    messages.append({"role": "assistant", "content": reply})
                    transcript.append({"question": question, "answer": reply})
                    announce(f"Q: {question}\nA: {reply}\n")
                model_results.append(transcript)
        except (error.URLError, TimeoutError, OSError, ValueError) as exc:
            announce(f"Automatic Ollama test unavailable: {exc}")
            model_results.append({"automatic_test_error": str(exc)})
        comparison[model_name] = model_results
    REPORT.write_text(json.dumps({
        "comparison": comparison,
        "note": "Real Ollama outputs with matched settings. Judge facts and topic switching manually; no accuracy guarantee."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    announce(f"Comparison report: {REPORT}")


def main() -> int:
    try:
        announce("=== Export 40.38M title-switch candidate and compare with old Ollama model ===")
        export_checkpoint()
        convert_gguf()
        register_ollama()
        announce("Running live multi-turn Ollama test...")
        test_ollama()
        announce(f"DONE. Candidate available in Ollama Desktop as {MODEL_NAME}.")
        announce("IMPORTANT: Compare the printed old/new answers before deciding which model to present.")
        return 0
    except Exception as exc:
        announce(f"STOPPED SAFELY: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
