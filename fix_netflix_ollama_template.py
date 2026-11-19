#!/usr/bin/env python3
r"""Repair Ollama prompt template for the user's EXISTING trained Netflix 40.38M GGUF.

Does not retrain, convert, replace weights, or modify old Ollama models.
One PowerShell command:
  & "$HOME\SLM-Movie-Recommendation\.venv\Scripts\python.exe" "$HOME\SLM-Movie-Recommendation\fix_netflix_ollama_template.py"
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib import request

HOME = Path.home()
ROOT = HOME / "SLM-Movie-Recommendation"
EXPORT = ROOT / "dialogues_v2_qwen3"
GGUF = EXPORT / "netflix-slm-40m-dialogue-v2-f32.gguf"
MODEL = "netflix-slm-40m-dialogue-v2-fixed"
MODELFILE = EXPORT / "Modelfile.dialogue-v2-fixed"
REPORT = ROOT / "dialogues_v2_run" / "ollama_fixed_template_test.json"
SYSTEM = "You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation."

# Exact training format: <|role|>\ncontent<|end|>\n (user/system)
# and <|assistant|>\ncontent<|eos|>\n for previous assistant replies.
# The trailing <|assistant|>\n opens the next generation.
TEMPLATE = '''{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|user|>
{{ .Content }}<|end|>
{{ else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|eos|>
{{ end }}{{ end }}<|assistant|>
'''


def run(*args: str) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stdout[-2500:]}")
    return result.stdout


def chat(messages: list[dict]) -> str:
    data = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 512, "num_predict": 96, "temperature": 0.15, "seed": 20261119},
    }).encode("utf-8")
    req = request.Request("http://127.0.0.1:11434/api/chat", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        result = json.load(resp)
    return result.get("message", {}).get("content", "").strip()


def main() -> int:
    try:
        print("=== Repair the chat template for YOUR trained Netflix 40.38M SLM ===", flush=True)
        if not GGUF.is_file():
            raise RuntimeError(f"Your existing converted GGUF is missing: {GGUF}")
        if GGUF.stat().st_size < 50_000_000:
            raise RuntimeError("GGUF appears incomplete. Refusing to use it.")
        print("Reusing your existing trained weights. No retraining or conversion.", flush=True)
        modelfile = f'''FROM ./{GGUF.name}
TEMPLATE """{TEMPLATE}"""
SYSTEM "{SYSTEM}"
PARAMETER num_ctx 512
PARAMETER num_predict 96
PARAMETER temperature 0.15
PARAMETER top_k 20
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.08
PARAMETER stop "<|eos|>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
'''
        MODELFILE.write_text(modelfile, encoding="utf-8")
        print(f"Registering corrected-template Ollama entry: {MODEL}", flush=True)
        print(run("ollama", "create", MODEL, "-f", str(MODELFILE))[-1400:], flush=True)
        shown = run("ollama", "show", MODEL, "--template")
        if "<|user|>" not in shown or "<|assistant|>" not in shown:
            raise RuntimeError("Ollama still did not register the required custom role tokens. STOP; do not trust results.")
        print("PASS: Ollama now uses the exact <|system|>/<|user|>/<|assistant|> training tokens.", flush=True)
        tests = [
            ["Who directed Army of the Dead?", "Is it a movie or a TV show?", "Now switch to Cobra Kai. Is it a TV show?", "Now switch back. Who directed Army of the Dead?"],
            ["Is Cobra Kai a movie or a TV show?", "Now what is Army of the Dead?", "Is that a movie?"],
            ["What is Stranger Things about?", "Is it a movie or a TV series?", "Now tell me about Money Heist."],
        ]
        output = []
        for i, questions in enumerate(tests, start=1):
            print(f"\nCONVERSATION {i}", flush=True)
            messages = []  # SYSTEM comes from the Modelfile; no duplicate system message.
            transcript = []
            for q in questions:
                messages.append({"role": "user", "content": q})
                answer = chat(messages)
                messages.append({"role": "assistant", "content": answer})
                transcript.append({"question": q, "answer": answer})
                print(f"Q: {q}\nA: {answer}\n", flush=True)
            output.append(transcript)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps({"model": MODEL, "same_trained_gguf": str(GGUF),
                                      "corrected_template": shown, "tests": output}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"DONE. Full results: {REPORT}", flush=True)
        print(f"In Ollama Desktop choose: {MODEL}", flush=True)
        print("IMPORTANT: Correct formatting cannot fix missing knowledge or model-capacity limits.", flush=True)
        return 0
    except Exception as exc:
        print(f"STOPPED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
