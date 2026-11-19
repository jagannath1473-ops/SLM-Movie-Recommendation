#!/usr/bin/env python3
"""Read-only preflight for Netflix 40M 512-token continuation training.

Checks that the 4-turn dataset fits the target window and that the original
step-2750 weights can load into the 512-token RoPE model and run on CUDA.
Does not train, overwrite, or save anything.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HOME = Path.home()
REPO = HOME / 'SLM-Movie-Recommendation'
RUN = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'outputs' / 'Netflix_SLM_3000_Training'
CUDA_PACKAGES = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'cuda_torch'

if not CUDA_PACKAGES.is_dir():
    raise SystemExit(f'Cannot find GPU PyTorch folder: {CUDA_PACKAGES}')
sys.path.insert(0, str(CUDA_PACKAGES))

import torch
from transformers import AutoTokenizer


def format_message(msg: dict) -> str:
    role = msg['role']
    ending = '<|eos|>' if role == 'assistant' else '<|end|>'
    return f'<|{role}|>\n{msg.get("content", "")}{ending}\n'


def main() -> None:
    print('=== Netflix SLM 512-token read-only preflight ===', flush=True)
    print('PyTorch:', torch.__version__, flush=True)
    print('CUDA available:', torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available; no model test attempted.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(REPO / 'netflix_tokenizer_8k', use_fast=True)
    for split in ('train', 'validation'):
        path = REPO / 'longchat_512_data' / f'longchat_{split}.jsonl'
        if not path.is_file():
            raise SystemExit(f'Missing generated dataset: {path}')
        lengths = []
        turn_counts = []
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row['messages']
                turn_counts.append(sum(m['role'] == 'user' for m in messages))
                text = ''.join(format_message(m) for m in messages)
                lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        lengths.sort()
        fit = sum(n <= 513 for n in lengths)
        print(f'{split}: {len(lengths)} conversations, '
              f'{sum(n == 4 for n in turn_counts)} have four user turns', flush=True)
        print(f'  Tokens: median={lengths[len(lengths)//2]}, '
              f'p90={lengths[min(len(lengths)-1, int(.9*len(lengths)))]}, '
              f'max={lengths[-1]}', flush=True)
        print(f'  Entire conversation fits 512-token input: '
              f'{fit}/{len(lengths)} ({100*fit/len(lengths):.1f}%)', flush=True)

    checkpoint_path = RUN / 'run_40m' / 'netflix_slm_40m_step2750.pt'
    if not checkpoint_path.is_file():
        raise SystemExit(f'Checkpoint missing: {checkpoint_path}')
    print('Loading checkpoint (read only)...', flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    print('Checkpoint context:', config['max_seq_len'], flush=True)
    config['max_seq_len'] = 512
    source = REPO / 'huggingface_netflix_slm_v2' / 'modeling_netflix_slm.py'
    spec = importlib.util.spec_from_file_location('netflix_model_512_preflight', source)
    if spec is None or spec.loader is None:
        raise SystemExit(f'Could not load architecture: {source}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Qwen3Model(**config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('Original 40M weights loaded into 512-token model: PASS', flush=True)
    model = model.to('cuda').eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, _ = model(torch.randint(0, config['vocab_size'], (1, 512), device='cuda'))
    torch.cuda.synchronize()
    print('512-token CUDA forward pass: PASS', tuple(logits.shape), flush=True)
    print('Peak GPU memory during check: '
          f'{torch.cuda.max_memory_allocated()/2**30:.2f} GB', flush=True)
    print('SUCCESS: No training performed; original files unchanged.', flush=True)


if __name__ == '__main__':
    main()
