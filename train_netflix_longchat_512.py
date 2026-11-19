#!/usr/bin/env python3
"""Safely continue the user's 40.38M Netflix SLM on 512-token multi-turn dialogs.

Windows project-specific, uses existing local CUDA torch installation. The source
step-2750 checkpoint and original Ollama model are NEVER written to. Run first:
    python train_netflix_longchat_512.py --benchmark-only
Then, after reviewing GPU memory/time:
    python train_netflix_longchat_512.py --steps 300

Only writes to SLM-Movie-Recommendation/longchat_512_run during real training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / 'SLM-Movie-Recommendation'
RUN = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'outputs' / 'Netflix_SLM_3000_Training'
CUDA_PACKAGES = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'cuda_torch'
SOURCE = RUN / 'run_40m' / 'netflix_slm_40m_step2750.pt'
DEFAULT_OUTPUT = REPO / 'longchat_512_run'

if not CUDA_PACKAGES.is_dir():
    raise SystemExit(f'CUDA package folder not found: {CUDA_PACKAGES}')
sys.path.insert(0, str(CUDA_PACKAGES))

import torch
from transformers import AutoTokenizer


def load_base():
    path = RUN / 'train_to_3000.py'
    if not path.is_file():
        raise SystemExit(f'Missing original training helper: {path}')
    spec = importlib.util.spec_from_file_location('netflix_longchat_training_base', path)
    if spec is None or spec.loader is None:
        raise SystemExit(f'Could not import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-only', action='store_true', help='Test backward pass and VRAM; no files are written')
    p.add_argument('--benchmark-steps', type=int, default=3)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--longchat-ratio', type=float, default=0.8)
    p.add_argument('--eval-every', type=int, default=75)
    p.add_argument('--save-every', type=int, default=75)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--seed', type=int, default=20261119)
    return p.parse_args()


def load_rows(base, path):
    if not path.is_file():
        raise SystemExit(f'Missing data: {path}')
    return [r for r in base.load_jsonl(path) if base.clean_conversation(r)]


def build_items(base, tokenizer, max_len, benchmark):
    long_rows = load_rows(base, REPO / 'longchat_512_data' / 'longchat_train.jsonl')
    short_rows = load_rows(base, REPO / 'data' / 'train.jsonl')
    if benchmark:
        # Benchmark requires only enough items to sample a few small batches.
        long_rows = long_rows[:80]
        short_rows = short_rows[:80]
    print(f'Preparing {len(long_rows)} multi-turn and {len(short_rows)} original conversations...', flush=True)
    long_items = base.make_items(long_rows, tokenizer, max_len)
    short_items = base.make_items(short_rows, tokenizer, max_len)
    if not long_items or not short_items:
        raise SystemExit('No training items were created. No training performed.')
    print(f'Training items: {len(long_items)} multi-turn, {len(short_items)} original', flush=True)
    return long_items, short_items


def batch(items, count, rng, device):
    selected = [items[rng.randrange(len(items))] for _ in range(count)]
    x = torch.stack([i[0] for i in selected]).to(device=device, dtype=torch.long)
    y = torch.stack([i[1] for i in selected]).to(device=device, dtype=torch.long)
    return x, y


def update(model, optimizer, long_items, short_items, args, rng):
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(args.grad_accum):
        items = long_items if rng.random() < args.longchat_ratio else short_items
        x, y = batch(items, args.microbatch, rng, 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / args.grad_accum).backward()
        total_loss += float(loss.detach().item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss / args.grad_accum


def evaluation_items(base, tokenizer, seed):
    rng = random.Random(seed + 100)
    long_rows = load_rows(base, REPO / 'longchat_512_data' / 'longchat_validation.jsonl')
    short_rows = load_rows(base, REPO / 'data' / 'validation.jsonl')
    # Sample before tokenizing for quick periodic evaluation. Entire splits stay unchanged.
    long_rows = rng.sample(long_rows, min(40, len(long_rows)))
    short_rows = rng.sample(short_rows, min(64, len(short_rows)))
    long_all = base.make_items(long_rows, tokenizer, 512)
    short_all = base.make_items(short_rows, tokenizer, 512)
    long_eval = rng.sample(long_all, min(96, len(long_all)))
    short_eval = rng.sample(short_all, min(64, len(short_all)))
    return long_eval, short_eval


def save_checkpoint(path, model, optimizer, config, step, record, source):
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'source_checkpoint': str(source),
        'training_objective': 'assistant_only_multiturn_512_sft',
        'evaluation': record,
        'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    # Write temporary checkpoint first; replace only within the NEW output folder.
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    args = parse_args()
    if args.microbatch < 1 or args.grad_accum < 1 or args.steps < 1:
        raise SystemExit('microbatch, grad-accum, and steps must be positive.')
    if not (0 <= args.longchat_ratio <= 1):
        raise SystemExit('longchat-ratio must be between 0 and 1.')
    if args.eval_every < 1 or args.save_every < 1 or args.benchmark_steps < 1:
        raise SystemExit('eval-every, save-every and benchmark-steps must be positive.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available. Refusing CPU training.')
    if not SOURCE.is_file():
        raise SystemExit(f'Original 40M checkpoint missing: {SOURCE}')
    if not args.benchmark_only:
        # Explicitly prohibit overwriting original project/model locations.
        out = args.output_dir.resolve()
        if out == REPO.resolve() or out == RUN.resolve() or out == SOURCE.parent.resolve():
            raise SystemExit('Refusing unsafe output directory.')
        if any((out / name).exists() for name in ('best.pt', 'latest.pt', 'training_log.jsonl')):
            raise SystemExit(f'Training files already exist in {out}; refusing overwrite. Preserve the run.')

    print('=== Netflix 40M -> 512-token multi-turn continuation ===', flush=True)
    print('Mode:', 'GPU BENCHMARK (NO SAVES)' if args.benchmark_only else 'TRAINING (NEW FOLDER ONLY)', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__, flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    base = load_base()
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'netflix_tokenizer_8k', use_fast=True)

    print('Loading step-2750 checkpoint (READ ONLY)...', flush=True)
    checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    if config.get('max_seq_len') != 128:
        raise SystemExit(f'Unexpected original context: {config.get("max_seq_len")}')
    config['max_seq_len'] = 512
    Model = base.import_model_class(REPO)
    model = Model(**config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    param_count = sum(p.numel() for p in model.parameters())
    if param_count != 40_384_512:
        raise SystemExit(f'Unexpected model parameter count: {param_count}')
    model = model.to('cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.05)
    long_items, short_items = build_items(base, tokenizer, 512, args.benchmark_only)
    rng = random.Random(args.seed)

    if args.benchmark_only:
        print('Running real 512-token BACKWARD passes (not just forward)...', flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for step in range(1, args.benchmark_steps + 1):
                loss = update(model, optimizer, long_items, short_items, args, rng)
                print(f'Benchmark update {step}/{args.benchmark_steps}: loss={loss:.4f}', flush=True)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print('GPU OUT OF MEMORY: try again with --microbatch 2', flush=True)
            raise SystemExit(2)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(json.dumps({
            'result': 'BENCHMARK PASS - NO FILES WRITTEN',
            'microbatch': args.microbatch,
            'gradient_accumulation': args.grad_accum,
            'context_tokens': 512,
            'peak_allocated_gpu_gb': round(peak, 2),
            'seconds_per_update': round(elapsed / args.benchmark_steps, 2),
            'rough_300_update_minutes_excluding_evaluation': round(elapsed / args.benchmark_steps * 300 / 60, 1),
            'warning': 'Time estimate is approximate; evaluation/checkpoint saves add time.'
        }, indent=2), flush=True)
        return

    long_eval, short_eval = evaluation_items(base, tokenizer, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / 'training_log.jsonl'
    print('New checkpoint folder:', args.output_dir, flush=True)
    print('Evaluating initial 512-token model...', flush=True)
    with torch.inference_mode():
        baseline_long = base.evaluate_loss(model, long_eval, 8, torch.device('cuda'))
        baseline_short = base.evaluate_loss(model, short_eval, 8, torch.device('cuda'))
    print(f'Baseline validation loss: multi-turn={baseline_long:.4f}, original={baseline_short:.4f}', flush=True)
    best_score = float('inf')
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        # Short warm-up, then cosine decay, from a conservative 5e-5 learning rate.
        warmup = min(20, max(1, args.steps // 10))
        if step <= warmup:
            lr = 5e-5 * (step / warmup)
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr = 1e-5 + 0.5 * (5e-5 - 1e-5) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        try:
            loss = update(model, optimizer, long_items, short_items, args, rng)
        except torch.cuda.OutOfMemoryError:
            print('GPU OUT OF MEMORY. Last saved checkpoint remains in the new output folder.', flush=True)
            raise SystemExit(2)
        if step == 1 or step % 25 == 0:
            print(f'Update {step}/{args.steps}: loss={loss:.4f}, lr={lr:.7f}, '
                  f'elapsed={(time.perf_counter()-started)/60:.1f} min', flush=True)
        should_eval = step % args.eval_every == 0 or step == args.steps
        should_save = step % args.save_every == 0 or step == args.steps
        record = {'step': step, 'train_loss': loss, 'learning_rate': lr}
        if should_eval:
            with torch.inference_mode():
                long_loss = base.evaluate_loss(model, long_eval, 8, torch.device('cuda'))
                short_loss = base.evaluate_loss(model, short_eval, 8, torch.device('cuda'))
            score = long_loss + 0.3 * short_loss
            record.update({'longchat_validation_loss': long_loss,
                           'original_validation_loss': short_loss,
                           'selection_score': score})
            print('Validation:', json.dumps(record), flush=True)
            if score < best_score:
                best_score = score
                save_checkpoint(args.output_dir / 'best.pt', model, optimizer, config, step, record, SOURCE)
                print('Saved new best checkpoint.', flush=True)
        if should_save:
            save_checkpoint(args.output_dir / 'latest.pt', model, optimizer, config, step, record, SOURCE)
            print('Saved new latest checkpoint.', flush=True)
        if should_eval or should_save:
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
    print('TRAINING COMPLETE. Original model and source checkpoint unchanged.', flush=True)
    print('New model checkpoints:', args.output_dir, flush=True)
    print('NEXT: convert the selected checkpoint to GGUF and test multi-turn in Ollama.', flush=True)


if __name__ == '__main__':
    main()
