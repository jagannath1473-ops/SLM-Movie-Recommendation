#!/usr/bin/env python3
r"""Controlled GPU fine-tuning for the user's 40.38M Netflix SLM.

Uses corrected dialogues_v2_data plus original single/multi-turn training examples.
Starts from ORIGINAL step-2750 checkpoint, not the failed synthetic longchat run.
Writes to a separate dialogues_v2_run folder. Includes real generated 4-turn tests
BEFORE any Ollama/GGUF conversion; lower validation loss alone is not success.

Windows:
  & "$HOME\SLM-Movie-Recommendation\.venv\Scripts\python.exe" "$HOME\SLM-Movie-Recommendation\train_netflix_dialogues_v2.py" --steps 2500
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / 'SLM-Movie-Recommendation'
RUN = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'outputs' / 'Netflix_SLM_3000_Training'
CUDA_PACKAGES = HOME / 'Documents' / 'Codex' / '2026-11-18' / 'new-chat' / 'cuda_torch'
SOURCE = RUN / 'run_40m' / 'netflix_slm_40m_step2750.pt'
DATA = REPO / 'dialogues_v2_data'
DEFAULT_OUTPUT = REPO / 'dialogues_v2_run'

if not CUDA_PACKAGES.is_dir():
    raise SystemExit(f'Missing CUDA PyTorch directory: {CUDA_PACKAGES}')
sys.path.insert(0, str(CUDA_PACKAGES))
import torch
from transformers import AutoTokenizer


def load_base():
    path = RUN / 'train_to_3000.py'
    if not path.is_file():
        raise SystemExit(f'Missing training helper: {path}')
    spec = importlib.util.spec_from_file_location('netflix_dialogue_training_base', path)
    if not spec or not spec.loader:
        raise SystemExit(f'Cannot import training helper: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2500)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--longchat-ratio', type=float, default=0.80)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--seed', type=int, default=20261119)
    return p.parse_args()


def read_rows(base, path):
    if not path.is_file():
        raise SystemExit(f'Missing dataset: {path}; run prepare_netflix_dialogues_v2.py first.')
    return [row for row in base.load_jsonl(path) if base.clean_conversation(row)]


def sample_batch(items, n, rng, device):
    selected = [items[rng.randrange(len(items))] for _ in range(n)]
    x = torch.stack([i[0] for i in selected]).to(device=device, dtype=torch.long)
    y = torch.stack([i[1] for i in selected]).to(device=device, dtype=torch.long)
    return x, y


def training_update(model, optimizer, long_items, short_items, args, rng):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(args.grad_accum):
        pool = long_items if rng.random() < args.longchat_ratio else short_items
        x, y = sample_batch(pool, args.microbatch, rng, 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / args.grad_accum).backward()
        losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return sum(losses) / len(losses)


def eval_loss(base, model, items):
    with torch.inference_mode():
        return base.evaluate_loss(model, items, batch_size=8, device=torch.device('cuda'))


def generate_reply(model, tokenizer, base, history, seed, max_tokens=56):
    """Actual autoregressive inference with MODEL-GENERATED prior assistant turns."""
    prompt = ''.join(base.format_message(m['role'], m['content']) for m in history)
    prompt += '<|assistant|>\n'
    tokens = tokenizer.encode(prompt, add_special_tokens=False)[-512:]
    if not tokens:
        return ''
    token_tensor = torch.tensor([tokens], dtype=torch.long, device='cuda')
    stop_ids = {tokenizer.convert_tokens_to_ids(s) for s in ('<|eos|>', '<|end|>', '<|user|>', '<|assistant|>')}
    stop_ids.discard(None)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    output = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(token_tensor[:, -512:])
            scores = logits[0, -1].float()
            for token_id in set(output):
                if scores[token_id] > 0:
                    scores[token_id] /= 1.12
                else:
                    scores[token_id] *= 1.12
            scores = scores / 0.3
            top_values, top_indices = torch.topk(scores, 20)
            probs = torch.softmax(top_values, dim=-1)
            order = torch.argsort(probs, descending=True)
            sorted_probs = probs[order]
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumulative - sorted_probs > 0.85] = 0
            sorted_probs /= sorted_probs.sum()
            choice = int(torch.multinomial(sorted_probs, 1, generator=generator).item())
            next_id = int(top_indices[order[choice]].item())
            if next_id in stop_ids:
                break
            output.append(next_id)
            token_tensor = torch.cat((token_tensor, torch.tensor([[next_id]], dtype=torch.long, device='cuda')), dim=1)
    return tokenizer.decode(output, skip_special_tokens=True).strip()


def get_target_title(response):
    match = re.match(r'^"([^"\n]+)"', response)
    return match.group(1) if match else None


def get_target_fact(response):
    # Answers are generated by prepare_netflix_dialogues_v2.py, and each has
    # one field value. A match is a conservative smoke-test, not proof of truth.
    patterns = [r'\bis a (TV Show|Movie)\b', r'\breleased in (\d{4})\b',
                r'\bis rated ([A-Z0-9-]+)\b', r'\bruns (\d+) minutes\b',
                r'\bhas (\d+) Seasons?\b',
                r'\badded to Netflix on ([A-Za-z]+ \d{1,2}, \d{4})\b',
                r'\bwas directed by (.+?)\.\s*$']
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.I)
        if match:
            return match.group(1)
    return None


def live_dialogue_test(model, tokenizer, base, examples, seed):
    model.eval()
    scores = []
    transcripts = []
    for index, example in enumerate(examples):
        messages = example['messages']
        history = [messages[0]]
        lines = []
        for turn in range(4):
            user = messages[turn * 2 + 1]
            expected = messages[turn * 2 + 2]['content']
            reply = generate_reply(model, tokenizer, base, history + [user], seed + index * 100 + turn)
            title = get_target_title(expected)
            fact = get_target_fact(expected)
            title_ok = bool(title and title.casefold() in reply.casefold())
            fact_ok = bool(fact and fact.casefold() in reply.casefold())
            scores.append({'turn': turn + 1, 'title': int(title_ok), 'fact': int(fact_ok)})
            lines.append({'question': user['content'], 'expected': expected,
                          'generated': reply, 'title_match': title_ok, 'fact_match': fact_ok})
            # The NEXT turn sees the model's own reply, not the teacher answer.
            history.extend([user, {'role': 'assistant', 'content': reply}])
        transcripts.append(lines)
    title_rate = sum(x['title'] for x in scores) / max(1, len(scores))
    fact_rate = sum(x['fact'] for x in scores) / max(1, len(scores))
    followup = [x for x in scores if x['turn'] in (2, 4)]
    followup_title_rate = sum(x['title'] for x in followup) / max(1, len(followup))
    model.train()
    return {'title_match_rate': round(title_rate, 3),
            'fact_match_rate': round(fact_rate, 3),
            'followup_title_match_rate': round(followup_title_rate, 3),
            'transcripts': transcripts}


def save_checkpoint(path, model, config, step, record):
    # No optimizer state: checkpoint is for inference/evaluation, and much smaller.
    payload = {'model_state_dict': model.state_dict(), 'config': config,
               'step': step, 'source_checkpoint': str(SOURCE),
               'training_objective': 'grounded_512_multiturn_sft_v2',
               'evaluation': record, 'parameter_count': sum(p.numel() for p in model.parameters())}
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    args = parse_args()
    if args.steps < 1 or args.microbatch < 1 or args.grad_accum < 1 or args.eval_every < 1:
        raise SystemExit('Invalid positive training arguments.')
    if not (0 < args.longchat_ratio < 1):
        raise SystemExit('--longchat-ratio must be between 0 and 1.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; refusing CPU training.')
    out = args.output_dir.resolve()
    if out in (REPO.resolve(), RUN.resolve(), SOURCE.parent.resolve(), DATA.resolve()):
        raise SystemExit('Unsafe output directory.')
    if any((out / name).exists() for name in ('best.pt', 'latest.pt', 'training_log.jsonl')):
        raise SystemExit('Output already contains checkpoints; refusing to overwrite.')
    for path in (SOURCE, DATA / 'dialogues_train.jsonl', DATA / 'dialogues_validation.jsonl'):
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')
    print('=== Netflix 40.38M: controlled grounded-dialogue training ===', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('SOURCE: original step-2750 checkpoint (read only)', flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    base = load_base()
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'netflix_tokenizer_8k', use_fast=True)
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)
    config = dict(source['config'])
    if config['max_seq_len'] != 128:
        raise SystemExit('Unexpected source model context.')
    config['max_seq_len'] = 512
    Model = base.import_model_class(REPO)
    model = Model(**config)
    model.load_state_dict(source['model_state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 40_384_512:
        raise SystemExit('Unexpected model size.')
    model.to('cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95), weight_decay=0.05)

    long_rows = read_rows(base, DATA / 'dialogues_train.jsonl')
    short_rows = read_rows(base, REPO / 'data' / 'train.jsonl')
    long_items = base.make_items(long_rows, tokenizer, 512)
    short_items = base.make_items(short_rows, tokenizer, 512)
    print('Training items:', len(long_items), 'grounded dialogue,', len(short_items), 'original', flush=True)
    if len(long_items) < 1000 or len(short_items) < 1000:
        raise SystemExit('Not enough training data; aborting.')
    rng = random.Random(args.seed)
    val_long_rows = read_rows(base, DATA / 'dialogues_validation.jsonl')
    val_short_rows = read_rows(base, REPO / 'data' / 'validation.jsonl')
    val_long_rows = rng.sample(val_long_rows, min(50, len(val_long_rows)))
    val_short_rows = rng.sample(val_short_rows, min(80, len(val_short_rows)))
    val_long = base.make_items(val_long_rows, tokenizer, 512)
    val_short = base.make_items(val_short_rows, tokenizer, 512)
    val_long = rng.sample(val_long, min(100, len(val_long)))
    val_short = rng.sample(val_short, min(80, len(val_short)))
    smoke = rng.sample(val_long_rows, min(2, len(val_long_rows)))

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training_log.jsonl'
    print('Initial validation loss (teacher-forced):', flush=True)
    initial_long = eval_loss(base, model, val_long)
    initial_short = eval_loss(base, model, val_short)
    print(f'  dialogue={initial_long:.4f}; original={initial_short:.4f}', flush=True)
    print('Initial real 4-turn generated conversation test:', flush=True)
    initial_live = live_dialogue_test(model, tokenizer, base, smoke, args.seed + 42)
    print(json.dumps(initial_live, ensure_ascii=False, indent=2), flush=True)
    best_score = float('inf')
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        warmup = min(100, max(1, args.steps // 10))
        if step <= warmup:
            lr = 3e-5 * step / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr = 5e-6 + (3e-5 - 5e-6) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        try:
            loss = training_update(model, optimizer, long_items, short_items, args, rng)
        except torch.cuda.OutOfMemoryError:
            print('CUDA OOM. Retry a NEW run with --microbatch 2. Existing checkpoints safe.', flush=True)
            raise SystemExit(2)
        if step == 1 or step % 100 == 0:
            print(f'Update {step}/{args.steps}: loss={loss:.4f}; elapsed={(time.perf_counter()-started)/60:.1f} min', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            long_loss = eval_loss(base, model, val_long)
            short_loss = eval_loss(base, model, val_short)
            live = live_dialogue_test(model, tokenizer, base, smoke, args.seed + 42)
            score = long_loss + 0.35 * short_loss - 0.15 * live['followup_title_match_rate']
            record = {'step': step, 'train_loss': loss, 'learning_rate': lr,
                      'dialogue_validation_loss': long_loss,
                      'original_validation_loss': short_loss,
                      'live_title_match_rate': live['title_match_rate'],
                      'live_fact_match_rate': live['fact_match_rate'],
                      'live_followup_title_match_rate': live['followup_title_match_rate'],
                      'selection_score': score, 'live_test': live['transcripts']}
            print('EVALUATION:', json.dumps(record, ensure_ascii=False, indent=2), flush=True)
            if score < best_score:
                best_score = score
                save_checkpoint(out / 'best.pt', model, config, step, record)
                print('Saved NEW best checkpoint.', flush=True)
            save_checkpoint(out / 'latest.pt', model, config, step, record)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print('TRAINING COMPLETE; source model and existing Ollama models unchanged.', flush=True)
    print('Output folder:', out, flush=True)
    print('IMPORTANT: Inspect live 4-turn results above BEFORE converting to GGUF.', flush=True)

if __name__ == '__main__':
    main()
