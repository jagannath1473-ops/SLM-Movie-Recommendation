#!/usr/bin/env python3
"""Create grounded multi-turn Netflix SFT conversations from existing Q/A JSONL.

Uses ONLY the existing train and validation splits. No invented movie facts.
Does not modify original data, model files, or checkpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

DEFAULT_REPO = Path.home() / 'SLM-Movie-Recommendation'
SYSTEM = 'You answer questions about Netflix movies and TV shows concisely and factually.'
QUOTED = re.compile(r'["“‘\']([^"”’\'\n]{2,75})["”’\']')
BAD = ('\ufffd', 'â€œ', 'â€', '????')


def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    # Refuse to overwrite any existing output.
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def extract_qa(rows: list[dict]) -> list[tuple[str, str]]:
    qa = []
    seen = set()
    for row in rows:
        msgs = row.get('messages', [])
        if not isinstance(msgs, list):
            continue
        for i, msg in enumerate(msgs):
            if msg.get('role') != 'assistant' or i == 0:
                continue
            prev = msgs[i - 1]
            if prev.get('role') != 'user':
                continue
            q = str(prev.get('content', '')).strip()
            a = str(msg.get('content', '')).strip()
            if not (8 <= len(q) <= 240 and 8 <= len(a) <= 350):
                continue
            if any(b in q or b in a for b in BAD):
                continue
            key = (q.casefold(), a.casefold())
            if key not in seen:
                qa.append((q, a))
                seen.add(key)
    return qa


def extract_title(question: str, answer: str) -> str | None:
    # The assistant answer usually begins with the referenced or recommended title.
    # Use only quoted spans, not guessed entities from arbitrary prose.
    for text in (answer, question):
        for match in QUOTED.finditer(text):
            title = match.group(1).strip()
            if (2 <= len(title) <= 55 and len(title.split()) <= 8
                    and not title.lower().startswith(('i ', 'we ', 'you ', 'it '))):
                return title
    return None


def title_in_question(question: str, title: str) -> bool:
    return bool(re.search(r'(?<!\w)' + re.escape(title) + r'(?!\w)', question, flags=re.I))


def refer_to_it(question: str, title: str) -> str | None:
    if not title_in_question(question, title):
        return None
    pattern = r'(?<!\w)' + re.escape(title) + r'(?!\w)'
    replaced = re.sub(pattern, 'it', question, count=1, flags=re.I)
    # Strip quotes that would otherwise leave the awkward phrase '"it"'.
    replaced = re.sub(r'["“‘\']it["”’\']', 'it', replaced, flags=re.I)
    replaced = re.sub(r'\s+', ' ', replaced).strip()
    return replaced if replaced.casefold() != question.casefold() else None


def as_conversation(pairs: list[tuple[str, str]]) -> dict:
    messages = [{'role': 'system', 'content': SYSTEM}]
    for q, a in pairs:
        messages.append({'role': 'user', 'content': q})
        messages.append({'role': 'assistant', 'content': a})
    return {'messages': messages}


def build(rows: list[dict], target: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    qa = extract_qa(rows)
    by_title: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for q, a in qa:
        title = extract_title(q, a)
        if title:
            by_title[title.casefold()].append((q, a, title))

    # For a genuine follow-up, require at least one other question that explicitly
    # names the same title and has a distinct answer/question.
    usable: dict[str, list[tuple[str, str, str]]] = {}
    for key, items in by_title.items():
        distinct = {q.casefold() for q, _, _ in items}
        explicit = [p for p in items if title_in_question(p[0], p[2])]
        if len(distinct) >= 2 and explicit:
            usable[key] = items
    keys = list(usable)
    out: list[dict] = []
    signatures = set()
    attempts = 0
    while len(out) < target and attempts < target * 60 and len(keys) >= 2:
        attempts += 1
        k1, k2 = rng.sample(keys, 2)
        a_group, b_group = usable[k1], usable[k2]
        a1 = rng.choice(a_group)
        b1_candidates = [p for p in b_group if title_in_question(p[0], p[2])]
        b1 = rng.choice(b1_candidates)
        a2_options = [(q, a, t) for q, a, t in a_group
                      if q.casefold() != a1[0].casefold() and
                      (follow := refer_to_it(q, t)) is not None and
                      follow.casefold() != a1[0].casefold()]
        b2_options = [(q, a, t) for q, a, t in b_group
                      if q.casefold() != b1[0].casefold() and
                      refer_to_it(q, t) is not None]
        if not a2_options or not b2_options:
            continue
        a2 = rng.choice(a2_options)
        b2 = rng.choice(b2_options)
        pairs = [
            (a1[0], a1[1]),
            (refer_to_it(a2[0], a2[2]), a2[1]),
            (b1[0], b1[1]),
            (refer_to_it(b2[0], b2[2]), b2[1]),
        ]
        if any(q is None for q, _ in pairs):
            continue
        signature = tuple(q.casefold() for q, _ in pairs)
        if signature in signatures:
            continue
        signatures.add(signature)
        out.append(as_conversation(pairs))

    # Supplement with explicit-title topic-switching dialogues, if the dataset
    # doesn't contain enough same-title questions for grounded anaphora examples.
    explicit = [(q, a) for q, a in qa if (t := extract_title(q, a)) and title_in_question(q, t)]
    attempts = 0
    while len(out) < target and attempts < target * 30 and len(explicit) >= 3:
        attempts += 1
        pairs = rng.sample(explicit, 3)
        signature = tuple(q.casefold() for q, _ in pairs)
        if signature in signatures:
            continue
        signatures.add(signature)
        out.append(as_conversation(pairs))

    stats = {
        'source_rows': len(rows),
        'unique_question_answer_pairs': len(qa),
        'titles_with_2_or_more_questions': len(usable),
        'generated_multiturn_conversations': len(out),
        'four_turn_conversations': sum(len(c['messages']) == 9 for c in out),
        'three_turn_conversations': sum(len(c['messages']) == 7 for c in out),
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_REPO / 'longchat_512_data')
    p.add_argument('--train-count', type=int, default=4000)
    p.add_argument('--validation-count', type=int, default=200)
    args = p.parse_args()
    train_path = args.repo / 'data' / 'train.jsonl'
    val_path = args.repo / 'data' / 'validation.jsonl'
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit(f'Dataset files missing under {args.repo / "data"}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / 'longchat_train.jsonl', args.output_dir / 'longchat_validation.jsonl']
    if any(path.exists() for path in outputs):
        raise SystemExit('Output already exists; refusing to overwrite. No changes made.')
    train, train_stats = build(read_jsonl(train_path), args.train_count, 20261119)
    val, val_stats = build(read_jsonl(val_path), args.validation_count, 20261120)
    if len(train) < 50:
        raise SystemExit(f'Only {len(train)} examples could be made; inspect data before training.')
    write_jsonl(outputs[0], train)
    write_jsonl(outputs[1], val)
    print('SUCCESS: Original dataset and checkpoints were not changed.')
    print(json.dumps({'train': train_stats, 'validation': val_stats,
                      'train_file': str(outputs[0]), 'validation_file': str(outputs[1])}, indent=2))
    print('\nFIRST TRAINING CONVERSATION:')
    for msg in train[0]['messages']:
        print(f"  {msg['role'].upper()}: {msg['content']}")
    print('\nNext: inspect these results before starting GPU fine-tuning.')


if __name__ == '__main__':
    main()
