#!/usr/bin/env python3
r"""Evaluate the SAVED best 40M Netflix dialogue checkpoint without training/export.

Run from PowerShell:
  & "$HOME\SLM-Movie-Recommendation\.venv\Scripts\python.exe" "$HOME\SLM-Movie-Recommendation\evaluate_netflix_dialogues_v2.py"

Writes only dialogues_v2_run/extended_evaluation.json. Does not change model files.
"""
from __future__ import annotations
import importlib.util
import json
import random
from pathlib import Path

HOME = Path.home()
REPO = HOME / 'SLM-Movie-Recommendation'
CHECKPOINT = REPO / 'dialogues_v2_run' / 'best.pt'
DATA = REPO / 'dialogues_v2_data' / 'dialogues_validation.jsonl'
REPORT = REPO / 'dialogues_v2_run' / 'extended_evaluation.json'
TRAIN_SCRIPT = REPO / 'train_netflix_dialogues_v2.py'


def load_training_helpers():
    if not TRAIN_SCRIPT.is_file():
        raise SystemExit(f'Missing existing helper: {TRAIN_SCRIPT}')
    spec = importlib.util.spec_from_file_location('netflix_dialogues_v2_eval_helper', TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit('Cannot load training helper.')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_dialogues(helper, model, tokenizer, base, rows):
    results = []
    for i, row in enumerate(rows):
        messages = row['messages']
        history = [messages[0]]
        turns = []
        for turn in range(4):
            question = messages[turn * 2 + 1]
            expected = messages[turn * 2 + 2]['content']
            generated = helper.generate_reply(model, tokenizer, base, history + [question],
                                              seed=20261120 + i * 101 + turn, max_tokens=56)
            title = helper.get_target_title(expected)
            fact = helper.get_target_fact(expected)
            title_ok = bool(title and title.casefold() in generated.casefold())
            fact_ok = bool(fact and fact.casefold() in generated.casefold())
            turns.append({'question': question['content'], 'expected': expected,
                          'generated': generated, 'title_match': title_ok,
                          'fact_match': fact_ok, 'is_followup': turn in (1, 3)})
            history.extend([question, {'role': 'assistant', 'content': generated}])
        results.append(turns)
    all_turns = [t for dialog in results for t in dialog]
    followups = [t for t in all_turns if t['is_followup']]
    return {
        'total_conversations': len(results),
        'total_answers': len(all_turns),
        'title_match_rate': round(sum(t['title_match'] for t in all_turns) / len(all_turns), 3),
        'fact_match_rate': round(sum(t['fact_match'] for t in all_turns) / len(all_turns), 3),
        'followup_title_match_rate': round(sum(t['title_match'] for t in followups) / len(followups), 3),
        'note': 'Automatic metrics are exact text-substring smoke tests, NOT a reliable measure of real-world truth.',
        'transcripts': results,
    }


def test_real_world_prompts(helper, model, tokenizer, base):
    cases = [
        ['What is Stranger Things about?', 'Is it a movie or a TV series?',
         'Now tell me about Money Heist.', 'Is that a movie or a TV show?'],
        ['What genre is Money Heist?', 'Who is the main character?',
         'Switch to Cobra Kai. Is it a film or a series?', 'What is it about?'],
        ['Recommend a Telugu comedy movie on Netflix.',
         'Can you recommend another Telugu comedy?',
         'Instead, what is the genre of Stranger Things?'],
        ['Is Cobra Kai a movie or TV show?', 'What is it about?',
         'Now tell me about Grey\'s Anatomy.', 'Is that a TV show?'],
    ]
    output = []
    for case_index, questions in enumerate(cases):
        history = [{'role': 'system', 'content': 'You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation.'}]
        turns = []
        for turn, question in enumerate(questions):
            user = {'role': 'user', 'content': question}
            answer = helper.generate_reply(model, tokenizer, base, history + [user],
                                           seed=20261130 + case_index * 101 + turn,
                                           max_tokens=70)
            turns.append({'question': question, 'generated': answer})
            history.extend([user, {'role': 'assistant', 'content': answer}])
        output.append(turns)
    return output


def main():
    for p in (CHECKPOINT, DATA, TRAIN_SCRIPT):
        if not p.is_file():
            raise SystemExit(f'Missing required file: {p}')
    helper = load_training_helpers()
    torch = helper.torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not detected. No CPU fallback: stopping safely.')
    base = helper.load_base()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'netflix_tokenizer_8k', use_fast=True)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    Model = base.import_model_class(REPO)
    model = Model(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to('cuda').eval()
    print('Loaded saved BEST checkpoint at update:', checkpoint['step'], flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    with DATA.open('r', encoding='utf-8') as f:
        validation = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(20261122)
    selected = rng.sample(validation, min(12, len(validation)))
    print('Testing', len(selected), 'additional validation dialogues (4 turns each)...', flush=True)
    validation_results = evaluate_dialogues(helper, model, tokenizer, base, selected)
    print('Testing practical Netflix conversations...', flush=True)
    practical_results = test_real_world_prompts(helper, model, tokenizer, base)
    report = {'checkpoint_step': checkpoint['step'],
              'checkpoint': str(CHECKPOINT),
              'validation': validation_results,
              'practical_conversations': practical_results,
              'notes': ['This is INFERENCE ONLY: no GPU training or Ollama export.',
                        'Facts from validation dialogues are compared with the supplied data, not externally verified.',
                        'Practical questions are qualitative tests; unsupported titles may cause hallucinations.',
                        'Exact-title metrics reward repeating a title even if the associated fact is wrong.']}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\nEVALUATION COMPLETE', flush=True)
    print('Best checkpoint step:', checkpoint['step'], flush=True)
    print('Validation title matching:', validation_results['title_match_rate'], flush=True)
    print('Validation factual matching:', validation_results['fact_match_rate'], flush=True)
    print('Validation follow-up title matching:', validation_results['followup_title_match_rate'], flush=True)
    print('Saved full report:', REPORT, flush=True)
    print('Upload extended_evaluation.json to ChatGPT. No retraining or conversion needed yet.', flush=True)


if __name__ == '__main__':
    main()
