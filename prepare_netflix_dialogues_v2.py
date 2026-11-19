#!/usr/bin/env python3
"""Build checked, fact-grounded Netflix 4-turn SFT dialogues from the user's own JSONL.

This is a replacement for the earlier synthetic conversation generator. It extracts
only directly supported catalog fields, makes explicit subject changes, and
keeps the title in every assistant response. It never overwrites source files.

Usage (Windows):
  python prepare_netflix_dialogues_v2.py

Creates NEW files in SLM-Movie-Recommendation/dialogues_v2_data/.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

REPO = Path.home() / 'SLM-Movie-Recommendation'
SYSTEM = 'You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation.'
QUOTED = re.compile(r'["“‘]([^"”’\n]{2,85})["”’]')
BAD_ANSWER = re.compile(r"(?i)(isn't in my data|not in my (?:netflix )?dataset|couldn.t find|don.t have any record|don.t have.*record|i don.t know|no record of|won.t invent|without guessing|not in my catalog|did you mean|perhaps you meant)")
BAD_QUESTION = re.compile(r'(?i)(which is longer|which came first|compared with|same type of title|recommend|suggest|find me|list (?:all|some)|which titles|how many (?:movies|tv shows|titles)|both .+ and .+|versus| vs\.? )')
PATTERNS = {
    'type': re.compile(r'\b(?:is|was)\s+(?:a|an)\s+(Movie|TV Show)\b', re.I),
    'year': re.compile(r'\b(?:was released|released|came out|premiered)\s+in\s+(19\d{2}|20\d{2})\b', re.I),
    'rating': re.compile(r'\b(?:is rated|was rated|rated)\s+(TV-MA|TV-14|TV-PG|TV-G|TV-Y7|TV-Y|PG-13|PG|NC-17|R|G|NR)\b', re.I),
    'duration': re.compile(r'\b(?:runs|lasts)\s+(\d{1,3})\s*(?:minutes|mins?|min)\b', re.I),
    'seasons': re.compile(r'\bhas\s+(\d{1,2})\s+[Ss]easons?\b', re.I),
    'added': re.compile(r'\b(?:was )?added to Netflix on\s+([A-Z][a-z]+ \d{1,2}, \d{4})', re.I),
    'director': re.compile(r'\b(?:was )?directed by\s+([^.;]+)', re.I),
}
DIRECT = {
    'type': ['Is {t} a movie or a TV show?', 'What kind of title is {t}?', 'Is {t} a film or a series?'],
    'year': ['When was {t} released?', 'What year did {t} come out?', 'Tell me the release year of {t}.'],
    'rating': ['What is the maturity rating of {t}?', 'How is {t} rated?', 'What rating does {t} have?'],
    'duration': ['How long does {t} run?', 'What is the runtime of {t}?', 'How many minutes long is {t}?'],
    'seasons': ['How many seasons does {t} have?', 'Tell me the season count for {t}.'],
    'added': ['When was {t} added to Netflix?', 'What date did {t} arrive on Netflix?'],
    'director': ['Who directed {t}?', 'Can you tell me the director of {t}?'],
}
FOLLOW = {
    'type': ['Is it a movie or a TV series?', 'And is that one a film or a series?', 'What type of title is it?'],
    'year': ['When did it come out?', 'What year was it released?', 'And its release year?'],
    'rating': ['What is it rated?', 'And its maturity rating?', 'What rating does that title have?'],
    'duration': ['How long is it?', 'And its runtime?', 'How many minutes long is that one?'],
    'seasons': ['How many seasons does it have?', 'And its number of seasons?'],
    'added': ['When was it added to Netflix?', 'What date did it arrive on Netflix?'],
    'director': ['Who directed it?', 'And who is its director?'],
}
SWITCH_PREFIX = ['', 'Now switch topics. ', 'Let us discuss a different title. ', 'Next title: ']


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def title_from_pair(question: str, answer: str) -> str | None:
    """Only accept a single title explicitly present in BOTH question and answer."""
    if BAD_QUESTION.search(question) or BAD_ANSWER.search(answer):
        return None
    if len(question) > 220 or len(answer) > 370:
        return None
    candidates = []
    answer_titles = {t.strip().casefold() for t in QUOTED.findall(answer)}
    for title in QUOTED.findall(question) + QUOTED.findall(answer):
        title = title.strip()
        if not (3 <= len(title) <= 70 and any(c.isalpha() for c in title)):
            continue
        if title.casefold() in {'movie', 'tv show', 'yes', 'no', 'it', 'unknown', 'series'}:
            continue
        # The answer must explicitly quote the SAME title, not a longer or
        # corrected title containing it as a substring (e.g. An Unfinished).
        # Reject a shorter prefix mistaken for a longer title (e.g.
        # 'Cobra Kai' versus 'Cobra Kai - The Afterparty').
        is_prefix_of_longer = re.search(re.escape(title) + r'\s*[-:]\s*\w', question, re.I)
        if title.casefold() in answer_titles and title.casefold() in question.casefold() and not is_prefix_of_longer:
            if title.casefold() not in [t.casefold() for t in candidates]:
                candidates.append(title)
    if len(candidates) != 1:
        return None
    return candidates[0]


def collect_facts(path: Path):
    # Gather evidence for each fact, keeping the source response to prevent
    # unsupported facts and to detect contradictory annotations.
    evidence = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    canonical = defaultdict(Counter)
    source_rows = 0
    for row in read_jsonl(path):
        source_rows += 1
        messages = row.get('messages', [])
        if len(messages) < 3 or messages[1].get('role') != 'user' or messages[2].get('role') != 'assistant':
            continue
        q, a = messages[1].get('content', ''), messages[2].get('content', '')
        title = title_from_pair(q, a)
        if not title:
            continue
        key = title.casefold()
        canonical[key][title] += 1
        for field, pattern in PATTERNS.items():
            match = pattern.search(a)
            if not match:
                continue
            value = match.group(1).strip().rstrip(',')
            if field == 'director':
                value = re.split(r'\s+(?:and|but|while|who)\s+', value)[0].strip() if len(value) > 60 else value
                if not (3 <= len(value) <= 60) or 'my data' in value.lower():
                    continue
            if field == 'duration' and not 10 <= int(value) <= 400:
                continue
            if field == 'seasons' and not 1 <= int(value) <= 30:
                continue
            if field == 'type':
                value = 'TV Show' if value.lower() == 'tv show' else 'Movie'
            evidence[key][field][value].append((q, a))
    facts = {}
    conflict_fields = 0
    for key, fields in evidence.items():
        chosen = {}
        for field, values in fields.items():
            # Never silently pick one of two conflicting facts. Drop that field.
            if len(values) != 1:
                conflict_fields += 1
                continue
            value = next(iter(values))
            chosen[field] = value
        if chosen.get('type') == 'Movie':
            chosen.pop('seasons', None)
        if chosen.get('type') == 'TV Show':
            chosen.pop('duration', None)
        if len(chosen) >= 2:
            # Choose a canonical spelling with the most independent evidence.
            title = canonical[key].most_common(1)[0][0]
            facts[key] = (title, chosen)
    return facts, {'source_rows': source_rows, 'titles_with_2plus_nonconflicting_facts': len(facts), 'conflicting_fact_fields_excluded': conflict_fields}


def answer(title: str, field: str, value: str) -> str:
    t = f'"{title}"'
    if field == 'type': return f'{t} is a {value}.'
    if field == 'year': return f'{t} was released in {value}.'
    if field == 'rating': return f'{t} is rated {value}.'
    if field == 'duration': return f'{t} runs {value} minutes.'
    if field == 'seasons': return f'{t} has {value} {"Season" if value == "1" else "Seasons"}.'
    if field == 'added': return f'{t} was added to Netflix on {value}.'
    if field == 'director': return f'{t} was directed by {value}.'
    raise KeyError(field)


def dialogue_for(title_a, facts_a, title_b, facts_b, rng, mode):
    fields_a = rng.sample(list(facts_a), 2)
    fields_b = rng.sample(list(facts_b), 2)
    pairs = []
    if mode == 'return':
        # A -> B -> A: teaches returning to a previously mentioned title.
        seq = [(title_a, fields_a[0], facts_a, False),
               (title_b, fields_b[0], facts_b, False),
               (title_a, fields_a[1], facts_a, False),
               (title_b, fields_b[1], facts_b, False)]
    elif mode == 'single':
        fields = rng.sample(list(facts_a), min(4, len(facts_a)))
        if len(fields) < 4: return None
        seq = [(title_a, fields[0], facts_a, False)] + [(title_a, f, facts_a, True) for f in fields[1:]]
    else:
        seq = [(title_a, fields_a[0], facts_a, False),
               (title_a, fields_a[1], facts_a, True),
               (title_b, fields_b[0], facts_b, False),
               (title_b, fields_b[1], facts_b, True)]
    for idx, (title, field, facts, follow) in enumerate(seq):
        if follow:
            question = rng.choice(FOLLOW[field])
        else:
            question = rng.choice(DIRECT[field]).format(t=f'"{title}"')
            if idx > 0:
                question = rng.choice(SWITCH_PREFIX[1:]) + question
        pairs.append((question, answer(title, field, facts[field]), title))
    messages = [{'role': 'system', 'content': SYSTEM}]
    for q, a, _ in pairs:
        messages += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    return {'messages': messages}


def build(facts, count: int, seed: int):
    rng = random.Random(seed)
    keys = list(facts)
    if len(keys) < 10:
        raise RuntimeError(f'Insufficient supported titles: {len(keys)}')
    eligible_four = [k for k in keys if len(facts[k][1]) >= 4]
    out, seen = [], set()
    attempts = 0
    while len(out) < count and attempts < count * 25:
        attempts += 1
        mode = rng.choices(['switch', 'return', 'single'], weights=[75, 15, 10])[0]
        a, b = rng.sample(keys, 2)
        if mode == 'single':
            if not eligible_four: mode = 'switch'
            else: a = rng.choice(eligible_four)
        title_a, facts_a = facts[a]
        title_b, facts_b = facts[b]
        example = dialogue_for(title_a, facts_a, title_b, facts_b, rng, mode)
        if example is None:
            continue
        sig = tuple(m['content'].casefold() for m in example['messages'] if m['role'] == 'user')
        if sig in seen:
            continue
        seen.add(sig)
        out.append(example)
    if len(out) != count:
        raise RuntimeError(f'Only generated {len(out)} of {count} requested dialogues.')
    return out


def audit(records):
    for i, row in enumerate(records):
        messages = row['messages']
        if len(messages) != 9 or [m['role'] for m in messages] != ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant']:
            raise ValueError(f'Bad dialogue structure at {i}')
        for j in (2, 4, 6, 8):
            response = messages[j]['content']
            if not response.startswith('"') or not response.endswith('.') or len(response) > 160:
                raise ValueError(f'Unexpected response at {i}/{j}: {response}')
    return True


def write_new(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--train-count', type=int, default=12000)
    p.add_argument('--validation-count', type=int, default=400)
    args = p.parse_args()
    out = args.output_dir or (args.repo / 'dialogues_v2_data')
    inputs = [args.repo / 'data' / 'train.jsonl', args.repo / 'data' / 'validation.jsonl']
    outputs = [out / 'dialogues_train.jsonl', out / 'dialogues_validation.jsonl', out / 'quality_report.json']
    for path in inputs:
        if not path.is_file(): raise SystemExit(f'Missing input file: {path}')
    if any(path.exists() for path in outputs):
        raise SystemExit(f'Output files already exist in {out}; refusing to overwrite.')
    print('Reading and auditing original data...', flush=True)
    train_facts, train_stats = collect_facts(inputs[0])
    val_facts, val_stats = collect_facts(inputs[1])
    print('Source title coverage:', len(train_facts), 'train,', len(val_facts), 'validation', flush=True)
    train = build(train_facts, args.train_count, seed=20261119)
    validation = build(val_facts, args.validation_count, seed=20261120)
    audit(train); audit(validation)
    out.mkdir(parents=True, exist_ok=True)
    write_new(outputs[0], train)
    write_new(outputs[1], validation)
    report = {'train': {**train_stats, 'generated_dialogues': len(train)},
              'validation': {**val_stats, 'generated_dialogues': len(validation)},
              'notes': ['No facts invented; all answer values extracted from original responses.',
                        'Conflicting facts excluded, not silently resolved.',
                        'Every response repeats the current title to reinforce topic tracking.',
                        'The original training and validation files remain unchanged.']}
    with outputs[2].open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('DATA PREPARATION COMPLETE. Original files unchanged.', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print('EXAMPLE DIALOGUE:', flush=True)
    for msg in train[0]['messages']:
        print(msg['role'].upper() + ': ' + msg['content'], flush=True)
    print('NEXT: Evaluate actual generated conversations before exporting a model.', flush=True)

if __name__ == '__main__':
    main()
