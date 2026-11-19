#!/usr/bin/env python3
"""Conservative title-switch repair experiment for an existing Netflix 40.38M SLM.

FIRST:  python netflix_title_switch_repair.py --audit
LATER:  python netflix_title_switch_repair.py --train --steps 400

This does not overwrite source data, the dialogues_v2_run checkpoint, or any Ollama
model. It creates a NEW experiment folder only. No accuracy guarantee.
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

REPO = Path.home() / "SLM-Movie-Recommendation"
SOURCE = REPO / "dialogues_v2_run" / "best.pt"
OUTPUT = REPO / "title_switch_repair_run"
SYSTEM = "You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation."
FOCUS = ("Army of the Dead", "Cobra Kai", "Grey's Anatomy", "Stranger Things", "Money Heist")

# All questions are generated from facts actually extracted from the user's
# existing train/validation JSONL, never from invented external facts.
DIRECT = {
    "type": ("Is {t} a movie or a TV show?", "Is {t} a movie or a TV series?", "Is {t} a film or a series?", "What type of title is {t}?"),
    "year": ("What year was {t} released?", "When did {t} come out?", "Tell me the release year of {t}."),
    "director": ("Who directed {t}?", "Who is the director of {t}?", "Can you tell me who directed {t}?"),
    "rating": ("What is {t} rated?", "What is the maturity rating of {t}?"),
    "duration": ("How long is {t}?", "What is the runtime of {t}?"),
    "seasons": ("How many seasons does {t} have?", "What is the season count of {t}?"),
    "added": ("When was {t} added to Netflix?", "On what date did Netflix add {t}?"),
}
FOLLOW = {
    "type": ("Is it a movie or a TV series?", "Is it a TV show?", "What type of title is it?"),
    "year": ("What year was it released?", "When did it come out?"),
    "director": ("Who directed it?", "And who is its director?"),
    "rating": ("What is it rated?", "What's its maturity rating?"),
    "duration": ("How long is it?", "What's its runtime?"),
    "seasons": ("How many seasons does it have?", "How many seasons are listed?"),
    "added": ("When was it added to Netflix?", "What date was it added?"),
}


def load_module(path: Path, name: str):
    if not path.is_file():
        raise SystemExit(f"Missing required project file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_facts(prep, repo: Path):
    train = repo / "data" / "train.jsonl"
    val = repo / "data" / "validation.jsonl"
    if not train.is_file() or not val.is_file():
        raise SystemExit("Missing data/train.jsonl or data/validation.jsonl")
    train_facts, train_stats = prep.collect_facts(train)
    val_facts, val_stats = prep.collect_facts(val)
    return train_facts, val_facts, train_stats, val_stats


def make_question(title, field, rng, *, changed=False, returned=False, follow=False):
    if changed:
        if returned:
            intro = rng.choice((f"Now switch back to {title}. ", f"Back to {title}: ",
                                f"Let's return to {title}. ", f"Earlier we discussed {title}. "))
        else:
            intro = rng.choice((f"Now switch to {title}. ", f"Let's discuss {title}. ",
                                f"Next, {title}. ", f"Change topic to {title}. "))
        # Sometimes title occurs only in the introduction, while the question
        # uses "it". This trains title tracking across both sentences.
        if rng.random() < 0.5:
            return intro + rng.choice(FOLLOW[field])
        return intro + rng.choice(DIRECT[field]).format(t=title)
    if follow:
        return rng.choice(FOLLOW[field])
    # The original v2 generator ALWAYS quoted titles. Most real users don't.
    displayed = f'"{title}"' if rng.random() < 0.2 else title
    return rng.choice(DIRECT[field]).format(t=displayed)


def create_dialogue(facts, keys, rng, prep, focus_keys):
    # 55% A-B-A-B, 20% A-A-B-B, 15% A-B-B-A, 10% one-title follow-ups.
    mode = rng.choices(("abab", "aabb", "abba", "aaaa"), weights=(55, 20, 15, 10))[0]
    if focus_keys and rng.random() < 0.25:
        a = rng.choice(focus_keys)
        b = rng.choice([x for x in keys if x != a])
    else:
        a, b = rng.sample(keys, 2)
    if mode == "aaaa" and len(facts[a][1]) < 4:
        mode = "abab"
    order = {"abab": (a, b, a, b), "aabb": (a, a, b, b),
             "abba": (a, b, b, a), "aaaa": (a, a, a, a)}[mode]
    fields = {}
    for key in set(order):
        needed = order.count(key)
        available = list(facts[key][1])
        if len(available) < needed:
            return None
        fields[key] = iter(rng.sample(available, needed))
    messages = [{"role": "system", "content": SYSTEM}]
    previous = None
    seen = set()
    for key in order:
        title, known = facts[key]
        field = next(fields[key])
        changed = previous is not None and key != previous
        question = make_question(title, field, rng, changed=changed,
                                 returned=key in seen, follow=previous == key and rng.random() < 0.75)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": prep.answer(title, field, known[field])})
        seen.add(key)
        previous = key
    return {"messages": messages}


def generate(facts, prep, count, seed):
    rng = random.Random(seed)
    keys = [key for key, (_, values) in facts.items() if len(values) >= 2]
    if len(keys) < 20:
        raise SystemExit(f"Only {len(keys)} titles with 2+ facts; cannot train safely")
    focus_keys = [k for k in keys if k in {name.casefold() for name in FOCUS}]
    result, signatures = [], set()
    for _ in range(count * 25):
        if len(result) == count:
            break
        row = create_dialogue(facts, keys, rng, prep, focus_keys)
        if row is None:
            continue
        signature = tuple(m["content"] for m in row["messages"] if m["role"] == "user")
        if signature in signatures:
            continue
        signatures.add(signature)
        result.append(row)
    if len(result) < count:
        raise SystemExit(f"Generated only {len(result)}/{count} unique dialogues")
    return result


def audit(prep, repo: Path):
    train_facts, val_facts, train_stats, val_stats = read_facts(prep, repo)
    print("=== READ-ONLY NETFLIX DATA AUDIT ===")
    print("Train facts:", json.dumps(train_stats, indent=2))
    print("Validation facts:", json.dumps(val_stats, indent=2))
    for name in FOCUS:
        value = train_facts.get(name.casefold())
        print(f"{name}: " + ("fields=" + ", ".join(sorted(value[1])) if value else "NOT IN CLEAN TRAIN FACTS"))
    sample = generate(train_facts, prep, 3, 20261119)
    print("\nSample naturally worded A/B conversation:")
    for m in sample[0]["messages"]:
        print(m["role"].upper() + ": " + m["content"])
    print("\nAUDIT COMPLETE: no files changed; no GPU training.")
    return train_facts, val_facts


def save_checkpoint(torch, path, model, config, step, metrics):
    payload = {"model_state_dict": model.state_dict(), "config": config,
               "step": step, "source_checkpoint": str(SOURCE),
               "training_objective": "natural_title_switch_sft_experiment",
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "evaluation": metrics}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def evaluate_generated(helper, model, tokenizer, base, examples, seed):
    model.eval()
    total = 0
    title_correct = 0
    fact_correct = 0
    joint_correct = 0
    transcripts = []
    for index, row in enumerate(examples):
        history = [row["messages"][0]]
        for turn in range(4):
            question = row["messages"][turn * 2 + 1]
            expected = row["messages"][turn * 2 + 2]["content"]
            reply = helper.generate_reply(model, tokenizer, base, history + [question],
                                          seed + index * 101 + turn, max_tokens=56)
            title = helper.get_target_title(expected)
            fact = helper.get_target_fact(expected)
            title_ok = bool(title and reply.casefold().startswith(f'"{title}"'.casefold()))
            fact_ok = bool(fact and fact.casefold() in reply.casefold())
            total += 1
            title_correct += title_ok
            fact_correct += fact_ok
            joint_correct += title_ok and fact_ok
            if len(transcripts) < 16:
                transcripts.append({"question": question["content"], "expected": expected,
                                    "generated": reply, "title_ok": title_ok, "fact_ok": fact_ok})
            history.extend((question, {"role": "assistant", "content": reply}))
    model.train()
    return {"answers": total, "title_rate": round(title_correct / total, 3),
            "fact_rate": round(fact_correct / total, 3),
            "joint_rate": round(joint_correct / total, 3), "transcripts": transcripts}


def train(args, prep, train_facts, val_facts):
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {OUTPUT}")
    if not SOURCE.is_file():
        raise SystemExit(f"Missing existing trained 40M v2 checkpoint: {SOURCE}")
    helper = load_module(REPO / "train_netflix_dialogues_v2.py", "repair_train_helper")
    torch = helper.torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. No CPU fallback or model changes.")
    base = helper.load_base()
    tokenizer = helper.AutoTokenizer.from_pretrained(REPO / "netflix_tokenizer_8k", use_fast=True)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    if config.get("max_seq_len") != 512:
        raise SystemExit("Expected existing v2 512-token checkpoint. No training performed.")
    Model = base.import_model_class(REPO)
    model = Model(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 40_384_512:
        raise SystemExit("Parameter count mismatch. No training performed.")
    model = model.to("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Generating fact-grounded natural-title switching examples...", flush=True)
    train_rows = generate(train_facts, prep, args.train_count, args.seed)
    val_rows = generate(val_facts, prep, args.validation_count, args.seed + 1)
    new_items = base.make_items(train_rows, tokenizer, 512)
    original_rows = helper.read_rows(base, REPO / "data" / "train.jsonl")
    old_rows = helper.read_rows(base, REPO / "dialogues_v2_data" / "dialogues_train.jsonl")
    old_items = base.make_items(old_rows, tokenizer, 512)
    original_items = base.make_items(original_rows, tokenizer, 512)
    validation_items = base.make_items(val_rows[:24], tokenizer, 512)
    live_rows = val_rows[24:30]
    if not all((new_items, old_items, original_items, validation_items, live_rows)):
        raise SystemExit("Empty training/evaluation data. No training performed.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; train items: new={len(new_items)}, v2={len(old_items)}, original={len(original_items)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def evaluate(step):
        val_loss = helper.eval_loss(base, model, validation_items)
        live = evaluate_generated(helper, model, tokenizer, base, live_rows, args.seed + 900)
        print(f"EVAL step={step}: validation_loss={val_loss:.4f}, title={live['title_rate']:.1%}, "
              f"fact={live['fact_rate']:.1%}, joint={live['joint_rate']:.1%}", flush=True)
        return {"step": step, "validation_loss": val_loss, **live}

    # Create the new directory only AFTER all dependencies/data/checkpoint load.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "experiment_info.json").write_text(json.dumps({
        "source": str(SOURCE), "parameter_count": 40_384_512,
        "new_dialogues": len(train_rows), "validation_dialogues": len(val_rows),
        "note": "New experiment. Original model, Ollama entries, data and checkpoints unchanged."
    }, indent=2), encoding="utf-8")
    print("Measuring baseline before any weight updates...", flush=True)
    baseline = evaluate(0)
    (OUTPUT / "baseline_evaluation.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    best = baseline
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            pool = rng.choices((new_items, old_items, original_items), weights=(70, 20, 10))[0]
            x, y = helper.sample_batch(pool, args.microbatch, rng, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Gentle decay; fine-tune the v2 model, not the old step-2750 model.
        lr = args.lr * (0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"TRAIN {step}/{args.steps} loss={total_loss:.4f} elapsed={(time.perf_counter()-start)/60:.1f}m", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            record = evaluate(step)
            with (OUTPUT / "training_log.jsonl").open("a", encoding="utf-8") as logfile:
                logfile.write(json.dumps({k:v for k,v in record.items() if k!="transcripts"})+"\n")
            save_checkpoint(torch, OUTPUT / "latest.pt", model, config, step, record)
            # Promote only if actual generated title+fact accuracy improves.
            improved = (record["joint_rate"] > best["joint_rate"] + 0.04 and
                        record["validation_loss"] < baseline["validation_loss"] * 1.15)
            if improved:
                best = record
                save_checkpoint(torch, OUTPUT / "best.pt", model, config, step, record)
                print("Saved new BEST candidate based on actual generated answers.", flush=True)
    (OUTPUT / "result.json").write_text(json.dumps({
        "baseline": {k:v for k,v in baseline.items() if k!="transcripts"},
        "best": {k:v for k,v in best.items() if k!="transcripts"},
        "candidate_improved_on_small_validation_set": best["step"] != 0,
        "warning": "Small synthetic test; still requires real Ollama comparison before finalizing."
    }, indent=2), encoding="utf-8")
    if best["step"] == 0:
        print("NO MEASURED IMPROVEMENT: do not replace the original Ollama model.", flush=True)
    else:
        print("POTENTIAL IMPROVEMENT: checkpoint saved at title_switch_repair_run/best.pt. Do NOT deploy until separately tested.", flush=True)
    print("All previous checkpoints and Ollama models are unchanged.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--audit", action="store_true", help="Read-only source fact audit; no GPU required")
    modes.add_argument("--train", action="store_true", help="Train separate candidate from dialogues_v2_run/best.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--train-count", type=int, default=3000)
    p.add_argument("--validation-count", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=20261119)
    args = p.parse_args()
    if any(x < 1 for x in (args.steps, args.microbatch, args.grad_accum,
                            args.train_count, args.validation_count, args.eval_every)):
        raise SystemExit("Steps, batch sizes, and counts must be positive")
    if not 0 < args.lr < 1e-3:
        raise SystemExit("Learning rate must be between 0 and 0.001")
    prep = load_module(REPO / "prepare_netflix_dialogues_v2.py", "repair_data_helper")
    train_facts, val_facts = audit(prep, REPO)
    if args.train:
        train(args, prep, train_facts, val_facts)


if __name__ == "__main__":
    main()
