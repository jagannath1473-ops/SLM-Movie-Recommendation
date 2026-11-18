"""Run several prompts with one checkpoint load."""

from pathlib import Path

import torch

from inference import generate_response, load_runtime


MODEL_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config, tokenizer, model = load_runtime(MODEL_DIR, device)

questions = [
    "Recommend a Netflix comedy movie.",
    "What is the plot of Stranger Things?",
    "Is The Crown a movie or a TV show?",
    "Recommend a romantic TV series on Netflix.",
]

for question in questions:
    response, stop_reason = generate_response(
        question,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
        seed=42,
    )
    print(f"QUESTION: {question}")
    print(f"RESPONSE: {response or '[empty response]'}")
    print(f"STOP: {stop_reason}\n")
