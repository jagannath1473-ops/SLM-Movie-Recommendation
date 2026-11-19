
# Netflix SLM V2 â€” 40.38M-Parameter Netflix Conversational Model

## Overview

This academic project develops a compact, domain-specific Small Language Model (SLM) to answer conversational questions about Netflix movies and TV shows. The project includes Netflix metadata preparation, supervised conversational training, multi-turn dialogue improvement, targeted title-switch repair, GGUF conversion, and a local Ollama demonstration.

**Final model:** `netflix-slm-40m-title-switch-f32.gguf`  
**Parameters:** 40,384,512 (approximately 40.38 million)  
**Format:** GGUF, F32  
**Deployment:** Ollama / Ollama Desktop  
**Training hardware for the improved model:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)

The earlier approximately 14.16M-parameter model remains in this repository for reference. Its files should not be confused with the final 40.38M GGUF model.

## Model architecture

The final model uses a Qwen3-inspired Transformer design:

| Component | Final configuration |
| --- | --- |
| Transformer layers | 12 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Key-value groups | 2 |
| Feed-forward dimension | 1,536 |
| Vocabulary size | 8,192 |
| Conversational fine-tuning context | 512 tokens |
| Architectural components | Grouped Query Attention, RoPE, RMSNorm, SwiGLU |

This is a student-trained, Qwen3-inspired architecture; it is **not** an official pretrained Qwen3 checkpoint.

## Data and preparation

- **Source reference:** [Netflix Movies and Series on Kaggle](https://www.kaggle.com/datasets/debayank2024/netflix-movies-and-series).
- **Locally identified selected Netflix CSV:** 1,500 catalog records, with fields including title, type, director, cast, country, release year, rating, and duration.
- **Prepared conversational dataset:** 29,986 examples, including Netflix question-answer interactions.
- **Prepared split:** 23,988 training, 2,999 validation, and 2,999 test examples.

The selected 1,500-row local catalog is distinct from the larger publicly discussed Netflix Titles collection. The raw CSV is not included in this repository; the source is linked above. The precise lineage of every generated example should be verified before making claims about complete coverage.

### Hosted processed datasets

- `data/train.jsonl`
- `data/validation.jsonl`
- `data/test.jsonl`
- `dialogues_v2_data/dialogues_train.jsonl`
- `dialogues_v2_data/dialogues_validation.jsonl`

The dialogue files support conversational training with follow-up questions and switching between Netflix titles.

## Development workflow

1. Select and inspect Netflix metadata.
2. Prepare conversational instruction/answer examples.
3. Train an initial small language model.
4. Build and train the larger approximately 40M-parameter Transformer.
5. Improve multi-turn dialogue handling and follow-up responses.
6. Perform targeted fine-tuning to reduce incorrect title switching.
7. Select the improved checkpoint and convert it to Hugging Face-compatible weights and GGUF.
8. Import and test the GGUF model in Ollama Desktop.

The targeted repair experiment saved its selected checkpoint at step 300.

## Evaluation â€” limited project tests

**Targeted synthetic validation (24 generated answers, selected step-300 checkpoint):**

| Metric | Result |
| --- | ---: |
| Title identification | 95.8% |
| Factual correctness | 87.5% |
| Joint title-and-fact correctness | 87.5% |
| Validation loss | 0.3934 |

**Separate local Ollama comparison (10 questions):** 10/10 correct title tracking and 9/10 correct factual answers. A separate five-question manual Ollama demonstration produced five satisfactory responses.

These results come from **small, project-specific tests**, not a large independent benchmark. The model can still produce incorrect information; for example, a tested response gave an incorrect release year for *Cobra Kai*.

## Run the final GGUF model in Ollama

Download `netflix-slm-40m-title-switch-f32.gguf` from this repository into a local folder.

Create a file named `Modelfile` in the same folder with:

```text
FROM ./netflix-slm-40m-title-switch-f32.gguf
PARAMETER temperature 0.2
```

Run these commands in that folder:

```bash
ollama create netflix-slm-40m-title-switch -f Modelfile
ollama run netflix-slm-40m-title-switch
```

Example prompts:

- Who directed Army of the Dead?
- How long is Army of the Dead?
- Is Cobra Kai a movie or a TV series?
- Is Grey's Anatomy a TV show?

The already registered local model can also be used in Ollama Desktop.

## Earlier model preserved in this repository

The original approximately 14.16M-parameter model is represented by older files such as `pytorch_model.bin`, `config.json`, `modeling_netflix_slm.py`, and `inference.py`. Its earlier README documented 1,000 training steps, CPU training, and best validation loss of 1.7522. Those figures **do not** describe the final 40.38M model. For the final deployment, use the GGUF file named above.

## Source code

[Netflix SLM GitHub repository](https://github.com/jagannath1473-ops/SLM-Movie-Recommendation)

The GitHub repository is intended to document the complete pipeline from data preparation through training, conversion, evaluation, and Ollama demonstration.

## Limitations

This is an experimental academic SLM, not an official Netflix model. Its answers are limited by the selected catalog and generated training data. It has no live connection to Netflix, may hallucinate titles or metadata, and should not be treated as a comprehensive or independently validated recommender. Verify factual answers against the underlying source data.

