# Netflix SLM V2 — 40.38M-Parameter Conversational Movie & TV Model

> **Final trained model: 40,384,512 parameters (40.38 million).** A compact, student-trained, Qwen3-inspired Transformer for Netflix catalog question answering, conversational follow-ups, and title-aware responses. This is **not** an official Netflix model or an official pretrained Qwen3 checkpoint.

## Project links

- **Model weights (GGUF) and processed datasets:** [Hugging Face — MMJagannath/Netflix-SLM-V2](https://huggingface.co/MMJagannath/Netflix-SLM-V2)
- **Source code:** [GitHub — SLM-Movie-Recommendation](https://github.com/jagannath1473-ops/SLM-Movie-Recommendation)
- **Catalog source reference:** [Netflix Movies and Series on Kaggle](https://www.kaggle.com/datasets/debayank2024/netflix-movies-and-series)

## 1. Overview

This academic project develops and deploys a small, domain-specific language model to answer questions about Netflix movies and television series. Its development stages include metadata preparation, conversational question–answer generation, training a smaller baseline model, training and improving a **40.38M-parameter model**, targeted title-switch fine-tuning, evaluation, conversion to GGUF, and local Ollama chat demonstration.

**Final artifact:** `netflix-slm-40m-title-switch-f32.gguf` (~162 MB, F32 GGUF)  
**Registered local Ollama name:** `netflix-slm-40m-title-switch-candidate`  
**Training hardware for improved model:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)

## 2. Final model architecture — exact specifications

| Component | Final model specification |
| --- | --- |
| **Total trainable model parameters** | **40,384,512 (40.38M)** |
| Architecture | Student-trained, Qwen3-inspired decoder-only Transformer |
| Transformer layers | 12 |
| Hidden/model dimension | 512 |
| Query attention heads | 8 |
| Attention head dimension | 64 |
| Key-value attention heads | 2 |
| Query heads per KV head | 4 |
| Feed-forward/intermediate dimension | 1,536 |
| Vocabulary size | 8,192 |
| Fine-tuning context window | 512 tokens |
| Attention mechanism | Grouped Query Attention (GQA) |
| Position encoding | Rotary Position Embeddings (RoPE), theta = 10,000 |
| Normalization | RMSNorm, epsilon = 1e-6 |
| Feed-forward activation | SwiGLU / SiLU gating |
| Attention dropout in Qwen3-compatible export | 0.0 |
| Embedding/output projection | Tied embeddings in the Qwen3-compatible conversion |
| Exported deployment artifact | F32 GGUF, approximately 162 MB |

The architecture was trained for this project and later mapped into a Qwen3-compatible representation for GGUF/Ollama interoperability. **Compatibility with the Qwen3 file architecture does not mean the model uses pretrained official Qwen3 weights.** The earlier ~14.16M-parameter baseline is a separate model.

## 3. Dataset and preparation

The selected local Netflix CSV contained **1,500 catalog records** with fields including title, type, director, cast, country, release year, maturity rating, and duration. These were used in a project that produced **29,986 conversational examples**. The original Kaggle source is linked above; do not confuse the selected 1,500-row CSV with the size of larger public Netflix catalog datasets.

| Prepared dataset split | Examples |
| --- | ---: |
| Training | 23,988 |
| Validation | 2,999 |
| Test | 2,999 |
| **Total** | **29,986** |

Processed files hosted within the **Hugging Face model repository**:

- `data/train.jsonl`
- `data/validation.jsonl`
- `data/test.jsonl`
- `dialogues_v2_data/dialogues_train.jsonl`
- `dialogues_v2_data/dialogues_validation.jsonl`

The additional dialogue datasets support multi-turn questions, follow-ups, and switches between Netflix titles. The raw CSV is not redistributed in this GitHub repository. The exact lineage of every generated example should be verified before making stronger claims about catalog coverage or independence.

## 4. Training workflow and verified hyperparameters

The uploaded Python scripts document several stages. **Values described as defaults below are command-line defaults, not evidence that every stage ran for exactly that many updates.** The selected final title-switch repair checkpoint was saved at **step 300**.

| Setting | Long-chat 512-token stage | Dialogue V2 stage | Title-switch repair |
| --- | --- | --- | --- |
| Script | `train_netflix_longchat_512.py` | `train_netflix_dialogues_v2.py` | `netflix_title_switch_repair.py` |
| Starting checkpoint | Original 40.38M step-2750 | Original 40.38M step-2750 | Dialogue V2 `best.pt` |
| Default update steps | 300 | 2,500 | 400; **selected checkpoint step 300** |
| Microbatch size | 4 | 4 | 4 |
| Gradient accumulation | 2 | 4 | 4 |
| Effective sequences per update | 8 | 16 | 16 |
| Optimizer | AdamW | AdamW | AdamW |
| Initial learning rate | 5e-5 | 3e-5 | 2e-5 |
| Adam betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| Weight decay | 0.05 | 0.05 | 0.05 |
| LR schedule | Warm-up + cosine decay to 1e-5 | Warm-up + cosine decay to 5e-6 | Cosine decay, 25% floor |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Mixed precision | CUDA BF16 autocast | CUDA BF16 autocast | CUDA BF16 autocast |
| Context length | 512 tokens | 512 tokens | 512 tokens |
| Default random seed | 20261119 | 20261119 | 20261119 |

The Dialogue V2 stage uses a default **80% long-dialogue sampling ratio**, mixing newly prepared multi-turn conversations with original training examples. The repair script defaults to **3,000 fact-grounded title-switch training dialogues** and **80 validation dialogues**, generated from existing question–answer facts rather than external invented movie facts.

### Pipeline at a glance

1. Inspect the selected Netflix metadata and prepare structured Q&A records.
2. Train the earlier ~14.16M baseline model and prepare the larger 40.38M architecture.
3. Train the 40.38M model and retain its checkpoint.
4. Generate longer dialogues and fine-tune for follow-up questions and title tracking.
5. Fine-tune a separate title-switch repair candidate, selecting the step-300 checkpoint.
6. Map model tensors to Qwen3-compatible weights and export to GGUF.
7. Register the final GGUF in Ollama with the **custom role-token chat template** used during training.
8. Compare answers and evaluate model behavior on small, project-specific tests.

## 5. Evaluation results — small project-specific tests

**Selected title-switch repair checkpoint (step 300):**

| Metric | Result |
| --- | ---: |
| Generated-response title identification (24 answers) | 95.8% |
| Generated-response factual correctness (24 answers) | 87.5% |
| Joint title-and-fact correctness (24 answers) | 87.5% |
| Validation loss | 0.3934 |

**Separate local Ollama test (10 questions):** 10/10 correct title tracking; 9/10 factually correct answers.  
**Additional manual Ollama demonstration:** 5/5 satisfactory responses on five questions.

These are **small, project-specific, partly synthetic tests**, not a large independently sampled benchmark or a guarantee of general accuracy. One observed error was an incorrect release year for *Cobra Kai*. More extensive evaluation against an independent, manually verified test set is necessary before making stronger performance claims.

**Baseline context:** The earlier ~14.16M model was separately reported with a validation loss of 1.7522 at 1,000 training steps. This number is **not directly comparable** to the final repair-stage validation loss without matching evaluation datasets and procedures.

## 6. Source-code inventory

| Stage | Uploaded scripts |
| --- | --- |
| Dataset preparation | `preprocess_dataset.py`, `prepare_netflix_longchat.py`, `prepare_netflix_longchat_export.py`, `prepare_netflix_dialogues_v2.py` |
| Training | `train_netflix_longchat_512.py`, `train_netflix_dialogues_v2.py`, `netflix_title_switch_repair.py` |
| Checks and evaluation | `check_longchat_512.py`, `evaluate_netflix_dialogues_v2.py`, `smoke_test.py` |
| Model conversion | `convert_to_qwen3_longchat.py` |
| Ollama deployment and fixes | `deploy_netflix_slm_40m_v2.py`, `deploy_netflix_title_switch_candidate.py`, `fix_netflix_ollama_template.py` |
| Earlier baseline | `SLM_Training_Part1.ipynb`, `modeling_netflix_slm.py`, `inference.py`, `config.json`, `tokenizer.json`, `chat_template.jinja` |

The older source files are retained for historical context. **The `config.json` and `modeling_netflix_slm.py` at the root of this GitHub repository describe the earlier model, not the final 40.38M GGUF artifact.** For final weights, use the Hugging Face GGUF file.

### Important reproducibility limitation

Some final training scripts load a local helper called `train_to_3000.py` and a saved step-2750 checkpoint from a separate Windows directory. **Those original training dependencies are not currently included in this GitHub repository.** The scripts also reference local CUDA/PyTorch installation paths. Accordingly, the repository documents the later stages but is **not yet a fully self-contained from-scratch training reproduction**. The final trained GGUF and processed datasets are available on Hugging Face.

## 7. Ollama deployment and live chat

The final GGUF is hosted on [Hugging Face](https://huggingface.co/MMJagannath/Netflix-SLM-V2). Download `netflix-slm-40m-title-switch-f32.gguf` to a local folder, and create a file named `Modelfile` in that same folder:

```text
FROM ./netflix-slm-40m-title-switch-f32.gguf
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|user|>
{{ .Content }}<|end|>
{{ else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|eos|>
{{ end }}{{ end }}<|assistant|>
"""
SYSTEM "You answer Netflix catalog questions concisely and factually. Follow the latest title mentioned in the conversation."
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
```

Run in a terminal from the folder containing both files:

```bash
ollama create netflix-slm-40m-title-switch -f Modelfile
ollama run netflix-slm-40m-title-switch
```

On the original development laptop, the already registered model is named `netflix-slm-40m-title-switch-candidate` and can be used in Ollama Desktop. The template above is important because the model was trained using custom `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, and `<|eos|>` markers, not the stock Qwen3 chat format.

**Example prompts:**

- Who directed *Army of the Dead*?
- How long is *Army of the Dead*?
- Is *Cobra Kai* a movie or a TV series?
- Now switch to *Grey's Anatomy*. Is it a TV show?

## 8. Limitations and responsible interpretation

This is an experimental student-built domain-specific SLM. It does not access the live Netflix catalog, may confuse titles or dates, and does not provide a verified real-time recommendation service. Results depend on the selected catalog and generated dialogues. Training stages and validation sets are not directly comparable without controlled experiments. Dataset rights remain subject to their original source terms; publishing code or weights does not change upstream dataset licensing.
