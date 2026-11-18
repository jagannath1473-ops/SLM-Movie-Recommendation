# Netflix SLM inference inspection report

## Scope and preservation

Inspected the GitHub `main` branch files, the full training notebook, the local exported Hugging Face model folder, the tokenizer, the 56.7 MB checkpoint, and the local JSONL/token binaries. No GitHub or Hugging Face files were changed or pushed. The checkpoint, architecture, tokenizer vocabulary, dataset, and parameter count remain unchanged.

Verified parameter count: **14,162,432**.

Verified dataset examples: **29,986**.

## Prompt format and special tokens

The V2 training notebook formats each conversation as:

```text
<|system|>
SYSTEM INSTRUCTION<|end|>
<|user|>
USER QUESTION<|end|>
<|assistant|>
ASSISTANT ANSWER<|eos|>
```

Verified token IDs in `tokenizer.json`:

| ID | Token |
|---:|---|
| 0 | `<|pad|>` |
| 1 | `<|unk|>` |
| 2 | `<|bos|>` |
| 3 | `<|eos|>` |
| 4 | `<|system|>` |
| 5 | `<|user|>` |
| 6 | `<|assistant|>` |
| 7 | `<|end|>` |

The tokenizer's BPE vocabulary has 8,192 entries. The tokenizer and chat template agree on `<|end|>` after non-assistant messages and `<|eos|>` after assistant messages.

## Main causes of repetitive and irrelevant output

1. **Inference omitted the system message used in every training example.** All 29,986 dataset conversations contain a system message. The most common exact system prompt appears 12,144 times. The old inference script began directly with `<|user|>`, making its prompt distribution different from training.

2. **The exported inference script used pure greedy decoding.** `argmax` always selects the same highest-probability continuation and exposes any dominant learned attractor. There was no temperature, top-k/top-p sampling, repetition penalty, no-repeat n-gram rule, or configurable seed.

3. **The checkpoint is substantially undertrained for instruction following.** Training used 1,000 steps, batch size 2, gradient accumulation 4, and context 128: 1,024,000 sampled token positions. The train binary contains 1,806,889 tokens, so training performed only about **0.57 nominal passes** over the train-token count, with random sampling and replacement.

4. **Loss was applied to every token, not only assistant-answer tokens.** The model spent capacity predicting system prompts and user questions as well as answers. This is valid causal-language-model training, but it is weaker instruction tuning for a small model.

5. **Random 128-token windows can begin mid-example and cross conversation boundaries.** The three splits are separate, but examples inside each split were concatenated without a dedicated document-boundary sampler. Some training batches therefore lack a complete system/user/assistant structure.

6. **The validation loss does not measure answer relevance.** The reported 1.7522 loss is next-token loss on random validation windows. It does not demonstrate correct movie selection, plot answering, or recommendation quality.

7. **The “Last Boy” behavior is a learned checkpoint attractor, not merely one duplicated answer.** “The Last Boy” occurs in 25 assistant answers and 12 user questions, largely as “The Last Boy Scout.” It is not the dataset's most common exact answer, yet the checkpoint repeatedly reconstructs malformed “The Last…” titles. That points to incomplete learning/mode concentration rather than a simple inference bug.

## Corrected inference behavior

The provided `inference.py`:

- restores the most common training system prompt by default;
- preserves the exact trained chat markers;
- always retains role markers when truncating a long user prompt;
- validates special-token IDs before loading the model;
- loads the original checkpoint strictly, without architecture changes;
- supports CPU, CUDA, or automatic device selection;
- supports temperature, top-k, top-p, seed, repetition penalty, and no-repeat n-grams;
- stops on `<|eos|>` and safely stops on unexpected role/end/pad tokens;
- reports the stop reason with `--show-details`;
- loads `tokenizer.json` directly, avoiding the misleading Transformers custom-model warning.

## Actual checkpoint responses

### Original inference.py (greedy)

| Question | Actual response |
|---|---|
| Recommend a Netflix comedy movie. | `'The Last Boy' is a Movie.` |
| What is the plot of Stranger Things? | `'The Last Stray' runs 97 minutes.` |
| Is The Crown a movie or a TV show? | `'The Last Boy' is a Movie.` |

### Corrected inference.py defaults

Settings: temperature 0.8, top-k 40, top-p 0.95, repetition penalty 1.12, no-repeat 3-gram, seed 42, and the most common training system prompt.

| Question | Actual response |
|---|---|
| Recommend a Netflix comedy movie. | `I don't have sequels or synopsis for "FY on the Bees"; my data only covers catalog metadata. What I can tell you is that it is a Movie, was released in 2018, and runs 85 minutes.` |
| What is the plot of Stranger Things? | `"The Last Courageous" isn't in my data. I'd rather not guess - if you check the spelling or give me another title, I'll look it up.` |
| Is The Crown a movie or a TV show? | `"The Last Boy de 'The Last Bastion' is a Movie.` |
| Recommend a romantic TV series on Netflix. | `In my data, S.o appears in 1 title: The Last Know (2021).` |

All four corrected runs stopped on `<|eos|>`.

## Verified improvements versus unresolved limits

Verified:

- Source compiles and strictly loads the original weights.
- Special tokens match the checkpoint.
- The prompt now matches the training role structure.
- Generation settings are configurable and seeded.
- Responses are no longer forced to be identical by greedy decoding.
- End-of-response handling is correct and observable.
- No checkpoint, parameter, architecture, tokenizer, or dataset change was made.

Not verified as improved:

- Factual accuracy.
- Correct title selection.
- Plot-question answering.
- Recommendation relevance.
- Generalization across the test split.

The real responses above remain poor and sometimes malformed. This is checkpoint quality, not something decoding alone can reliably repair.

## Next step toward Ollama

Do **not** start with a Modelfile pointing at `pytorch_model.bin`; Ollama cannot directly execute this custom PyTorch architecture.

The next step is a **no-retraining compatibility conversion**:

1. Create a standard Hugging Face `Qwen3ForCausalLM`-compatible config with the same dimensions and parameter count.
2. Map the custom state-dict tensor names to the standard Qwen3 names.
3. Compare logits from the custom model and mapped model on identical token sequences. Proceed only if they match within a small numerical tolerance.
4. Convert the verified standard model to GGUF.
5. Create an Ollama Modelfile that points to the GGUF and includes the exact system/user/assistant template and stop tokens.
6. Re-run the same prompt suite in Python and Ollama and compare outputs.

This conversion preserves the trained weights. If exact logit equivalence cannot be achieved, direct Ollama integration requires adding this custom architecture to llama.cpp/Ollama rather than pretending it is a supported model.
