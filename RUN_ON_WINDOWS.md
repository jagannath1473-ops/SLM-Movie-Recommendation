# Run the corrected Netflix SLM on Windows

The corrected script does not retrain or alter the model. Put it beside the existing checkpoint and supporting files.

## Your current folder layout

Your trained export is already here:

`C:\Users\jagan\SLM-Movie-Recommendation\huggingface_netflix_slm_v2`

Your working Python environment is already here:

`C:\Users\jagan\SLM-Movie-Recommendation\.venv`

## Replace only the inference script

Open PowerShell and run these commands after changing the first path to wherever you downloaded this fix folder:

```powershell
$fix = "C:\path\to\SLM_Inference_Fix"
$project = "C:\Users\jagan\SLM-Movie-Recommendation"
$model = Join-Path $project "huggingface_netflix_slm_v2"

# Preserve the original inference script once.
if (-not (Test-Path "$model\inference_original.py")) {
    Copy-Item "$model\inference.py" "$model\inference_original.py"
}

Copy-Item "$fix\inference.py" "$model\inference.py" -Force
Copy-Item "$fix\smoke_test.py" "$model\smoke_test.py" -Force
```

No other model file should be replaced.

## Run one question

```powershell
& "$project\.venv\Scripts\python.exe" "$model\inference.py" `
  "Recommend a Netflix comedy movie." `
  --temperature 0.8 `
  --top-k 40 `
  --top-p 0.95 `
  --repetition-penalty 1.12 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --show-details
```

## Run all smoke-test questions

```powershell
& "$project\.venv\Scripts\python.exe" "$model\smoke_test.py"
```

## Useful settings

- Reproducible sampling: keep the same `--seed`.
- Try another valid answer: change `--seed`, for example `--seed 7`.
- Greedy decoding: use `--temperature 0`. This can reproduce the checkpoint's dominant repetitive answer.
- More conservative sampling: try `--temperature 0.7 --top-k 20 --top-p 0.9`.
- More variety: try `--temperature 0.9 --top-k 50`, but hallucination risk increases.

The system prompt now defaults to the most common system instruction in the training dataset. It can be changed with `--system-prompt`, but keeping the default best matches training.

## Files that must stay beside inference.py

- `pytorch_model.bin`
- `modeling_netflix_slm.py`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json` (recommended metadata; the corrected script does not depend on it)
- `chat_template.jinja`

## Important interpretation

Sampling controls can reduce identical repeated outputs. They cannot repair facts or instruction-following knowledge that the checkpoint did not learn. Treat every title, year, plot, cast member, and recommendation as unverified.
