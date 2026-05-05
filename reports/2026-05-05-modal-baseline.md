# Modal Baseline Report - 2026-05-05

## Summary
This report tracks the first measurable masked-token diffusion baseline for `nanoDiffusionCodex`.

The model is not ready for HumanEval-style functional scoring yet. It is a 281k-parameter byte-level diffusion denoiser trained for only 100 steps, so the useful metric today is masked-token reconstruction quality. HumanEval should be added after the sampler can produce syntactically plausible function bodies.

## Training Run
- Runner: Modal
- Run URL: https://modal.com/apps/suryavanshi/main/ap-99tfJulwbdlX4hHCfQtmkj
- GPU: T4
- Dataset: `codeparrot/codeparrot-clean`
- Text field: `content`
- Train examples: 512
- Validation examples: 64
- Sequence length: 128 byte tokens
- Steps: 100
- Model parameters: 281,568
- Metric: token-weighted masked-token cross entropy and masked-token perplexity

## Results
| Run | Train Examples | Val Examples | Steps | Final Train Loss | Best Val Loss | Masked-Token Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial 20-step smoke | 64 | 16 | 20 | ~12.06 | 11.7853 | ~131,010 |
| Current baseline | 512 | 64 | 100 | 5.0213 | 4.2917 | 73.0875 |

The current baseline is a large improvement over the first smoke run, but it is still primarily a pipeline validation result. A perplexity of ~73 on byte tokens means the model has started learning local code-byte regularities, not that it can generate correct programs.

## Secondary Dataset Check
A one-step Modal sanity check confirmed that the second Hugging Face dataset path loads and trains:

- Run URL: https://modal.com/apps/suryavanshi/main/ap-d4bJtOyPjOp27bN5gt2hjZ
- Dataset: `code_search_net`
- Config: `python`
- Text field: `whole_func_string`
- Train examples: 8
- Validation examples: 2
- Steps: 1
- Result: training and validation completed on CUDA.

Use it locally or on Colab via:

```bash
python scripts/prepare_hf_dataset.py --preset codesearchnet-python --max-samples 2048 --val-samples 256
```

## Interpretation
- The end-to-end path is working: Hugging Face streaming, token manifests, noising, training, checkpointing, evaluation, and Modal GPU execution.
- More data and steps materially improve masked-token reconstruction.
- HumanEval is intentionally deferred because current checkpoints are not trained long enough for meaningful pass@k.

## Next Benchmark Targets
1. Train on 2k-10k examples for 1k-5k steps.
2. Save Modal checkpoints to a persistent `modal.Volume` so inference samples can be compared across runs.
3. Add syntax-validity rate on generated Python snippets before HumanEval.
4. Add HumanEval once generated completions can reliably produce parseable function bodies.
