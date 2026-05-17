# 100M BPE FIM Smoke Report - 2026-05-05

## Summary
This run validates the requested architecture upgrade:

- BPE tokenizer path
- Fill-in-the-middle conditioning
- 2,048-token context length
- Approximately 100M parameters
- Modal CUDA training path

This is a plumbing and memory smoke test, not a quality benchmark. It ran only one optimization step on a tiny dataset.

## Modal Run
- Run URL: https://modal.com/apps/suryavanshi/main/ap-yobneO7e5lJuTLPNowOxCD
- GPU: T4
- Dataset: `codeparrot/codeparrot-clean`
- Train examples: 8
- Validation examples: 2
- Tokenizer: BPE
- Requested model vocab size: 16,384
- Actual tiny-run tokenizer vocab size: 5,888
- FIM rate: 1.0
- Context length: 2,048
- Batch size: 1
- Steps: 1

## Architecture
| Component | Value |
| --- | ---: |
| Parameters | 100,407,552 |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Feed-forward width | 3,072 |
| Max sequence length | 2,048 |
| Diffusion timesteps | 16 in Modal smoke, 32 in default config |
| Model vocab size | 16,384 |

## Result
The model initialized, trained a BPE tokenizer, built FIM manifests, ran one forward/backward/optimizer step on CUDA, evaluated, and checkpointed.

The loss values are intentionally not meaningful because this run used only 8 training examples and one step.

## Follow-up
1. Train the BPE tokenizer on at least thousands of examples so the actual tokenizer vocab approaches the requested 16,384 tokens.
2. Add gradient checkpointing or FlashAttention before doing long 2k-context runs with larger batches.
3. Run a real baseline: 10k train examples, 1k validation examples, 1k-5k steps, batch size 1-2, FIM rate 0.5.
