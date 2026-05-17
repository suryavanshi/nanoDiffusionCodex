# nanoDiffusionCodex

A compact, code-first diffusion project scaffold for training, evaluation, and inference experiments.

nanoDiffusionCodex is designed as a clean starting point: lightweight structure, explicit plans, and simple script entrypoints so you can iterate quickly.

## ✨ What’s in this repository

- **Project roadmap** with milestones and implementation notes.
- **Implemented MVP diffusion pipeline** for byte-level code denoising.
- **CLI-style script entrypoints** for dataset prep, train, eval, infer, and Modal training.
- **Colab notebook** for GPU training in Google Colab.
- **Baseline reports** for tracking Modal training performance over time.
- **Smoke tests** for basic integrity checks.

## 📌 Current status

This repository now has a first real MVP path: a byte-level tokenizer, masked discrete diffusion objective, tiny Transformer denoiser, Hugging Face dataset preparation script, training/eval/inference entrypoints, and a Modal smoke-training script.

If you are extending this repo, start with:

1. `plan.md` to understand priorities,
2. `docs/architecture.md` for module boundaries,
3. `scripts/train.py`, `scripts/eval.py`, and `scripts/infer.py` as integration entrypoints.

The original scaffold next steps are now complete: the end-to-end training loop exists, the baseline dataset interface lives under `src/nano_diffusion/data/`, YAML config loading is implemented, and smoke tests cover the tokenizer path.

## 🧠 Model Architecture

The default config in `src/nano_diffusion/configs/default.yaml` now defines a BPE-tokenized masked diffusion denoiser with fill-in-the-middle support:

| Component | Default |
| --- | ---: |
| Tokenizer | BPE |
| Vocabulary | 16,384 tokens |
| Sequence length | 2,048 |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Feed-forward width | 3,072 |
| Diffusion timesteps | 32 |
| Schedule | cosine masking |
| Dropout | 0.1 |
| Parameters | 100,407,552 |

Architecture details:
- BPE tokenization is trained from the selected Hugging Face code slice and saved to `data/processed/tokenizer.json`.
- Fill-in-the-middle samples use `<fim_prefix>prefix<fim_suffix>suffix<fim_middle>middle`; the model conditions on prefix/suffix and denoises only the middle target span.
- Forward diffusion samples a timestep and masks eligible denoising targets according to the schedule.
- The denoiser uses token embeddings, learned position embeddings, and learned timestep conditioning.
- The backbone is a bidirectional `torch.nn.TransformerEncoder` with GELU feed-forward blocks and pre-norm layers.
- The output head is tied to the token embedding matrix.
- Evaluation reports token-weighted masked-token cross entropy and masked-token perplexity.

Reports include both the earlier tiny baseline and the current 100M BPE/FIM smoke test:
- `reports/2026-05-05-modal-baseline.md`
- `reports/2026-05-05-100m-bpe-fim-smoke.md`
- `reports/2026-05-05-100m-bpe-fim-5k-1k.md`
- `reports/2026-05-13-100m-bpe-fim-a100-50k-10k.md`

## Latest Result

The latest large Modal run completed on an A100:

| Metric | Value |
| --- | ---: |
| Experiment | `bpe-fim-100m-a100-50k-10k-bg` |
| Parameters | 100,407,552 |
| Train examples | 50,000 |
| Validation examples | 1,000 |
| Optimizer updates | 10,000 |
| Context length | 2,048 |
| Effective batch | 16 |
| Final train loss | 7.90895 |
| Final validation loss | 7.15374 |
| Final masked-token perplexity | 1,278.88 |

This is a clear improvement over the earlier 5k-example run, which had masked-token perplexity around 16,819. It is still high. The main known issue to fix before reading too much into train loss is that training currently averages microbatch losses equally during gradient accumulation, while evaluation is token-weighted. That means train loss and validation loss are not perfectly comparable when the number of masked tokens varies across microbatches.

For reference, random prediction over a 16,384-token vocabulary has loss `ln(16384) = 9.70`; the final validation loss of `7.15` is better than random, but still not close to a useful code model.

## 🚀 Quick start

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_hf_dataset.py --max-samples 512 --val-samples 64 --max-seq-len 2048
python scripts/train.py
python scripts/eval.py --checkpoint runs/nano-diffusion-bpe-fim/best.pt
python scripts/infer.py --checkpoint runs/nano-diffusion-bpe-fim/best.pt --prompt "def add(a, b):\n    "
```

To run the small remote smoke test on Modal. The training entrypoint currently requests an A100 GPU for the larger 2k-context model:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --total-steps 20
```

For a less noisy 2k-context run, keep the CUDA batch at 1 and raise the effective batch with gradient accumulation:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py::main \
  --experiment-name bpe-fim-100m-accum \
  --max-samples 50000 \
  --val-samples 1000 \
  --total-steps 10000 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 0.0001 \
  --min-learning-rate 0.00001 \
  --warmup-steps 500 \
  --eval-batches 64
```

The completed large A100 run was:

```bash
/Users/kb/Library/Python/3.9/bin/modal run --detach scripts/modal_train.py::main \
  --background \
  --experiment-name bpe-fim-100m-a100-50k-10k-bg \
  --max-samples 50000 \
  --val-samples 1000 \
  --total-steps 10000 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 0.0001 \
  --min-learning-rate 0.00001 \
  --warmup-steps 500 \
  --eval-batches 64
```

For Colab training, open [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb). The data prep script supports two built-in Hugging Face presets:

```bash
python scripts/prepare_hf_dataset.py --preset codeparrot-clean
python scripts/prepare_hf_dataset.py --preset codesearchnet-python
```

Baseline Modal performance reports live in [reports/](reports/).

## 🗂️ Repository layout

```text
.
├── README.md
├── plan.md
├── docs/
│   └── architecture.md
├── notebooks/
│   └── colab_train.ipynb
├── reports/
│   ├── 2026-05-05-modal-baseline.md
│   ├── 2026-05-05-100m-bpe-fim-smoke.md
│   ├── 2026-05-05-100m-bpe-fim-5k-1k.md
│   └── 2026-05-13-100m-bpe-fim-a100-50k-10k.md
├── scripts/
│   ├── prepare_hf_dataset.py
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   └── modal_train.py
├── src/
│   └── nano_diffusion/
│       ├── configs/
│       ├── data/
│       ├── diffusion/
│       ├── evaluation/
│       ├── inference/
│       ├── models/
│       ├── training/
│       └── utils/
└── tests/
    └── test_smoke.py
```

## 🧭 Suggested next steps

- Add unit tests for BPE tokenizer training, FIM manifest masks, `MaskingDiffusion.q_sample`, checkpoint reload, and one synthetic 1-step training run.
- Fix gradient accumulation to weight each microbatch by masked-token count, matching evaluation's token-weighted loss.
- Run a short A100 throughput comparison with `--batch-size 2 --gradient-accumulation-steps 8` and `--batch-size 4 --gradient-accumulation-steps 4`; prior utilization showed high compute use but only about 4GB of 40GB A100 memory used.
- Train larger baselines: 50k-200k examples for 10k+ optimizer updates, then track masked-token perplexity in `reports/`.
- Consider modernizing the backbone after the loss accounting fix: RoPE, RMSNorm, SwiGLU, and FlashAttention.
- Add gradient checkpointing or FlashAttention before longer 2k-context runs with larger batches.
- Add syntax-validity rate for generated Python before running HumanEval.
- Run HumanEval/pass@k once generated completions are consistently parseable function bodies.

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
