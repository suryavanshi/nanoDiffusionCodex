# Handoff

## Current State
- Revised the project plan toward an MVP: byte-level tokenization, masked discrete diffusion, tiny Transformer denoiser, Hugging Face streaming data prep, and Modal smoke training.
- Implemented the actual training path under `src/nano_diffusion/`:
  - `data/tokenizer.py`: deterministic UTF-8 byte tokenizer with PAD/MASK/BOS/EOS.
  - `data/manifest.py` and `data/dataset.py`: JSONL token manifests and PyTorch dataset loader.
  - `diffusion/discrete.py`: timestep-conditioned masking process.
  - `models/transformer.py`: small Transformer denoiser with timestep embeddings.
  - `training/loop.py`: train/eval loop, masked CE loss, checkpointing, metrics JSONL.
  - `inference/sampler.py`: iterative confidence-based unmasking sampler.
- Added user-facing scripts:
  - `scripts/prepare_hf_dataset.py`
  - `scripts/train.py`
  - `scripts/eval.py`
  - `scripts/infer.py`
  - `scripts/modal_train.py`
- Added packaging/dependency metadata in `pyproject.toml` and `requirements.txt`.

## Modal Test Result
Main 20-step command run:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --total-steps 20 --max-samples 64 --val-samples 16
```

Result:
- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-VL7n55bPLVcMT5SeafYSSy
- Dataset: `codeparrot/codeparrot-clean`
- Train examples: 64
- Validation examples: 16
- Steps: 20
- Device: CUDA on Modal T4
- Parameters: 281,568
- Best validation loss: 11.78528118133545
- Training loss fell from about 56.8 to about 12.1 during the smoke run.

After adding BOS/EOS protection in the noising process, a shorter current-code smoke also passed:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --total-steps 5 --max-samples 32 --val-samples 8
```

- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-inppGzcdADk9BD9mflpNMa
- Device: CUDA on Modal T4
- Steps: 5
- Train examples: 32
- Validation examples: 8
- Best validation loss: 27.17182159423828
- Training loss fell from about 56.3 to about 35.0.

Larger baseline run requested after the initial implementation:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --total-steps 100 --max-samples 512 --val-samples 64 --max-seq-len 128
```

- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-99tfJulwbdlX4hHCfQtmkj
- Dataset: `codeparrot/codeparrot-clean`
- Train examples: 512
- Validation examples: 64
- Steps: 100
- Best validation loss: 4.291656715487971
- Masked-token perplexity: 73.08745336917994

Second dataset smoke:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --dataset-name code_search_net --dataset-config python --text-field whole_func_string --total-steps 1 --max-samples 8 --val-samples 2 --max-seq-len 128
```

- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-d4bJtOyPjOp27bN5gt2hjZ
- Result: `code_search_net` / `python` loads and trains through the same path.

Performance report: `reports/2026-05-05-modal-baseline.md`.

Colab notebook: `notebooks/colab_train.ipynb`.

100M BPE/FIM architecture smoke:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py --total-steps 1 --max-samples 8 --val-samples 2 --max-seq-len 2048 --tokenizer-type bpe --vocab-size 16384 --fim-rate 1.0 --model-dim 768 --model-layers 12 --model-heads 12 --batch-size 1
```

- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-yobneO7e5lJuTLPNowOxCD
- Result: ran BPE tokenizer training, FIM manifest generation, one CUDA train/eval/checkpoint step.
- Parameters: 100,407,552
- Context length: 2,048
- Requested model vocab: 16,384
- Actual tiny-run tokenizer vocab: 5,888
- Report: `reports/2026-05-05-100m-bpe-fim-smoke.md`.

Longer 100M BPE/FIM Modal run:

```bash
/Users/kb/Library/Python/3.9/bin/modal run scripts/modal_train.py::main \
  --experiment-name bpe-fim-100m-5k-1k \
  --total-steps 1000 \
  --max-samples 5000 \
  --val-samples 500 \
  --max-seq-len 2048 \
  --tokenizer-type bpe \
  --vocab-size 16384 \
  --fim-rate 0.5 \
  --model-dim 768 \
  --model-layers 12 \
  --model-heads 12 \
  --batch-size 1
```

- Modal app URL: https://modal.com/apps/suryavanshi/main/ap-JhDzyWDP63PZ00wytLvAZy
- Artifact volume/path: `nano-diffusion-codex-artifacts` / `bpe-fim-100m-5k-1k`
- Parameters: 100,407,552
- Train/validation examples: 5,000 / 500
- Final train loss: 11.302684783935547
- Final validation loss: 9.7302860938307
- Final masked-token perplexity: 16819.36354835089
- Local checkpoint artifacts downloaded to `checkpoints/bpe-fim-100m-5k-1k/`.
- Report: `reports/2026-05-05-100m-bpe-fim-5k-1k.md`.
- HumanEval was not run for this checkpoint; no `eval/` directory exists in the Modal volume.
- Modal cleanup check on 2026-05-05: `modal app list` and `modal container list` both showed no active apps/containers, so there were no Modal tasks to stop.

Perplexity-improvement patch after that run:
- Added `gradient_accumulation_steps` to `src/nano_diffusion/training/loop.py`; `total_steps` now counts optimizer updates, with each update accumulating multiple microbatches.
- Added cosine LR scheduling with warmup and a minimum LR.
- Added configurable `eval_batches`, so masked-token perplexity can be measured over more validation batches than the previous default of 8.
- Exposed `--gradient-accumulation-steps`, `--learning-rate`, `--min-learning-rate`, `--warmup-steps`, and `--eval-batches` in `scripts/modal_train.py::main`.
- Added `scripts/eval.py --max-batches`, defaulting to the checkpoint config's `training.eval_batches`.
- Updated the default YAML to use effective batch 16, LR `1e-4`, min LR `1e-5`, 100 warmup steps, and 64 eval batches.
- Modal CUDA smoke passed:
  - `bpe-fim-accum-smoke`: 2 optimizer updates, 2 microbatches per update.
  - `bpe-fim-accum-smoke-lrlog`: 1 optimizer update, confirmed warmup logging used `lr=5e-05` for step 1 with `warmup_steps=2`.
- Switched `run_modal_training` to request `gpu="A100"` and raised its timeout to 12 hours for long runs.
- Attempted attached A100 runs first; observed ~1.9 sec/update after warmup startup, projecting about 5.4 hours for 10k optimizer updates before eval/checkpoint overhead.
- Completed large A100 background run:
  - App URL: https://modal.com/apps/suryavanshi/main/ap-2vToUgtB8eUILaVKXl361E
  - Function call id: `fc-01KQW79JCJRYTJB5X1C8B6SDPT`
  - Experiment: `bpe-fim-100m-a100-50k-10k-bg`
  - Volume/path: `nano-diffusion-codex-artifacts` / `bpe-fim-100m-a100-50k-10k-bg`
  - Config: 50k train examples, 1k val examples, 10k optimizer updates, context 2048, BPE vocab 16384, FIM rate 0.5, batch size 1, gradient accumulation 16, LR `1e-4`, min LR `1e-5`, warmup 500, eval batches 64.
  - Final train loss: 7.908954381942749
  - Final validation loss: 7.153740362882263
  - Final masked-token perplexity: 1278.8804974902407
  - Best validation loss/perplexity matched the final eval.
  - Checkpoints exist in the Modal volume: `runs/best.pt`, `runs/best_model_only.pt`, `runs/last.pt`, and `runs/metrics.jsonl`.
  - Local artifacts downloaded to `checkpoints/bpe-fim-100m-a100-50k-10k-bg/`: `best_model_only.pt`, `tokenizer.json`, `config.json`, `training_result.json`, and `metrics.jsonl`.
  - Final report: `reports/2026-05-13-100m-bpe-fim-a100-50k-10k.md`.
  - Modal cleanup check on 2026-05-06: no active apps and no active containers.
- Created heartbeat automation `check-a100-diffusion-training`; if it fires after this handoff, it should avoid duplicating work unless local checkpoint downloads/reports are still missing.

Architecture/perplexity diagnosis:
- The architecture does not look obviously broken. Random loss for a 16,384-token vocabulary is `ln(16384) = 9.70`, and the A100 run reached validation loss 7.15, so it learned meaningful signal.
- The masked-token perplexity is still high because this is a 16k-BPE FIM masked-token objective, not standard causal LM perplexity. FIM only supervises the middle target span and is a harder eval.
- Biggest implementation concern: training loss during gradient accumulation currently averages each microbatch loss equally, while validation is token-weighted by masked-token count. Fix this first so optimization and reported train loss match the eval objective.
- A100 utilization check during the run showed strong compute utilization, about 92-96% GPU util, but only about 4GB/40GB memory used. Next throughput experiment should try `batch_size=2, gradient_accumulation_steps=8` and `batch_size=4, gradient_accumulation_steps=4`.
- Architecture is plain: bidirectional `TransformerEncoder`, learned absolute positions, GELU FFN, pre-norm. After loss accounting and batch-size tuning, consider RoPE, RMSNorm, SwiGLU, and FlashAttention.

## Local Verification
Passed:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/nano_pycache python3 -m compileall -q scripts src tests
python3 scripts/prepare_hf_dataset.py --help
python3 scripts/train.py --help
python3 scripts/eval.py --help
python3 scripts/infer.py --help
PYTHONPATH=src python3 -c "from nano_diffusion.data.tokenizer import ByteTokenizer; t=ByteTokenizer(); ids,mask=t.encode('def f():\n    pass\n', 32); assert t.decode(ids).startswith('def f'); print('tokenizer ok', len(ids), sum(mask))"
```

Not run locally:
- `python3 -m pytest -q` because `pytest` is not installed in the system Python.
- Local training because `torch`, `datasets`, and `PyYAML` are not installed locally. The Modal image installed and exercised these dependencies successfully.

## Next Best Steps
1. Install local dev dependencies or use a virtualenv so `pytest` and short CPU tests can run locally.
2. Add unit tests for `MaskingDiffusion.q_sample`, checkpoint reload, and one 1-step synthetic training run.
3. Add a small synthetic-data mode for training tests that does not depend on Hugging Face availability.
4. Improve sampling quality: add temperature/top-k options, stop at generated EOS, and compare deterministic versus stochastic unmasking.
5. Fix gradient accumulation to weight losses by masked-token count, then rerun a short sanity job and compare train/eval loss behavior.
6. Benchmark A100 throughput with larger microbatches while keeping effective batch 16: try `batch_size=2, grad_accum=8` and `batch_size=4, grad_accum=4`.
7. Scale carefully: larger manifests first, then schedule/loss-weight experiments, then full HumanEval once generations are parseable.
