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
5. Move Modal outputs to a persistent `modal.Volume` if checkpoints from remote training should be reused locally.
6. Scale carefully: larger manifests first, then model width/depth, then schedule/loss-weight experiments.
