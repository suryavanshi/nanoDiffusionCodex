# Nano Diffusion for Code — Project Plan

## Vision
Build a **nano-scale diffusion model for code generation/editing** with fast iterative decoding, compact parameter count, and practical training/inference workflows suitable for a small team.

## Product Goals
- **Primary use case:** code completion and short-to-medium code generation from natural language + context.
- **Latency target:** interactive local/edge inference with efficient iterative denoising.
- **Model footprint target (initial):** 100M–500M params (v0), with clear path to larger variants.
- **Quality target:** competitive on HumanEval/MBPP class tasks for its size band.

## Core Technical Strategy
1. **Representation**
   - Current implementation uses a Hugging Face-trained BPE tokenizer with 16,384 tokens.
   - Keep byte-level tokenization available for smoke tests and debugging.
   - Add optional AST-aware auxiliary objectives later.

2. **Diffusion Objective for Discrete Tokens**
   - Current implementation uses masked discrete diffusion over token sequences.
   - Fill-in-the-middle conditioning is implemented with prefix/suffix visible and only the middle span denoised.
   - Evaluation reports token-weighted masked-token loss and perplexity.
   - Near-term fix: make training gradient accumulation token-weighted as well; it currently averages microbatch losses equally.

3. **Backbone Architecture**
   - Current baseline is a bidirectional `TransformerEncoder` denoiser with token, learned position, and timestep embeddings.
   - 100M-parameter config: 12 layers, hidden size 768, 12 heads, 2,048 context, tied output head.
   - The architecture is functional but basic; after loss accounting is fixed, consider RoPE, RMSNorm, SwiGLU, and FlashAttention.
   - Keep modularity for future hybrid recurrent/state-space blocks.

4. **Sampling/Decoding**
   - Iterative denoising sampler for code tokens.
   - Track tradeoff curves: steps vs quality vs latency.
   - Add speculative/early-exit heuristics after baseline stability.

## Roadmap

### Phase 0 — Foundations (Week 1)
- Define repository architecture and coding standards.
- Implement configuration system and experiment tracking stubs.
- Add data pipeline skeleton and tokenizer tooling.
- Create baseline training/evaluation CLI entrypoints.

### Phase 1 — MVP Diffusion Training Loop (Complete)
- Implemented byte-level and BPE tokenization paths.
- Implemented masked discrete diffusion: sample a timestep, mask eligible code tokens according to a cosine schedule, and train the model to reconstruct masked positions.
- Implemented fill-in-the-middle manifests and denoise masks.
- Implemented a bidirectional Transformer denoiser with token, position, and timestep embeddings.
- Prepared Hugging Face code slices via streaming download into reproducible JSONL token manifests.
- Validated end-to-end train/eval/infer loop on Modal, including a 100M-parameter A100 run.

Latest completed A100 run:
- Experiment: `bpe-fim-100m-a100-50k-10k-bg`
- Data: 50k train examples, 1k validation examples
- Model: 100,407,552 parameters, 2,048 context, 16,384 BPE vocab
- Training: 10k optimizer updates, effective batch 16
- Final validation loss: 7.153740362882263
- Final masked-token perplexity: 1278.8804974902407

### Phase 2 — Quality + Stability (Weeks 5–8)
- Fix token-weighted training loss under gradient accumulation so optimization matches validation accounting.
- Benchmark larger A100 microbatches: `batch_size=2, grad_accum=8` and `batch_size=4, grad_accum=4`.
- Expand dataset quality filters and dedup, then scale from 50k to 200k+ examples.
- Add syntax-validity checks before HumanEval/MBPP-style benchmarks.
- Introduce checkpoint averaging and richer run reports.

### Phase 3 — Inference Optimization (Weeks 9–12)
- Optimize sampler implementation and caching.
- Add quantization and batched serving benchmarks.
- Package inference API + simple playground.

### Phase 4 — Advanced Research Extensions
- Distillation from stronger teacher models.
- Structural conditioning (AST, retrieval, repo context).
- Controlled editing modes (fill-in-the-middle, patch generation).

## Evaluation Plan
- **Functional correctness:** pass@k on coding benchmarks.
- **Latency:** median and p95 end-to-end generation time by token budget and denoising steps.
- **Efficiency:** throughput, memory footprint, and parameter-efficiency curves.
- **Safety/quality checks:** syntax validity, test compilation pass rate, basic policy filters.

## Data Strategy
- Start with permissively licensed public code sources.
- Normalize, deduplicate, and filter low-signal files.
- Preserve language balance (Python, JS/TS, Java, Go, Rust, C++ in phases).
- Keep dataset manifests + hashes for reproducibility.

## Risks and Mitigations
- **Diffusion decoding cost too high:** invest early in step-reduction and distillation.
- **Small model underfitting:** curriculum + stronger data curation + scaling experiments.
- **Misleading training loss:** make training and validation both token-weighted, especially when FIM denoise spans vary in length.
- **A100 memory underuse:** increase per-device microbatch size before increasing model size.
- **Evaluation drift:** frozen benchmark versions and reproducible harness.

## Deliverables
- Reproducible training pipeline. Complete for MVP.
- Baseline nano diffusion checkpoints. Local A100 artifacts exist under `checkpoints/bpe-fim-100m-a100-50k-10k-bg/`.
- Benchmark report with quality/latency tradeoffs. In progress; A100 masked-token perplexity report exists, HumanEval should wait until generated Python is parseable.
- Inference package and docs for local experimentation.
