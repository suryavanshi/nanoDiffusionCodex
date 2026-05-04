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
   - Start with text-level tokenization (BPE/SentencePiece) for simplicity and reproducibility.
   - Add optional AST-aware auxiliary objectives later.

2. **Diffusion Objective for Discrete Tokens**
   - Implement a discrete diffusion/noising process over token sequences.
   - Compare a few schedules (linear/cosine/custom) and timestep parameterizations.
   - Train with denoising objective conditioned on prompt/context.

3. **Backbone Architecture**
   - Decoder-style Transformer with diffusion timestep conditioning.
   - Parameter-efficient baseline with modern norms/activations and rotary embeddings.
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

### Phase 1 — MVP Diffusion Training Loop (Weeks 2–4)
- Implement noising + denoising training objective.
- Implement minimal Transformer denoiser with timestep embeddings.
- Train on small curated code corpus subset.
- Validate end-to-end train/eval/infer loop.

### Phase 2 — Quality + Stability (Weeks 5–8)
- Add improved schedules, loss weighting, and curriculum.
- Expand dataset quality filters and dedup.
- Add benchmark harness (HumanEval-like, MBPP-like).
- Introduce checkpoint averaging and robust logging.

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
- **Evaluation drift:** frozen benchmark versions and reproducible harness.

## Deliverables
- Reproducible training pipeline.
- Baseline nano diffusion checkpoints.
- Benchmark report with quality/latency tradeoffs.
- Inference package and docs for local experimentation.
