# Architecture (Initial)

- `src/nano_diffusion/configs/`: experiment and runtime configs.
- `src/nano_diffusion/data/`: dataset ingestion and preprocessing.
- `src/nano_diffusion/models/`: Transformer denoiser and related modules.
- `src/nano_diffusion/diffusion/`: noising schedules and denoising objectives.
- `src/nano_diffusion/training/`: training loop, optimization, checkpointing.
- `src/nano_diffusion/inference/`: samplers and serving adapters.
- `src/nano_diffusion/evaluation/`: benchmarks and metrics.
- `src/nano_diffusion/utils/`: shared infrastructure utilities.
- `scripts/`: thin CLI entrypoints for train/eval/infer.
