# nanoDiffusionCodex

A compact, code-first diffusion project scaffold for training, evaluation, and inference experiments.

nanoDiffusionCodex is designed as a clean starting point: lightweight structure, explicit plans, and simple script entrypoints so you can iterate quickly.

## ✨ What’s in this repository

- **Project roadmap** with milestones and implementation notes.
- **Package scaffolding** for core diffusion components.
- **CLI-style script entrypoints** for train / eval / infer workflows.
- **Architecture notes** to guide future extension.
- **Smoke-test skeleton** to keep basic integrity checks in place.

## 📌 Current status

This repository is currently in an early scaffold phase. The structure is in place, while most implementation modules are intentionally minimal placeholders.

If you are extending this repo, start with:

1. `plan.md` to understand priorities,
2. `docs/architecture.md` for module boundaries,
3. `scripts/train.py`, `scripts/eval.py`, and `scripts/infer.py` as integration entrypoints.

## 🚀 Quick start

From the repository root:

```bash
python scripts/train.py
python scripts/eval.py
python scripts/infer.py
```

## 🗂️ Repository layout

```text
.
├── README.md
├── plan.md
├── docs/
│   └── architecture.md
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
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

- Implement the first end-to-end minimal training loop.
- Define a baseline dataset interface under `src/nano_diffusion/data/`.
- Add configuration loading/validation from `src/nano_diffusion/configs/default.yaml`.
- Expand smoke tests into module-level unit tests.

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
