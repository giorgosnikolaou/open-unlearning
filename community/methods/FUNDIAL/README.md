# FUNDIAL: Focused UNDIAL (NAACL 2025)

A variant of UNDIAL (Section 3.2 of the original paper) that restricts the
self-distillation forget loss to **noun chunks** or **named entities**
identified by spaCy. The intuition: function words carry little information,
so concentrating the unlearning signal on content tokens improves the
unlearning/utility trade-off.

- Authors: Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramón Huerta, Ivan Vulić
- Paper: https://arxiv.org/pdf/2402.10052

# Setup

Install spaCy and download the English model once:

```bash
pip install "spacy>=3.7,<4.0"
python -m spacy download en_core_web_sm
```

# Hyperparameters

Inherits the UNDIAL search ranges:
- `learning_rate` ∈ {1e-5, 1e-4, 3e-4}
- `beta` (logit penalty) ∈ {3, 10, 30}
- `alpha` (retain weight) ∈ {0, 1, 2, 5}

FUNDIAL-specific:
- `mask_type`: `noun` (uses `doc.noun_chunks`) or `entity` (uses `doc.ents`)
- `empty_mask_fallback`: `zero` (sample contributes 0 forget loss when no
  noun/entity is detected — paper-faithful) or `uniform` (fall back to standard
  UNDIAL on those samples)

# Caveats

- `en_core_web_sm` only handles English. For other languages, swap
  `spacy_model` and download the corresponding model.
- spaCy runs once at the start of training to build per-sample masks; runtime
  is dominated by NER/parsing on the forget split (seconds, not minutes).

# Run

```bash
bash run.sh
```

# Citation

```bibtex
@misc{dong2024undial,
      title={UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models},
      author={Yijiang River Dong and Hongzhou Lin and Mikhail Belkin and Ramon Huerta and Ivan Vulić},
      year={2024},
      eprint={2402.10052},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.10052},
}
```
