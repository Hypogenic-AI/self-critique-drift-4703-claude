# Representation Drift Under Self-Reflection

Does self-critique reshape a language model's internal representations, or is it just shallow re-sampling? This project measures representation drift in Pythia-2.8B's residual stream during self-critique, comparing it against a paraphrase control condition.

## Key Findings

- **Self-critique induces 41% larger representational displacement** than paraphrasing (L2: 50.7 vs 35.8), statistically significant in 29/33 layers (Bonferroni-corrected)
- **Maximum drift occurs in middle layers (12-16)**, consistent with prior work on reasoning-related geometric restructuring
- **Critique drift is more directionally consistent** across samples than paraphrase drift (0.596 vs 0.479 cosine consistency), suggesting a systematic "critique subspace"
- **CKA reveals genuine representational reorganization**: Base-Critique CKA (0.224) is roughly half of Base-Paraphrase CKA (0.428)
- **Linear probes achieve 97.3% accuracy** on critique activations for correct/incorrect classification (vs 58.7% baseline), though partly confounded by explicit correctness information in critique text

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers transformer-lens numpy scipy scikit-learn matplotlib seaborn openai datasets

# Run experiments (requires OPENAI_API_KEY for stimulus generation)
export USER=researcher  # needed for PyTorch cache
python src/generate_stimuli.py    # ~15 min, generates critique/paraphrase text via GPT-4.1
python src/extract_activations.py # ~10 min, extracts Pythia-2.8B activations (needs GPU)
python src/analysis.py            # ~5 min, runs all analyses and generates plots
```

## File Structure

```
REPORT.md              # Full research report with results
planning.md            # Research plan and methodology
src/
  generate_stimuli.py    # GPT-4.1 stimulus generation
  extract_activations.py # TransformerLens activation extraction
  analysis.py            # Analysis pipeline (7 metrics + plots)
results/
  data/                  # Raw activations (.npy) and stimuli (.json)
  plots/                 # 7 visualization figures
  analysis_results.json  # All numerical results
datasets/gsm8k/          # GSM8K dataset
papers/                  # 19 related papers
code/                    # Reference implementations
literature_review.md     # Comprehensive literature review
```

## Hardware
- 2x NVIDIA RTX 3090 (24GB)
- Python 3.12.8, PyTorch 2.10.0, TransformerLens 2.15.4

See [REPORT.md](REPORT.md) for full methodology, results, and discussion.
