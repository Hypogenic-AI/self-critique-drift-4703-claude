# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?**

Resources include 19 papers, 4 datasets, and 5 code repositories covering self-critique mechanisms, representation geometry, mechanistic interpretability, and probing methodologies.

---

## Papers

Total papers downloaded: **19**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Emergent Manifold Separability during Reasoning in LLMs | Polo, Chun, Chung | 2026 | 2602.20338.pdf | MCT methodology for probe-free separability |
| 2 | From Emergence to Control: Probing Self-Reflection | Zhu et al. | 2025 | 2506.12217.pdf | Self-reflection vectors, activation steering |
| 3 | Truth as a Trajectory | - | 2026 | 2603.01326.pdf | Layer-wise displacement trajectories |
| 4 | The Geometry of Truth | Marks, Tegmark | 2024 | 2310.06824.pdf | Truth directions, mass-mean probing |
| 5 | Representation Engineering | Zou et al. | 2023 | 2310.01405.pdf | LAT, reading/control vectors |
| 6 | Self-RAG | Asai et al. | 2023 | 2310.11511.pdf | Reflection tokens for self-critique |
| 7 | Reflexion | Shinn et al. | 2023 | 2303.11366.pdf | Verbal self-reflection framework |
| 8 | Let's Verify Step by Step | Lightman et al. | 2023 | 2305.20050.pdf | Process supervision, PRM800K |
| 9 | Chain-of-Thought Prompting | Wei et al. | 2022 | 2205.10625.pdf | Foundational CoT |
| 10 | LMs Don't Always Say What They Think | Turpin et al. | 2024 | 2305.01610.pdf | Unfaithful CoT reasoning |
| 11 | The Reversal Curse | Berglund et al. | 2024 | 2309.12288.pdf | Asymmetric representations |
| 12 | Probing for Arithmetic Errors | - | 2025 | 2507.12379.pdf | Error probing in LMs |
| 13 | STREAM | - | 2025 | 2510.19875.pdf | Mechanistic interp scaling |
| 14 | Superscopes | - | 2025 | 2503.02078.pdf | Feature amplification |
| 15 | Can LLMs Predict Failures | - | 2025 | 2512.20578.pdf | Self-awareness circuits |
| 16 | Recursive Concept Evolution | - | 2026 | 2602.15725.pdf | Compositional reasoning |
| 17 | Representation Geometry | - | 2024 | 2404.14082.pdf | Geometry analysis |
| 18 | Linear Representations | - | 2023 | 2310.15916.pdf | Linear structure in LLMs |
| 19 | ReDeEP | - | 2024 | 2401.10474.pdf | Mechanistic hallucination detection |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets

Total datasets downloaded: **4**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | openai/gsm8k | 7,473 train + 1,319 test | Math reasoning | datasets/gsm8k/ | Step-by-step solutions |
| MATH-500 | HuggingFaceH4/MATH-500 | 500 test | Competition math | datasets/math/ | Multi-subject, leveled |
| TruthfulQA | truthful_qa | 817 validation | Truth/factual QA | datasets/truthfulqa/ | Multiple choice format |
| ARC-Challenge | allenai/ai2_arc | 1,119 train + 1,172 test | Science reasoning | datasets/arc_challenge/ | Multiple choice |

See [datasets/README.md](datasets/README.md) for download instructions and sample data.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth direction probing | code/geometry-of-truth/ | Mass-mean probes, PCA, interventions |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE toolkit | code/representation-engineering/ | LAT, reading/control vectors |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interp | code/TransformerLens/ | Hook-based activation caching |
| reflexion | github.com/noahshinn024/reflexion | Self-reflection agents | code/reflexion/ | Verbal self-critique protocol |
| self-reflection | github.com/matthewrenze/self-reflection | Self-reflection study | code/self-reflection/ | 9 LLMs, 8 reflection types |

See [code/README.md](code/README.md) for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Paper-finder service** with diligent mode for initial discovery (176 papers from semantic scholar + arxiv)
2. **arXiv API** programmatic search across 7 query combinations (106 unique papers)
3. **Semantic Scholar API** across 5 queries (99 unique papers)
4. **Relevance scoring** using keyword matching and citation counts to rank ~380 papers
5. **Top 19 papers** downloaded based on composite relevance scores

### Selection Criteria

Papers were selected based on:
- Direct relevance to self-critique/self-reflection mechanisms in LLMs
- Methodology applicable to measuring representation drift (probing, MCT, PCA)
- Mechanistic interpretability of reasoning processes
- Established baselines and benchmarks in the field
- Code availability for reproducibility

### Challenges Encountered

- The "ProbingReflection" repository referenced in Zhu et al. (2025) was not found on GitHub
- The `lighteval/MATH` dataset was no longer available; used `HuggingFaceH4/MATH-500` instead
- Some paper-finder results timed out; supplemented with direct arXiv API searches
- arXiv ID 2309.12288 was "The Reversal Curse" (not a linear representations paper as initially expected)

### Gaps and Workarounds

- **No direct precedent for self-critique representation drift study** — this is the primary research gap we address
- **No dedicated self-critique activation dataset** — will need to generate activations from models during self-critique using TransformerLens
- **MCT implementation not publicly available** — will need to implement from Polo et al.'s paper description (based on Chung et al., 2018 neuroscience methodology)

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **GSM8K** for math reasoning with self-critique (clear correct/incorrect, step-by-step)
- **TruthfulQA** for truth direction analysis (established baselines from Geometry of Truth)
- **Custom Boolean logic** following Polo et al. for controlled compositional reasoning

### 2. Baseline Methods
- Pre-critique representations (no self-reflection)
- Random perturbation (noise of equal magnitude)
- Paraphrase control (surface rewriting without critique)
- Standard CoT without self-critique step

### 3. Evaluation Metrics
- **Manifold Capacity (MCT)** — primary, probe-free separability measure
- **Linear probe accuracy** (SVM, logistic regression) — secondary, compare with MCT
- **Cosine similarity of separation directions** — direction conservation
- **Intrinsic dimensionality** (TwoNN, Participation Ratio) — complexity changes
- **CKA** — overall representation similarity pre/post-critique
- **Layer-wise displacement magnitude** — where critique has strongest effect

### 4. Code to Adapt/Reuse
- **TransformerLens** — activation extraction and hooking (primary tool)
- **geometry-of-truth** — probing methodology, PCA visualization
- **representation-engineering** — contrastive stimulus design, reading vectors
- **reflexion** — self-critique protocol design

### 5. Experimental Pipeline
```
1. Select model (e.g., Llama-3.1-8B or Qwen2.5-7B)
2. Generate initial responses on GSM8K/TruthfulQA/Boolean logic
3. Extract residual stream activations at all layers (via TransformerLens)
4. Prompt model for self-critique of its own response
5. Extract post-critique residual stream activations
6. Compute MCT, probe accuracy, cosine similarity, CKA between pre/post
7. Analyze layer-wise and token-wise patterns
8. Repeat for 2-3 critique rounds to study convergence
9. Validate with activation steering (inject/suppress critique vectors)
```
