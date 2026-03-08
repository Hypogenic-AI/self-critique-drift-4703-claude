# Code Repositories for "Representation Drift Under Self-Reflection"

This directory contains cloned reference repositories relevant to the research project:
**"Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?"**

All repositories were cloned with `--depth 1` (shallow clone) to save space.

---

## 1. Geometry of Truth

- **URL:** https://github.com/saprmarks/geometry-of-truth
- **Location:** `code/geometry-of-truth/`
- **Paper:** [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) (Marks & Tegmark)
- **Purpose:** Demonstrates that LLM internal representations of true/false statements exhibit linear geometric structure. Provides probing tools for extracting "truth directions" from model activations.
- **Key Files:**
  - `generate_acts.py` — Generate and cache model activations per layer for datasets
  - `probes.py` — Probe class definitions (linear probes for truth direction)
  - `generalization.ipynb` — Train probes on one dataset, test generalization to another
  - `interventions.py` — Causal intervention experiments on truth representations
  - `dataexplorer.ipynb` — Visualization of dataset representations
  - `utils.py` — Dataset management utilities
  - `datasets/` — True/false statement datasets (cities, negations, etc.)
- **Dependencies:** `torch`, `transformers`, `sentencepiece`, `pandas`, `plotly`, `tqdm`
- **Relevance to our research:** Core methodology for probing internal truth representations before and after self-critique. We can adapt their probe training pipeline to measure whether self-reflection shifts truth directions in activation space (representation drift).

---

## 2. Representation Engineering (RepE)

- **URL:** https://github.com/andyzoujm/representation-engineering
- **Location:** `code/representation-engineering/`
- **Paper:** [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) (Zou et al.)
- **Purpose:** A top-down framework for reading and controlling high-level cognitive properties (truthfulness, fairness, harmlessness) via population-level representations in LLMs.
- **Key Files:**
  - `repe/rep_reading_pipeline.py` — RepReading pipeline: extract representation directions for concepts
  - `repe/rep_control_pipeline.py` — RepControl pipeline: steer model behavior via activation interventions
  - `repe/rep_readers.py` — Core RepReader classes for computing concept directions
  - `repe/rep_control_reading_vec.py` — Reading vector computation for control
  - `examples/honesty/` — Honesty/truthfulness experiments
  - `examples/primary_emotions/` — Emotion representation experiments
  - `examples/memorization/` — Memorization detection via representations
- **Dependencies:** `accelerate`, `scikit-learn`, `transformers` (Python >= 3.9)
- **Relevance to our research:** Provides the RepReading framework for extracting concept-level representation vectors. We can use RepE to measure how self-critique changes representation directions for truthfulness, confidence, and other safety-relevant properties across layers.

---

## 3. TransformerLens

- **URL:** https://github.com/TransformerLensOrg/TransformerLens
- **Location:** `code/TransformerLens/`
- **Paper/Docs:** [TransformerLens Documentation](https://TransformerLensOrg.github.io/TransformerLens/)
- **Purpose:** A mechanistic interpretability library for GPT-2 style language models. Provides hook-based access to all internal activations (residual stream, attention patterns, MLP outputs) with caching and intervention capabilities.
- **Key Files:**
  - `transformer_lens/HookedTransformer.py` — Main model class with activation hooks
  - `transformer_lens/hook_points.py` — Hook point infrastructure for caching/editing activations
  - `transformer_lens/ActivationCache.py` — Cache object for stored activations
  - `Main_Demo.ipynb` — Comprehensive demo of library features
  - `demos/` — Additional example notebooks
- **Dependencies:** `torch`, `transformers`, `einops`, `fancy-einsum`, `jaxtyping`, `accelerate`, `datasets`, `wandb`, `pandas`, `tqdm`, `rich`
- **Install:** `pip install transformer_lens`
- **Relevance to our research:** Primary toolkit for extracting internal activations during self-critique sequences. HookedTransformer lets us cache residual stream activations at every layer before and after self-reflection, enabling direct measurement of representation drift. Also supports activation patching for causal analysis of which components drive drift.

---

## 4. Reflexion

- **URL:** https://github.com/noahshinn024/reflexion
- **Location:** `code/reflexion/`
- **Paper:** [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., NeurIPS 2023)
- **Purpose:** A framework where LLM agents reflect on their mistakes using verbal self-feedback and use that reflection to improve on subsequent attempts. Demonstrates self-reflection as verbal reinforcement learning.
- **Key Files:**
  - `hotpotqa_runs/` — Reasoning experiments on HotPotQA with ReAct and CoT agents
  - `alfworld_runs/` — Decision-making experiments in AlfWorld environments
  - `programming_runs/reflexion.py` — Core reflexion loop for programming tasks
  - `programming_runs/generators/` — LLM-based code generation with reflection
  - `programming_runs/executors/` — Code execution and test validation
- **Dependencies:** `openai`, `langchain`, `transformers`, `tiktoken`, `gym`, `pandas`, `scikit-learn`
- **Relevance to our research:** Provides the self-reflection protocol and agent architecture. We can adapt their reflexion loop to generate self-critique sequences, then use TransformerLens to probe how internal representations change between the initial response and the post-reflection response.

---

## 5. Self-Reflection in LLM Agents

- **URL:** https://github.com/matthewrenze/self-reflection
- **Location:** `code/self-reflection/`
- **Paper:** [Self-Reflection in LLM Agents: Effects on Problem-Solving Performance](https://arxiv.org/abs/2405.06682) (Renze)
- **Purpose:** Systematic study of self-reflection effects on LLM problem-solving performance across nine LLMs and eight reflection types. Includes comprehensive experimental data and analysis scripts.
- **Key Files:**
  - `source/1_solve_with_baseline.py` — Baseline problem-solving (no reflection)
  - `source/2_reflect_on_solutions.py` — Self-reflection on incorrect answers
  - `source/4_solve_with_reflections.py` — Re-answering with reflection guidance
  - `source/8_analyze_details.py` — Statistical analysis (McNemar test)
  - `source/agents/` — Agent type definitions
  - `data/reflections/` — Generated self-reflection text data
  - `data/dialogs/` — Full dialog transcripts for analysis
- **Dependencies:** Python, OpenAI API
- **Relevance to our research:** Provides structured self-reflection data across multiple LLMs and reflection types. The reflection text and dialog data can serve as input stimuli for probing experiments. Their taxonomy of reflection types (explanation, advice, keywords, etc.) helps us systematically test which forms of self-critique cause the most representation drift.

---

## Note on ProbingReflection Repository

A dedicated "ProbingReflection" repository was not found on GitHub. Web searches for repos specifically combining probing with self-reflection in LLMs did not surface a repo by that exact name. The matthewrenze/self-reflection repo was cloned as the closest substitute, providing structured self-reflection experiments and data. The combination of Geometry of Truth (probing) + Reflexion (self-reflection protocol) + TransformerLens (activation access) covers the functionality such a repo would provide.

---

## How These Repos Work Together for Our Research

```
                    Self-Critique Loop
                    ┌─────────────────┐
                    │   Reflexion /   │
                    │ Self-Reflection │  (Generate self-critique sequences)
                    └────────┬────────┘
                             │
                     Prompt sequences
                             │
                    ┌────────▼────────┐
                    │ TransformerLens │  (Extract activations at each layer,
                    │  HookedTransf.  │   before and after self-critique)
                    └────────┬────────┘
                             │
                    Cached activations
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────────┐ ┌───▼──────────────┐
     │ Geometry of    │ │ RepE       │ │ Direct cosine    │
     │ Truth probes   │ │ RepReading │ │ similarity /PCA  │
     └────────────────┘ └────────────┘ └──────────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                   Measure representation drift
                   (direction shifts, probe accuracy
                    changes, geometric restructuring)
```

**Experimental pipeline:**
1. Use Reflexion/Self-Reflection protocols to generate self-critique sequences
2. Feed sequences through TransformerLens to cache all internal activations
3. Apply Geometry-of-Truth probes to measure truth direction shifts
4. Use RepE RepReading to track concept-level representation changes
5. Quantify representation drift across layers and critique iterations
