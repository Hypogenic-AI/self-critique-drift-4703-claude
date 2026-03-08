# Literature Review: Representation Drift Under Self-Reflection

## Research Area Overview

This review covers the intersection of **self-critique/self-reflection in LLMs** and **mechanistic interpretability of internal representations**. The central question is whether self-critique induces structured shifts in internal representations (particularly the residual stream), resulting in more distinct reasoning subspaces with improved linear separability.

The literature spans three key areas:
1. **Self-critique and self-reflection mechanisms** in LLMs (how models evaluate and revise their own outputs)
2. **Representation geometry and probing** (how to measure and characterize internal representations)
3. **Mechanistic interpretability** (understanding what computations transformers perform internally)

---

## Key Papers

### Paper 1: Emergent Manifold Separability during Reasoning in LLMs
- **Authors:** Polo, Chun, Chung (Harvard, NYU, Flatiron Institute)
- **Year:** 2026 (under review)
- **Source:** arXiv 2602.20338
- **Key Contribution:** Discovers that concept manifolds in the residual stream undergo transient geometric restructuring during chain-of-thought reasoning — representations become linearly separable exactly when needed for computation, then compress immediately afterward.
- **Methodology:** Manifold Capacity Theory (MCT) from computational neuroscience — a probe-free, training-free measure of linear separability. Also uses SVM/logistic probes, intrinsic dimensionality (TwoNN, Participation Ratio), and attention-capacity correlation analysis.
- **Model:** Ministral 3 8B Reasoning (32 layers, dim 4096)
- **Dataset:** Compositional Boolean logic task (256 balanced expressions, tree height 5)
- **Results:**
  - Manifold capacity spikes transiently during active computation (pulse-like dynamics)
  - Critical distinction: high probe accuracy ≠ high manifold capacity (retention vs. readiness)
  - Intrinsic dimensionality stays ~10 throughout CoT
  - Middle layers (13-15) initiate geometric restructuring
  - Subspace rotation between computation steps
  - Attention correlates with manifold capacity (r=0.723)
- **Code:** Not publicly available
- **Relevance:** ★★★★★ — Directly provides the methodology (MCT) and baseline expectations for our research. The retention vs. readiness distinction is exactly what we need to differentiate shallow vs. deep self-critique effects.

### Paper 2: From Emergence to Control: Probing and Modulating Self-Reflection in Language Models
- **Authors:** Zhu et al.
- **Year:** 2025
- **Source:** arXiv 2506.12217
- **Key Contribution:** Shows self-reflection is encoded as a separable direction in hidden states (even in pretrained models), and can be enhanced/suppressed via activation steering.
- **Methodology:** Contrastive hidden-state analysis using difference-in-means to extract self-reflection vectors. UMAP visualization across layers. Activation steering (adding/subtracting reflection vectors).
- **Models:** DeepSeek-R1-Distill-Qwen-1.5B/7B, Qwen2.5-1.5B/7B, Llama 3.1 8B Instruct
- **Datasets:** MATH500, AIME 2024, GPQA Diamond
- **Results:**
  - Hidden states clearly separate reflective from non-reflective contexts across all 28 layers
  - Self-reflection vectors are domain-invariant (cosine sim >0.95 across MATH/GPQA)
  - Enhancement: up to +12pp accuracy gains; Suppression: >32-50% token reduction with minimal accuracy loss
  - Reflection latent in pretrained models (0.6% → 18.6% with induced probing)
- **Code:** GitHub: ProbingReflection (referenced but not found publicly)
- **Relevance:** ★★★★★ — Directly demonstrates that self-reflection has a geometric signature in representations. The difference-in-means methodology for extracting "critique vectors" is directly transferable.

### Paper 3: Truth as a Trajectory
- **Authors:** (2026)
- **Source:** arXiv 2603.01326
- **Key Contribution:** Shows that reasoning validity can be detected from layer-wise displacement trajectories in the residual stream, not just static activations.
- **Methodology:** Layer-wise displacement vectors d_{t,l} = h_{t,l+1} - h_{t,l}, unrolled into sequences and classified by LSTM. Also tests kinematic descriptors (velocity, acceleration, curvature).
- **Models:** Llama-3.1-8B, Qwen2.5-14B/32B, Qwen2.5-30B-MoE
- **Datasets:** ARC-Easy/Challenge, OpenBookQA, CommonsenseQA, CosmosQA, BoolQ, TruthfulQA, RealToxicityPrompts
- **Results:**
  - Cross-task generalization: TaT trained on ARC-C transfers to other reasoning benchmarks (79.31% OOD avg vs. 70.49% for linear probes)
  - Displacement-based trajectories generalize better than raw activations
  - Correct reasoning produces smoother trajectories
  - Sequential order matters (Set MLP < LSTM)
- **Code:** Not publicly available
- **Relevance:** ★★★★ — The displacement trajectory approach provides a complementary method to MCT for tracking how self-critique alters the reasoning process (not just static snapshots).

### Paper 4: The Geometry of Truth (Marks & Tegmark)
- **Authors:** Marks, Tegmark (MIT)
- **Year:** 2024 (COLM)
- **Source:** arXiv 2310.06824
- **Key Contribution:** Demonstrates that LLMs develop linear representations of truth/falsehood that generalize across topics.
- **Methodology:** Activation patching for localization, PCA visualization, mass-mean probing (difference-in-means direction), causal interventions via activation shifting.
- **Models:** LLaMA-2-7B/13B/70B
- **Datasets:** Custom true/false statement datasets across diverse topics
- **Results:**
  - Truth/falsehood separates along top PCs in mid layers
  - Mass-mean probing outperforms logistic regression for generalization
  - Causal interventions along truth directions flip model judgments
  - Linear structure emerges with scale
- **Code:** https://github.com/saprmarks/geometry-of-truth
- **Relevance:** ★★★★★ — Provides the foundational probing methodology. The mass-mean probe and PCA approach for tracking truth directions pre/post-critique is directly applicable.

### Paper 5: Representation Engineering (Zou et al.)
- **Authors:** Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika, Dombrowski, Goel, Li, Glass, Hazan, Klivans, Hendrycks
- **Year:** 2023
- **Source:** arXiv 2310.01405
- **Key Contribution:** Top-down approach to reading and controlling LLM representations using contrastive stimuli and PCA.
- **Methodology:** Linear Artificial Tomography (LAT) — design contrastive stimuli, extract hidden states, apply PCA to differences. Three control methods: reading vectors, contrast vectors, LoRRA.
- **Models:** LLaMA-2-Chat (7B/13B/70B), Vicuna-33B
- **Datasets:** TruthfulQA, RACE, CSQA, OBQA, ARC
- **Results:**
  - +18.1pp on TruthfulQA MC1 over zero-shot
  - Works with as few as 5-10 stimulus examples
  - Extracts honesty, truthfulness, hallucination, emotion, morality directions
  - Control vectors can steer model behavior causally
- **Code:** https://github.com/andyzoujm/representation-engineering
- **Relevance:** ★★★★★ — Provides the toolkit for extracting "self-critique directions" via contrastive stimuli. The reading/control vector approach enables both measurement and causal intervention.

### Paper 6: Self-RAG
- **Authors:** Asai et al.
- **Year:** 2023
- **Source:** arXiv 2310.11511
- **Key Contribution:** Trains LMs to self-critique via reflection tokens (Retrieve, IsRel, IsSup, IsUse) integrated into generation.
- **Models:** Llama 2-7B/13B
- **Datasets:** PubHealth, ARC-Challenge, PopQA, TriviaQA
- **Code:** https://selfrag.github.io/
- **Relevance:** ★★★ — Provides a self-critique framework where critique is internalized as reflection tokens. Good candidate for studying how critique tokens alter internal states.

### Paper 7: Reflexion (Shinn et al.)
- **Authors:** Shinn, Cassano, Gopinath, Narasimhan, Yao
- **Year:** 2023 (NeurIPS)
- **Source:** arXiv 2303.11366
- **Key Contribution:** Framework for LLM agents to learn from verbal self-reflection without weight updates.
- **Models:** GPT-4, GPT-3.5
- **Datasets:** AlfWorld, HotpotQA, HumanEval, MBPP
- **Code:** https://github.com/noahshinn024/reflexion
- **Relevance:** ★★★ — Provides an in-context self-critique protocol. Since learning happens purely through context, it's ideal for studying activation-level (not weight-level) representation drift.

### Paper 8: Let's Verify Step by Step (Lightman et al.)
- **Authors:** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe
- **Year:** 2023
- **Source:** arXiv 2305.20050
- **Key Contribution:** Shows process supervision (per-step verification) outperforms outcome supervision for reasoning.
- **Models:** GPT-4 variants
- **Datasets:** MATH (PRM800K released: 800K step-level labels)
- **Code:** PRM800K dataset publicly available
- **Relevance:** ★★★ — Process supervision framing supports analyzing representation drift at each reasoning step, not just final outputs.

### Paper 9: Chain-of-Thought Prompting (Wei et al.)
- **Authors:** Wei, Wang, Schuurmans, Bosma, Chi, Le, Zhou
- **Year:** 2022
- **Source:** arXiv 2205.10625
- **Relevance:** ★★★ — Foundational CoT paper. Self-critique often occurs after CoT reasoning; understanding CoT's representation effects is prerequisite.

### Paper 10: Language Models Don't Always Say What They Think (Turpin et al.)
- **Authors:** Turpin, Michael, Perez, Bowman
- **Year:** 2024
- **Source:** arXiv 2305.01610
- **Relevance:** ★★★★ — Shows CoT explanations can be unfaithful to models' actual reasoning. Directly motivates studying whether self-critique produces genuine representation change or shallow surface modification.

---

## Common Methodologies

### Probing Techniques
- **Linear probes (LR, SVM):** Train linear classifiers on frozen activations. Used in most papers. Risk: can overfit in high-dimensional spaces, giving false positives for linear separability.
- **Mass-mean probing (difference-in-means):** Marks & Tegmark (2024), Zhu et al. (2025). More robust than LR for recovering true feature directions.
- **Manifold Capacity Theory (MCT):** Polo et al. (2026). Probe-free geometric measure. Avoids training confounds entirely. Preferred for our research.

### Representation Extraction
- **Residual stream activations:** Standard across all papers. Extract h^(l) at specific token positions and layers.
- **Layer-wise displacements:** d_{t,l} = h_{t,l+1} - h_{t,l}. Isolates active computation from static content (TaT, 2026).
- **Contrastive stimuli:** Design paired inputs that differ only in the concept of interest, extract difference in activations (RepE, Geometry of Truth).

### Visualization and Analysis
- **PCA:** Primary dimensionality reduction for visualization (RepE, Geometry of Truth).
- **UMAP:** Used for higher-dimensional structure visualization (Zhu et al.).
- **Intrinsic dimensionality:** TwoNN, Participation Ratio (Polo et al.).
- **Cosine similarity of hyperplane normals:** Track whether separation directions are conserved across conditions.

### Causal Interventions
- **Activation patching:** Swap activations between conditions to identify causally relevant positions (Geometry of Truth).
- **Activation steering/addition:** Add/subtract concept vectors to control behavior (RepE, Zhu et al.).
- **Normalized Indirect Effect (NIE):** Quantify causal impact of interventions.

---

## Standard Baselines

| Baseline | Description | Used In |
|----------|-------------|---------|
| Linear probe (LR/SVM) | Train classifier on frozen activations | All probing papers |
| Random baseline | Permuted labels or random projections | MCT, probing papers |
| Zero-shot | Model without any intervention | RepE, Self-RAG |
| Few-shot ICL | In-context learning baseline | RepE, Reflexion |
| CCS (Burns et al.) | Unsupervised truth direction finding | Geometry of Truth, RepE |

---

## Evaluation Metrics

| Metric | Purpose | Papers |
|--------|---------|--------|
| Manifold capacity (α) | Probe-free linear separability | Polo et al. |
| Linear probe accuracy | Classifier-based separability | All probing papers |
| Cosine similarity | Direction conservation across conditions | Geometry of Truth, Zhu et al. |
| Intrinsic dimensionality | Representation complexity | Polo et al. |
| CKA (Centered Kernel Alignment) | Representation similarity across layers/conditions | Standard metric |
| NIE (Normalized Indirect Effect) | Causal intervention strength | Geometry of Truth |
| Task accuracy | Downstream performance change | All papers |

---

## Gaps and Opportunities

1. **No paper directly studies representation drift during self-critique.** The manifold separability paper studies CoT reasoning dynamics; the self-reflection probing paper identifies reflection directions but doesn't track temporal evolution through a critique cycle. Our research fills this gap.

2. **MCT has not been applied to self-critique settings.** The manifold capacity methodology from Polo et al. is ideal for measuring whether self-critique creates genuine geometric restructuring vs. superficial changes.

3. **The retention vs. readiness distinction has not been tested for self-critique.** Polo et al. show that probe accuracy and manifold capacity can diverge. This distinction could reveal whether self-critique genuinely restructures representations for computation or merely retains information passively.

4. **Layer-wise displacement trajectories during self-critique are unstudied.** The TaT methodology could reveal whether effective self-critique smooths trajectories (like correct reasoning) or introduces new deviations.

5. **Cross-round representation tracking is missing.** No paper tracks how the same concept's representation evolves across multiple rounds of self-critique.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **GSM8K / MATH-500** — Math reasoning with clear correct/incorrect labels. Step-by-step solutions enable process-level analysis.
2. **TruthfulQA** — Factual truth/falsehood with established truth direction baselines.
3. **ARC-Challenge** — Reasoning with multiple choice (clear signal for separability).
4. **Custom Boolean logic tasks** — Following Polo et al.'s approach for controlled compositional reasoning.

### Recommended Baselines
1. **Pre-critique representations** — Internal states before self-critique (control condition)
2. **Random perturbation** — Add random noise of equal magnitude to self-critique shift
3. **Paraphrase control** — Model rephrases without critiquing (surface change without semantic change)
4. **Linear probe accuracy** — Compare with MCT to test retention vs. readiness

### Recommended Metrics
1. **Manifold Capacity (MCT)** — Primary metric for linear separability (probe-free)
2. **Linear probe accuracy (SVM, LR)** — Secondary metric, compare with MCT
3. **Cosine similarity of separation directions** — Track direction conservation pre/post-critique
4. **Intrinsic dimensionality (TwoNN, PR)** — Measure representation complexity changes
5. **Layer-wise displacement magnitude** — Track where self-critique has strongest effect
6. **CKA between pre/post-critique representations** — Overall representation similarity

### Recommended Models
1. **Small open models** (1.5B-7B) for detailed mechanistic analysis — Qwen2.5, DeepSeek-R1-Distill
2. **Llama-3.1-8B** — Well-studied baseline with existing probing results
3. **Reasoning-specialized models** vs. general models — Compare whether reasoning training affects self-critique dynamics

### Methodological Considerations
- **Textual contamination:** During self-critique, the model has already generated an answer. Use structural token anchoring (Polo et al.) to avoid trivially decoding the model's own output.
- **Multiple self-critique rounds:** Track representation drift across 2-3 rounds to see if changes compound or converge.
- **Causal validation:** Use activation steering (RepE) to verify that identified self-critique directions are functionally meaningful.
- **Scale dependence:** Test across model sizes to understand whether representation drift scales with capability.
