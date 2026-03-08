# Research Plan: Representation Drift Under Self-Reflection

## Motivation & Novelty Assessment

### Why This Research Matters
Self-critique and self-reflection are increasingly used in LLM prompting (chain-of-thought, Reflexion, self-consistency), but it is unknown whether these techniques produce genuine internal representational change or merely re-sample outputs. Understanding this has direct implications for AI safety (can models truly "reconsider"?) and for building recursive self-improvement agents.

### Gap in Existing Work
From the literature review: Polo et al. (2026) study manifold separability during CoT reasoning but not during self-critique. Zhu et al. (2025) find self-reflection vectors in hidden states but don't track temporal evolution through a critique cycle. No paper directly measures representation drift before vs. after self-critique.

### Our Novel Contribution
We are the first to systematically measure how self-critique text alters internal representations in the residual stream, using multiple complementary metrics (cosine drift, linear probes, CKA, PCA) across layers. We distinguish between structured representational change and mere surface variation using controlled experiments with critique vs. paraphrase conditions.

### Experiment Justification
- **Experiment 1 (Representation Drift Magnitude)**: Measures whether critique induces larger/different activation shifts than length-matched control text. Tests the basic claim.
- **Experiment 2 (Linear Separability)**: Tests whether post-critique representations better separate correct from incorrect reasoning. Connects representation change to functional improvement.
- **Experiment 3 (Layer-wise Analysis)**: Identifies which layers are most affected by self-critique, revealing computational depth of the critique process.

## Research Question
Does processing self-critique text induce structured shifts in a language model's residual stream representations, and do these shifts differ systematically from surface-level text variation (paraphrase control)?

## Hypothesis Decomposition
1. **H1 (Drift Magnitude)**: Self-critique induces larger representation drift (cosine distance) in middle-to-late layers than a length-matched paraphrase control.
2. **H2 (Structured Change)**: Post-critique representations show improved linear separability for correct vs. incorrect answers compared to pre-critique representations.
3. **H3 (Layer Specificity)**: The critique-induced drift is concentrated in specific layers (predicted: middle layers 40-70% depth), not uniform across all layers.
4. **H4 (Direction Consistency)**: The direction of critique-induced drift is consistent across questions (low variance in drift direction), suggesting a systematic "critique subspace."

## Proposed Methodology

### Approach
Use TransformerLens to extract residual stream activations from Pythia-2.8B while it processes GSM8K math questions under three conditions: base (question + answer), critique (question + answer + critique + revision), and control (question + answer + paraphrase). Compare activations across conditions using cosine similarity, linear probes, CKA, and PCA.

### Why This Model?
Pythia-2.8B is well-supported by TransformerLens, fits easily on a single RTX 3090, and has enough capacity to process mathematical reasoning text. Using a base (non-instruction-tuned) model tests whether critique-structure processing is an emergent capability, not just instruction following.

### Experimental Steps
1. **Stimulus Generation**: Use GPT-4.1 API to generate matched critique and paraphrase text for 150 GSM8K questions (100 train, 50 test for probes).
2. **Activation Extraction**: Run all three conditions through Pythia-2.8B, extracting residual stream activations at the last token of each prompt, at all 32 layers.
3. **Drift Analysis**: Compute per-layer cosine similarity between base and critique conditions, and base and control conditions.
4. **Probe Training**: Train linear probes (logistic regression) on activations to classify correct vs. incorrect answers, separately for pre- and post-critique conditions.
5. **CKA Analysis**: Compute CKA between base and critique activations per layer.
6. **PCA Visualization**: Visualize activation space in 2D for selected layers.
7. **Statistical Testing**: Paired t-tests or Wilcoxon signed-rank tests comparing drift magnitudes across conditions.

### Baselines
1. **Pre-critique representations** (base condition)
2. **Paraphrase control** (surface text change without critique content)
3. **Random noise** (add Gaussian noise of same magnitude as critique drift)

### Evaluation Metrics
- Cosine similarity between conditions (per layer)
- Linear probe accuracy (correct/incorrect classification)
- CKA (Centered Kernel Alignment) between conditions
- PCA explained variance and cluster separation
- Displacement magnitude (L2 norm of activation difference)

### Statistical Analysis Plan
- Paired Wilcoxon signed-rank tests for drift magnitude comparison (critique vs. control)
- Bootstrap confidence intervals for probe accuracy differences
- Significance level: α = 0.05 with Bonferroni correction for multiple layers

## Expected Outcomes
- **If self-critique is meaningful**: Larger, more structured drift in mid-to-late layers for critique vs. control. Improved linear separability post-critique. Consistent drift directions.
- **If self-critique is shallow**: Similar drift magnitude for critique and control. No improvement in separability. Random drift directions.

## Timeline and Milestones
1. Environment setup + stimulus generation: 20 min
2. Activation extraction: 30 min
3. Analysis + visualization: 30 min
4. Documentation: 20 min

## Potential Challenges
- Pythia-2.8B may not show strong self-critique effects (base model). Mitigation: focus on relative differences between conditions.
- Sequence length variation between conditions. Mitigation: use last-token activations and control for length.
- GPU memory for activation caching. Mitigation: process one example at a time, aggregate statistics.

## Success Criteria
1. Statistically significant difference in drift magnitude between critique and control conditions (p < 0.05)
2. Clear layer-specific patterns in drift magnitude
3. At least 3 of 4 hypotheses supported or clearly refuted with evidence
