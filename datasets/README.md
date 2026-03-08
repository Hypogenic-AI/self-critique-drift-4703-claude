# Datasets for Representation Drift Under Self-Reflection

This directory contains datasets used in the research project "Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?"

## Overview

| Dataset | Source | Size | Task Type | Key Use in Research |
|---------|--------|------|-----------|-------------------|
| GSM8K | `openai/gsm8k` | 7,473 train / 1,319 test | Grade-school math word problems | Step-by-step reasoning with clear correct answers; ideal for self-critique loops |
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 test | Competition mathematics | Harder math requiring multi-step reasoning; tests whether self-critique improves or degrades representations on difficult problems |
| TruthfulQA | `truthful_qa` (multiple_choice) | 817 validation | Factual truth assessment | Truth/falsehood probing; critical for analyzing whether self-critique moves representations along a "truth direction" |
| ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | 1,119 train / 1,172 test / 299 val | Science reasoning (multiple choice) | General reasoning with clear labels; good for linear separability analysis of correct vs. incorrect answers |

## Why These Datasets

### GSM8K (Math Reasoning)
- **Self-critique relevance**: Models frequently make arithmetic or logical errors that can be caught during self-critique. Comparing hidden states before and after self-critique reveals whether the model's internal representation of the problem changes.
- **Labels**: Each problem has a unique numerical answer (extracted after `####`), enabling binary correct/incorrect classification.
- **Fields**: `question`, `answer` (includes chain-of-thought reasoning and final answer after `####`)

### MATH-500 (Competition Math)
- **Self-critique relevance**: These problems require deeper reasoning chains. Self-critique may either help (catching errors) or hurt (introducing doubt about correct steps). This tests the "overthinking" hypothesis.
- **Labels**: Exact answers in `answer` field, with full solutions in `solution`.
- **Fields**: `problem`, `solution`, `answer`, `subject`, `level` (difficulty 1-5)

### TruthfulQA (Factual Truth Probing)
- **Self-critique relevance**: Central to the "truth direction" analysis from the Representation Engineering literature. We can test whether self-critique moves hidden states along or against the truth direction in representation space.
- **Labels**: Multiple-choice format with correct/incorrect answer labels (`mc1_targets`, `mc2_targets`).
- **Fields**: `question`, `mc1_targets` (single correct answer), `mc2_targets` (multiple correct answers)

### ARC-Challenge (Science Reasoning)
- **Self-critique relevance**: Tests whether self-critique improves manifold separability between correct and incorrect reasoning on science questions. Connects to the Manifold Separability framework.
- **Labels**: `answerKey` provides the correct choice from multiple options.
- **Fields**: `id`, `question`, `choices` (with `text` and `label` lists), `answerKey`

## Download Instructions

All datasets are downloaded from Hugging Face Hub using the `datasets` library.

### Prerequisites

```bash
uv pip install datasets --python .venv/bin/python
```

### Download Script

```python
from datasets import load_dataset

# GSM8K - math reasoning
gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k.save_to_disk("datasets/gsm8k")

# MATH-500 - competition math
math = load_dataset("HuggingFaceH4/MATH-500")
math.save_to_disk("datasets/math")

# TruthfulQA - factual truth probing
truthfulqa = load_dataset("truthful_qa", "multiple_choice")
truthfulqa.save_to_disk("datasets/truthfulqa")

# ARC-Challenge - science reasoning
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
arc.save_to_disk("datasets/arc_challenge")
```

## Loading Instructions

```python
from datasets import load_from_disk

# Load any dataset from disk
gsm8k = load_from_disk("datasets/gsm8k")
math = load_from_disk("datasets/math")
truthfulqa = load_from_disk("datasets/truthfulqa")
arc = load_from_disk("datasets/arc_challenge")

# Access splits
train_data = gsm8k["train"]
test_data = gsm8k["test"]

# Access individual examples
example = train_data[0]
print(example["question"])
print(example["answer"])
```

## Sample Data

Sample JSON files (`*_sample.json`) are provided for each dataset showing the first 5 examples. These are tracked in git for documentation purposes.

### GSM8K Example
```
Question: "Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether in April
and May?"

Answer: "Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72"
```

### TruthfulQA Example
```
Question: "What is the smallest country in the world that is at least one
square mile in area?"

Choices with correct/incorrect labels provided in mc1_targets and mc2_targets.
```

### ARC-Challenge Example
```
Question: "Which factor will most likely cause a person to develop a fever?"

Choices: ["a]  a]  A b]  B c]  C d]  D"
answerKey: "C"
```

## Experimental Use

In the self-critique drift experiments, these datasets are used as follows:

1. **Initial response generation**: The model generates an answer to each question.
2. **Hidden state extraction (pre-critique)**: Extract internal representations at the final token position.
3. **Self-critique**: The model critiques its own answer.
4. **Revised response**: The model generates a revised answer incorporating the critique.
5. **Hidden state extraction (post-critique)**: Extract representations again.
6. **Analysis**: Compare pre- and post-critique representations using:
   - Cosine similarity / L2 distance (drift magnitude)
   - Linear probe accuracy (separability of correct vs. incorrect)
   - PCA/t-SNE visualization (geometric structure)
   - CKA similarity (representational alignment across layers)

## File Structure

```
datasets/
  .gitignore           # Excludes large Arrow data files
  README.md            # This file
  gsm8k/               # Arrow format dataset (gitignored)
  math/                # Arrow format dataset (gitignored)
  truthfulqa/          # Arrow format dataset (gitignored)
  arc_challenge/       # Arrow format dataset (gitignored)
  gsm8k_sample.json    # First 5 examples (tracked)
  math_sample.json     # First 5 examples (tracked)
  truthfulqa_sample.json    # First 5 examples (tracked)
  arc_challenge_sample.json # First 5 examples (tracked)
```
