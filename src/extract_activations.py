"""
Extract residual stream activations from Pythia-2.8B using TransformerLens
for three conditions: base, critique, and paraphrase control.
"""

import json
import os
import gc
import numpy as np
import torch
import transformer_lens as tl

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-2.8b"
MAX_SEQ_LEN = 512


def load_model():
    """Load Pythia-2.8B via TransformerLens."""
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = tl.HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.float16,
    )
    model.eval()
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def construct_prompts(stimulus):
    """Construct three prompts from a stimulus entry."""
    q = stimulus["question"]
    initial = stimulus["initial_answer"]
    critique = stimulus["critique"]
    paraphrase = stimulus["paraphrase"]

    base_prompt = f"Question: {q}\nAnswer: {initial}"
    critique_prompt = f"Question: {q}\nAnswer: {initial}\n{critique}"
    paraphrase_prompt = f"Question: {q}\nAnswer: {initial}\n{paraphrase}"

    return base_prompt, critique_prompt, paraphrase_prompt


def extract_residual_stream(model, text, max_len=MAX_SEQ_LEN):
    """
    Extract residual stream activations at the last token for all layers.
    Returns: numpy array of shape (n_layers+1, d_model) — includes embedding layer (layer 0).
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("hook_resid_post") or name == "hook_embed",
        )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    last_pos = tokens.shape[1] - 1

    activations = np.zeros((n_layers + 1, d_model), dtype=np.float32)

    # Layer 0: embedding
    if "hook_embed" in cache:
        activations[0] = cache["hook_embed"][0, last_pos].float().cpu().numpy()

    # Layers 1..n_layers: residual stream after each transformer block
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key in cache:
            activations[layer + 1] = cache[key][0, last_pos].float().cpu().numpy()

    del cache
    return activations, tokens.shape[1]


def extract_all_activations(model, stimuli, save_every=25):
    """Extract activations for all stimuli across three conditions."""
    n = len(stimuli)
    n_layers = model.cfg.n_layers + 1  # +1 for embedding
    d_model = model.cfg.d_model

    # Store activations: (n_samples, n_layers, d_model)
    base_acts = np.zeros((n, n_layers, d_model), dtype=np.float32)
    critique_acts = np.zeros((n, n_layers, d_model), dtype=np.float32)
    paraphrase_acts = np.zeros((n, n_layers, d_model), dtype=np.float32)

    # Store metadata
    seq_lengths = {"base": [], "critique": [], "paraphrase": []}
    labels = []  # 1 if initial answer is correct, 0 if wrong

    for i, stim in enumerate(stimuli):
        if i % 10 == 0:
            print(f"Extracting activations {i}/{n}...")
            gc.collect()
            torch.cuda.empty_cache()

        base_p, crit_p, para_p = construct_prompts(stim)

        base_acts[i], base_len = extract_residual_stream(model, base_p)
        critique_acts[i], crit_len = extract_residual_stream(model, crit_p)
        paraphrase_acts[i], para_len = extract_residual_stream(model, para_p)

        seq_lengths["base"].append(base_len)
        seq_lengths["critique"].append(crit_len)
        seq_lengths["paraphrase"].append(para_len)
        labels.append(1 if stim.get("initial_is_correct", False) else 0)

        if (i + 1) % save_every == 0:
            _save_checkpoint(base_acts, critique_acts, paraphrase_acts,
                           labels, seq_lengths, i + 1)

    # Final save
    _save_checkpoint(base_acts, critique_acts, paraphrase_acts,
                    labels, seq_lengths, n)

    return base_acts, critique_acts, paraphrase_acts, np.array(labels), seq_lengths


def _save_checkpoint(base, critique, paraphrase, labels, seq_lengths, count):
    """Save intermediate results."""
    outdir = "results/data"
    np.save(f"{outdir}/base_activations.npy", base[:count])
    np.save(f"{outdir}/critique_activations.npy", critique[:count])
    np.save(f"{outdir}/paraphrase_activations.npy", paraphrase[:count])
    np.save(f"{outdir}/labels.npy", np.array(labels[:count]))
    with open(f"{outdir}/seq_lengths.json", "w") as f:
        json.dump({k: v[:count] for k, v in seq_lengths.items()}, f)
    print(f"  Checkpoint saved ({count} samples)")


def main():
    # Load stimuli
    stimuli_path = "results/data/stimuli.json"
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    print(f"Loaded {len(stimuli)} stimuli")

    # Load model
    model = load_model()

    # Extract activations
    base_acts, critique_acts, paraphrase_acts, labels, seq_lengths = \
        extract_all_activations(model, stimuli)

    print(f"\nDone! Shapes:")
    print(f"  Base: {base_acts.shape}")
    print(f"  Critique: {critique_acts.shape}")
    print(f"  Paraphrase: {paraphrase_acts.shape}")
    print(f"  Labels: {labels.shape} (correct: {labels.sum()}, wrong: {(1-labels).sum()})")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
