"""
extract_router_data.py
~~~~~~~~~~~~~~~~~~~~~~
Run a prompt through Nemotron-3 Nano and collect MoE routing data as a
structured Python object — no charts, just the raw numbers.

Output shape
------------
RouterData.scores      : np.ndarray  [seq_len, num_moe_layers, n_routed_experts]
RouterData.topk_indices: np.ndarray  [seq_len, num_moe_layers, top_k]
RouterData.topk_weights: np.ndarray  [seq_len, num_moe_layers, top_k]

Where
  seq_len          = number of input tokens
  num_moe_layers   = number of MoE blocks in the network (23 for this config)
  n_routed_experts = 128
  top_k            = 6

scores[t, l, e]  is the sigmoid routing score (before top-k selection) that
                 token t received at MoE layer l for expert e.
                 This is the raw routing probability prior to normalisation.

topk_indices[t, l, :]  are the 6 expert indices actually selected for token t
                       at MoE layer l.

topk_weights[t, l, :]  are the corresponding normalised + scaled routing weights
                       (norm_topk_prob=True, routed_scaling_factor=2.5).

Usage
-----
  python extract_router_data.py \
      --prompt "The attention mechanism in transformers works by" \
      --output router_data.npz

  # or import as a library:
  from extract_router_data import extract
  data = extract(prompt="Hello world", model_path="/path/to/model")
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default local path where the weights are already downloaded.
DEFAULT_MODEL_PATH = "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class RouterData:
    """All MoE routing information for one forward pass.

    Attributes
    ----------
    prompt : str
        The original prompt string.
    tokens : List[str]
        Decoded string for each input token (length = seq_len).
    token_ids : List[int]
        Raw vocabulary ids (length = seq_len).
    moe_layer_indices : List[int]
        Backbone layer indices (0-based) that are MoE blocks.
        These are the indices into model.backbone.layers.
    scores : np.ndarray  shape [seq_len, num_moe_layers, n_routed_experts]
        Sigmoid routing scores — the raw probability each expert receives
        before the top-k gate is applied.  Values in (0, 1).
    topk_indices : np.ndarray  shape [seq_len, num_moe_layers, top_k]
        Expert indices selected by the top-k gate.
    topk_weights : np.ndarray  shape [seq_len, num_moe_layers, top_k]
        Final routing weights after normalisation and scaling.
    """
    prompt: str
    tokens: List[str]
    token_ids: List[int]
    moe_layer_indices: List[int]
    scores: np.ndarray       # [seq_len, num_moe_layers, n_routed_experts]
    topk_indices: np.ndarray  # [seq_len, num_moe_layers, top_k]
    topk_weights: np.ndarray  # [seq_len, num_moe_layers, top_k]


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def extract(
    prompt: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device_map: str = "auto",
) -> RouterData:
    """Load the model (if not already loaded), run *prompt*, and return a
    RouterData object containing per-token per-layer routing information.

    Parameters
    ----------
    prompt : str
        Text to feed the model.
    model_path : str
        Local path or HF hub id for the Nemotron-3 Nano checkpoint.
    device_map : str
        Passed to ``from_pretrained``; ``"auto"`` distributes across all
        visible GPUs.

    Returns
    -------
    RouterData
    """
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
    )
    model.eval()

    return extract_with_model(prompt, model, tokenizer)


def extract_with_model(
    prompt: str,
    model,
    tokenizer,
) -> RouterData:
    """Run *prompt* through an already-loaded model and return RouterData.

    Useful when you want to reuse one loaded model across multiple prompts.

    Parameters
    ----------
    prompt : str
        Text to feed the model.
    model : NemotronHForCausalLM (or AutoModelForCausalLM result)
        The loaded model.
    tokenizer
        Matching tokenizer.

    Returns
    -------
    RouterData
    """
    # ------------------------------------------------------------------
    # 1. Identify MoE layers
    # ------------------------------------------------------------------
    # model.backbone.layers is a ModuleList of NemotronHBlock instances.
    # Each block exposes .block_type which is one of
    #   "mamba" | "attention" | "mlp" | "moe"
    moe_layer_indices: List[int] = []
    for idx, block in enumerate(model.backbone.layers):
        if block.block_type == "moe":
            moe_layer_indices.append(idx)

    if not moe_layer_indices:
        raise RuntimeError(
            "No MoE blocks found in model.backbone.layers. "
            "Check that the model loaded correctly and that block_type is set."
        )

    print(
        f"Found {len(moe_layer_indices)} MoE layers at backbone indices: "
        f"{moe_layer_indices}"
    )

    # ------------------------------------------------------------------
    # 2. Register forward hooks on each router (NemotronHTopkRouter)
    # ------------------------------------------------------------------
    # For each MoE block: model.backbone.layers[i].mixer  →  NemotronHMOE
    #                     .mixer.gate                      →  NemotronHTopkRouter
    #
    # NemotronHTopkRouter.forward(hidden_states) returns (topk_indices, topk_weights)
    # but we also want the *full* sigmoid scores before the top-k selection.
    # We recompute them cheaply inside the hook using the captured inputs.

    # Ordered list of captures — one entry per MoE layer, in layer order.
    captures: List[dict] = [None] * len(moe_layer_indices)
    layer_idx_to_capture_slot = {
        layer_idx: slot for slot, layer_idx in enumerate(moe_layer_indices)
    }

    def make_hook(backbone_layer_idx: int):
        slot = layer_idx_to_capture_slot[backbone_layer_idx]

        def hook(module, inputs, outputs):
            # inputs: tuple — first element is hidden_states BEFORE the
            # view(-1, hidden_size) inside forward.
            # outputs: (topk_indices, topk_weights)
            hidden_states = inputs[0]  # (batch, seq_len, hidden_size) or already flat

            # Recompute sigmoid scores over all 128 experts.
            # This is a single linear op — negligible overhead.
            flat = hidden_states.reshape(-1, module.config.hidden_size).to(torch.float32)
            with torch.no_grad():
                router_logits = F.linear(flat, module.weight.to(torch.float32))
                scores = router_logits.sigmoid()  # (seq_len, n_routed_experts)

            topk_indices, topk_weights = outputs
            captures[slot] = {
                "scores": scores.cpu().float().numpy(),         # (seq_len, 128)
                "topk_indices": topk_indices.cpu().numpy(),     # (seq_len, top_k)
                "topk_weights": topk_weights.cpu().float().numpy(),  # (seq_len, top_k)
            }

        return hook

    handles = []
    for layer_idx in moe_layer_indices:
        gate = model.backbone.layers[layer_idx].mixer.gate
        handles.append(gate.register_forward_hook(make_hook(layer_idx)))

    # ------------------------------------------------------------------
    # 3. Tokenise and run a single forward pass
    # ------------------------------------------------------------------
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    # Move input_ids to the same device as the embedding layer.
    first_param_device = next(model.parameters()).device
    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

    token_ids: List[int] = inputs["input_ids"][0].tolist()
    tokens: List[str] = [
        tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids
    ]
    seq_len = len(token_ids)

    print(f"Running forward pass on {seq_len} tokens ...")
    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks.
    for h in handles:
        h.remove()

    # ------------------------------------------------------------------
    # 4. Assemble the output arrays
    # ------------------------------------------------------------------
    num_moe_layers = len(moe_layer_indices)

    # Determine n_routed_experts and top_k from the first capture.
    first = captures[0]
    n_routed_experts = first["scores"].shape[-1]
    top_k = first["topk_indices"].shape[-1]

    # Assert routing dimensions match model config.
    cfg_n_experts = model.config.n_routed_experts
    cfg_top_k = model.config.num_experts_per_tok
    assert n_routed_experts == cfg_n_experts, (
        f"Captured expert dimension {n_routed_experts} != "
        f"model.config.n_routed_experts={cfg_n_experts}"
    )
    assert top_k == cfg_top_k, (
        f"Captured top-k {top_k} != "
        f"model.config.num_experts_per_tok={cfg_top_k}"
    )

    scores_arr = np.zeros((seq_len, num_moe_layers, n_routed_experts), dtype=np.float32)
    topk_idx_arr = np.zeros((seq_len, num_moe_layers, top_k), dtype=np.int32)
    topk_w_arr = np.zeros((seq_len, num_moe_layers, top_k), dtype=np.float32)

    for slot, cap in enumerate(captures):
        if cap is None:
            raise RuntimeError(
                f"Hook for MoE layer slot {slot} (backbone layer "
                f"{moe_layer_indices[slot]}) did not fire. "
                "Check that the model ran the MoE block during the forward pass."
            )
        scores_arr[:, slot, :] = cap["scores"]          # (seq_len, 128)
        topk_idx_arr[:, slot, :] = cap["topk_indices"]  # (seq_len, top_k)
        topk_w_arr[:, slot, :] = cap["topk_weights"]    # (seq_len, top_k)

    print(
        f"Collected routing data: "
        f"tokens={seq_len}, moe_layers={num_moe_layers}, "
        f"experts={n_routed_experts}, top_k={top_k}"
    )
    print(f"  scores shape:       {scores_arr.shape}")
    print(f"  topk_indices shape: {topk_idx_arr.shape}")
    print(f"  topk_weights shape: {topk_w_arr.shape}")

    return RouterData(
        prompt=prompt,
        tokens=tokens,
        token_ids=token_ids,
        moe_layer_indices=moe_layer_indices,
        scores=scores_arr,
        topk_indices=topk_idx_arr,
        topk_weights=topk_w_arr,
    )


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save(data: RouterData, path: str) -> None:
    """Save a RouterData object to an .npz file."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez(
        path,
        scores=data.scores,
        topk_indices=data.topk_indices,
        topk_weights=data.topk_weights,
        token_ids=np.array(data.token_ids, dtype=np.int32),
        moe_layer_indices=np.array(data.moe_layer_indices, dtype=np.int32),
        # Store strings as object arrays.
        tokens=np.array(data.tokens, dtype=object),
        prompt=np.array([data.prompt], dtype=object),
    )
    print(f"Saved RouterData to {path}")


def load(path: str) -> RouterData:
    """Load a RouterData object previously saved with save()."""
    d = np.load(path, allow_pickle=True)
    return RouterData(
        prompt=str(d["prompt"][0]),
        tokens=d["tokens"].tolist(),
        token_ids=d["token_ids"].tolist(),
        moe_layer_indices=d["moe_layer_indices"].tolist(),
        scores=d["scores"],
        topk_indices=d["topk_indices"],
        topk_weights=d["topk_weights"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Nemotron-3 Nano MoE routing data for a prompt."
    )
    parser.add_argument(
        "--prompt",
        default="The attention mechanism in transformers works by",
        help="Prompt text to feed the model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local path or HF hub id for the Nemotron-3 Nano checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="router_data.npz",
        help="Where to write the .npz file (default: router_data.npz).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to from_pretrained (default: auto).",
    )
    args = parser.parse_args()

    data = extract(
        prompt=args.prompt,
        model_path=args.model,
        device_map=args.device_map,
    )
    save(data, args.output)

    # Print a quick sanity summary.
    print("\n--- Quick summary ---")
    print(f"Prompt   : {data.prompt!r}")
    print(f"Tokens   : {data.tokens}")
    print(f"MoE layers ({len(data.moe_layer_indices)}): {data.moe_layer_indices}")
    print(f"scores      shape: {data.scores.shape}  dtype={data.scores.dtype}")
    print(f"topk_indices shape: {data.topk_indices.shape}  dtype={data.topk_indices.dtype}")
    print(f"topk_weights shape: {data.topk_weights.shape}  dtype={data.topk_weights.dtype}")

    # Show which experts token 0 chose at each layer.
    print("\nToken 0 top-6 expert selections per MoE layer:")
    for slot, layer_idx in enumerate(data.moe_layer_indices):
        experts = data.topk_indices[0, slot, :]
        weights = data.topk_weights[0, slot, :]
        print(f"  backbone layer {layer_idx:2d}: experts={experts.tolist()}  weights={weights.round(4).tolist()}")


if __name__ == "__main__":
    main()
