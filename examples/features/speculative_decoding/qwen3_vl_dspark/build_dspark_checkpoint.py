#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct an untrained Qwen3-VL-30B-A3B DSpARK checkpoint.

The checkpoint is intended for vLLM/vLLM-Ascend integration testing only. It
contains deterministic structural weights, not a trained speculative model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    EXAMPLE_DIR / "artifacts" / "Qwen3-VL-30B-A3B-Instruct-DSpARK-fixture"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-config",
        type=Path,
        default=EXAMPLE_DIR / "target_config.json",
        help="Qwen3-VL root config.json used to derive draft geometry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--markov-rank", type=int, default=8)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument(
        "--target-layer-ids",
        default="11,23,35,47",
        help="Comma-separated, zero-based target text-layer indices.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing generated checkpoint.",
    )
    return parser.parse_args()


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def load_target_text_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("model_type") not in {"qwen3_vl", "qwen3_vl_moe"}:
        raise ValueError(
            "target config must describe Qwen3-VL or Qwen3-VL-MoE; got "
            f"model_type={root.get('model_type')!r}"
        )
    text = root.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("target config does not contain a text_config object")
    return root, text


def build_draft_config(
    target_root: dict[str, Any],
    text: dict[str, Any],
    args: argparse.Namespace,
    target_layer_ids: list[int],
) -> dict[str, Any]:
    hidden_size = positive_int(text.get("hidden_size"), "text hidden_size")
    target_num_layers = positive_int(
        text.get("num_hidden_layers"), "text num_hidden_layers"
    )
    if not target_layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    if target_layer_ids != sorted(set(target_layer_ids)):
        raise ValueError("target_layer_ids must be unique and strictly increasing")
    if target_layer_ids[0] < 0 or target_layer_ids[-1] >= target_num_layers:
        raise ValueError(
            f"target_layer_ids must be within [0, {target_num_layers - 1}]"
        )

    vocab_size = positive_int(text.get("vocab_size"), "text vocab_size")
    if not 0 <= args.mask_token_id < vocab_size:
        raise ValueError(
            f"mask_token_id must be within target vocabulary [0, {vocab_size - 1}]"
        )

    rope_theta = float(text.get("rope_theta", 1_000_000.0))
    return {
        "architectures": ["Qwen3VLDSparkModel"],
        "attention_bias": bool(text.get("attention_bias", False)),
        "attention_dropout": float(text.get("attention_dropout", 0.0)),
        "block_size": args.block_size,
        "bos_token_id": text.get("bos_token_id"),
        "dflash_config": {
            "causal": False,
            "mask_token_id": args.mask_token_id,
            "target_layer_ids": target_layer_ids,
            "use_aux_hidden_state": True,
        },
        "draft_vocab_size": vocab_size,
        "dspark_block_size": args.block_size,
        "dspark_bonus_anchor": False,
        "dspark_target_layer_ids": target_layer_ids,
        "dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [i + 1 for i in target_layer_ids],
        "enable_confidence_head": False,
        "eos_token_id": text.get("eos_token_id"),
        "fixture_metadata": {
            "purpose": "structural integration test only; weights are untrained",
            "target_architectures": target_root.get("architectures", []),
            "target_model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "target_model_type": target_root.get("model_type"),
        },
        "head_dim": positive_int(text.get("head_dim"), "text head_dim"),
        "hidden_act": text.get("hidden_act", "silu"),
        "hidden_size": hidden_size,
        "initializer_range": float(text.get("initializer_range", 0.02)),
        "intermediate_size": positive_int(
            text.get("intermediate_size"), "text intermediate_size"
        ),
        "markov_head_type": "vanilla",
        "markov_rank": args.markov_rank,
        "mask_token_id": args.mask_token_id,
        "max_position_embeddings": positive_int(
            text.get("max_position_embeddings"), "text max_position_embeddings"
        ),
        "model_type": "qwen3",
        "n_predict": args.block_size,
        "num_attention_heads": positive_int(
            text.get("num_attention_heads"), "text num_attention_heads"
        ),
        "num_hidden_layers": args.num_draft_layers,
        "num_key_value_heads": positive_int(
            text.get("num_key_value_heads"), "text num_key_value_heads"
        ),
        "rms_norm_eps": float(text.get("rms_norm_eps", 1e-6)),
        "rope_parameters": {
            "rope_theta": rope_theta,
            "rope_type": "default",
        },
        "rope_theta": rope_theta,
        "sample_from_anchor": True,
        "target_hidden_size": hidden_size,
        "target_layer_ids": target_layer_ids,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_aux_hidden_state": True,
        "use_cache": True,
        "vocab_size": vocab_size,
    }


def zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.bfloat16)


def ones(*shape: int) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.bfloat16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_weights(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    target_features = len(config["target_layer_ids"])
    target_hidden_size = config["target_hidden_size"]
    vocab_size = config["vocab_size"]
    draft_vocab_size = config["draft_vocab_size"]
    markov_rank = config["markov_rank"]

    weights: dict[str, torch.Tensor] = {
        "fc.weight": zeros(hidden_size, target_features * target_hidden_size),
        "hidden_norm.weight": ones(hidden_size),
        "markov_head.markov_w1.weight": zeros(vocab_size, markov_rank),
        "markov_head.markov_w2.weight": zeros(draft_vocab_size, markov_rank),
        "norm.weight": ones(hidden_size),
    }
    for layer_idx in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": ones(hidden_size),
                f"{prefix}.mlp.down_proj.weight": zeros(hidden_size, intermediate_size),
                f"{prefix}.mlp.gate_proj.weight": zeros(intermediate_size, hidden_size),
                f"{prefix}.mlp.up_proj.weight": zeros(intermediate_size, hidden_size),
                f"{prefix}.post_attention_layernorm.weight": ones(hidden_size),
                f"{prefix}.self_attn.k_norm.weight": ones(head_dim),
                f"{prefix}.self_attn.k_proj.weight": zeros(
                    num_kv_heads * head_dim, hidden_size
                ),
                f"{prefix}.self_attn.o_proj.weight": zeros(
                    hidden_size, num_heads * head_dim
                ),
                f"{prefix}.self_attn.q_norm.weight": ones(head_dim),
                f"{prefix}.self_attn.q_proj.weight": zeros(
                    num_heads * head_dim, hidden_size
                ),
                f"{prefix}.self_attn.v_proj.weight": zeros(
                    num_kv_heads * head_dim, hidden_size
                ),
            }
        )
    return weights


def ensure_output_is_writable(output_dir: Path, force: bool) -> None:
    generated_files = (
        output_dir / "config.json",
        output_dir / "model.safetensors",
        output_dir / "fixture_manifest.json",
    )
    existing = [path for path in generated_files if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite {names} in {output_dir}; pass --force"
        )


def main() -> None:
    args = parse_args()
    positive_int(args.num_draft_layers, "num_draft_layers")
    positive_int(args.block_size, "block_size")
    positive_int(args.markov_rank, "markov_rank")
    target_layer_ids = [
        int(value.strip())
        for value in args.target_layer_ids.split(",")
        if value.strip()
    ]

    target_root, text = load_target_text_config(args.target_config)
    config = build_draft_config(target_root, text, args, target_layer_ids)
    weights = build_weights(config)

    ensure_output_is_writable(args.output_dir, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    weights_path = args.output_dir / "model.safetensors"
    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    config_path.write_bytes(config_bytes)
    save_file(
        weights,
        str(weights_path),
        metadata={"format": "pt", "fixture": "untrained-qwen3-vl-dspark"},
    )

    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in weights.values()
    )
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "omitted_shared_weights": [
            "embed_tokens.weight",
            "lm_head.weight",
        ],
        "source_target_config_sha256": sha256_file(args.target_config),
        "tensor_bytes": tensor_bytes,
        "tensor_count": len(weights),
        "tensors": {
            name: list(tensor.shape) for name, tensor in sorted(weights.items())
        },
        "weights_sha256": sha256_file(weights_path),
    }
    (args.output_dir / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {config_path}")
    print(f"wrote {weights_path} ({tensor_bytes / 1024**2:.1f} MiB of tensors)")
    print(f"weights sha256: {manifest['weights_sha256']}")


if __name__ == "__main__":
    main()
