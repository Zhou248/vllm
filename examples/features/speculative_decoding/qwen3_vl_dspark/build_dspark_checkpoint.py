#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct an untrained Qwen3-VL DSpARK checkpoint from its real config.

The checkpoint contains deterministic structural weights for vLLM and
vLLM-Ascend integration testing. It is not a trained speculative model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_SOURCE_CONFIG = REPO_ROOT / "config.json"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "artifacts" / "Qwen3-VL-DSpARK-config-fixture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=DEFAULT_SOURCE_CONFIG,
        help="Training-side qwen3_vl_dflash config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def load_source_config(path: Path) -> dict[str, Any]:
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("model_type") != "qwen3_vl_dflash":
        raise ValueError(
            "source config must use model_type='qwen3_vl_dflash'; got "
            f"{root.get('model_type')!r}"
        )
    architectures = root.get("architectures")
    if not isinstance(architectures, list) or (
        "Qwen3VLForConditionalGenerationDFlash" not in architectures
    ):
        raise ValueError(
            "source config must declare Qwen3VLForConditionalGenerationDFlash"
        )
    require_mapping(root.get("text_config"), "text_config")
    require_mapping(root.get("dflash_config"), "dflash_config")
    return root


def derive_draft_geometry(source: Mapping[str, Any]) -> dict[str, Any]:
    text = require_mapping(source.get("text_config"), "text_config")
    dflash = require_mapping(source.get("dflash_config"), "dflash_config")

    target_num_layers = positive_int(
        text.get("num_hidden_layers"), "text_config.num_hidden_layers"
    )
    if dflash.get("num_target_layers") != target_num_layers:
        raise ValueError(
            "dflash_config.num_target_layers must match "
            f"text_config.num_hidden_layers ({target_num_layers})"
        )

    num_draft_layers = positive_int(
        dflash.get("num_hidden_layers"), "dflash_config.num_hidden_layers"
    )
    layer_types = dflash.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_draft_layers:
        raise ValueError(
            "dflash_config.layer_types must contain one entry per draft layer"
        )

    target_layer_ids = dflash.get("target_layer_ids")
    if (
        not isinstance(target_layer_ids, list)
        or not target_layer_ids
        or any(
            not isinstance(layer_id, int) or isinstance(layer_id, bool)
            for layer_id in target_layer_ids
        )
    ):
        raise ValueError("dflash_config.target_layer_ids must be an integer list")
    if target_layer_ids != sorted(set(target_layer_ids)):
        raise ValueError("target_layer_ids must be unique and strictly increasing")
    if target_layer_ids[0] < 0 or target_layer_ids[-1] >= target_num_layers:
        raise ValueError(
            f"target_layer_ids must be within [0, {target_num_layers - 1}]"
        )
    if dflash.get("num_target_feature_layers") != len(target_layer_ids):
        raise ValueError(
            "dflash_config.num_target_feature_layers must match the number of "
            "target_layer_ids"
        )

    vocab_size = positive_int(text.get("vocab_size"), "text_config.vocab_size")
    draft_vocab_size = positive_int(
        dflash.get("draft_vocab_size", vocab_size), "draft_vocab_size"
    )
    mask_token_id = dflash.get("mask_token_id")
    if (
        not isinstance(mask_token_id, int)
        or isinstance(mask_token_id, bool)
        or not 0 <= mask_token_id < vocab_size
    ):
        raise ValueError(
            f"dflash_config.mask_token_id must be within [0, {vocab_size - 1}]"
        )

    dtype_name = dflash.get("dtype", text.get("dtype", source.get("dtype")))
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported draft dtype {dtype_name!r}")

    return {
        "block_size": positive_int(
            dflash.get("block_size"), "dflash_config.block_size"
        ),
        "draft_vocab_size": draft_vocab_size,
        "dtype": dtype_by_name[dtype_name],
        "head_dim": positive_int(text.get("head_dim"), "text_config.head_dim"),
        "hidden_size": positive_int(text.get("hidden_size"), "text_config.hidden_size"),
        "intermediate_size": positive_int(
            text.get("intermediate_size"), "text_config.intermediate_size"
        ),
        "layer_types": list(layer_types),
        "markov_rank": positive_int(
            dflash.get("markov_rank"), "dflash_config.markov_rank"
        ),
        "num_attention_heads": positive_int(
            text.get("num_attention_heads"), "text_config.num_attention_heads"
        ),
        "num_draft_layers": num_draft_layers,
        "num_key_value_heads": positive_int(
            text.get("num_key_value_heads"), "text_config.num_key_value_heads"
        ),
        "target_hidden_size": positive_int(
            text.get("hidden_size"), "text_config.hidden_size"
        ),
        "target_layer_ids": list(target_layer_ids),
        "vocab_size": vocab_size,
    }


def full(shape: tuple[int, ...], value: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype)


def build_weights(
    geometry: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    hidden_size = geometry["hidden_size"]
    intermediate_size = geometry["intermediate_size"]
    num_heads = geometry["num_attention_heads"]
    num_kv_heads = geometry["num_key_value_heads"]
    head_dim = geometry["head_dim"]
    target_features = len(geometry["target_layer_ids"])
    target_hidden_size = geometry["target_hidden_size"]
    dtype = geometry["dtype"]

    def zeros(*shape: int) -> torch.Tensor:
        return full(shape, 0.0, dtype)

    def ones(*shape: int) -> torch.Tensor:
        return full(shape, 1.0, dtype)

    weights: dict[str, torch.Tensor] = {
        "fc.weight": zeros(hidden_size, target_features * target_hidden_size),
        "hidden_norm.weight": ones(hidden_size),
        "markov_head.markov_w1.weight": zeros(
            geometry["vocab_size"], geometry["markov_rank"]
        ),
        "markov_head.markov_w2.weight": zeros(
            geometry["draft_vocab_size"], geometry["markov_rank"]
        ),
        "norm.weight": ones(hidden_size),
    }
    for layer_idx in range(geometry["num_draft_layers"]):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    source_config = load_source_config(args.source_config)
    geometry = derive_draft_geometry(source_config)
    weights = build_weights(geometry)

    ensure_output_is_writable(args.output_dir, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    weights_path = args.output_dir / "model.safetensors"
    config_bytes = (json.dumps(source_config, indent=2, sort_keys=True) + "\n").encode()
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
        "derived_contract": {
            key: value for key, value in geometry.items() if key != "dtype"
        }
        | {"dtype": str(geometry["dtype"]).removeprefix("torch.")},
        "omitted_runtime_shared_weights": [
            "embed_tokens.weight",
            "lm_head.weight",
        ],
        "omitted_training_only_weights": [
            "confidence_head.*",
        ],
        "purpose": "structural integration test only; weights are untrained",
        "source_config_sha256": sha256_file(args.source_config),
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
