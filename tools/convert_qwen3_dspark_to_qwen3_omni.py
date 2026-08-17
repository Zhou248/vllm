# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a dedicated Qwen3-Omni DSpark checkpoint for smoke testing.

The converted checkpoint is shape-compatible, not trained for Qwen3-Omni. It is
intended only to validate model loading and the speculative-decoding data path.
The converter rebuilds the dense draft's hidden width, Q/KV head geometry,
attention and MLP projections, norms, FC bridge, vocab-facing tensors, and
Markov tables to dimensions accepted by the Omni thinker contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class TargetTextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int | None


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ConversionPlan:
    source_feature_count: int
    source_target_hidden_size: int
    target_layer_ids: tuple[int, ...]
    target: TargetTextConfig
    source_draft_hidden_size: int
    source_intermediate_size: int
    draft_num_hidden_layers: int
    source_num_attention_heads: int
    source_num_key_value_heads: int
    source_head_dim: int
    draft_input_vocab_size: int
    draft_output_vocab_size: int
    markov_rank: int
    mask_token_id: int
    fc_key: str
    embed_key: str | None
    lm_head_key: str | None
    markov_w1_key: str
    markov_w2_key: str
    d2t_key: str | None
    attention_projection_keys: frozenset[str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_config(
    model: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[dict[str, Any], Path]:
    local_path = Path(model).expanduser()
    if local_path.is_file():
        config_path = local_path
    elif local_path.is_dir():
        config_path = local_path / "config.json"
    else:
        config_path = Path(
            hf_hub_download(
                repo_id=model,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot find config.json for {model!r}.")
    return _load_json(config_path), config_path


def _resolve_draft_directory(
    model: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> Path:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    if local_path.exists():
        raise ValueError("--draft-model must be a model directory or Hub model ID.")
    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
            local_files_only=local_files_only,
        )
    )


def _as_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _extract_target_text_config(config: dict[str, Any]) -> TargetTextConfig:
    architectures = set(config.get("architectures") or ())
    model_type = config.get("model_type")
    is_omni = bool(
        architectures
        & {
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen3OmniMoeThinkerForConditionalGeneration",
        }
    ) or model_type in {"qwen3_omni_moe", "qwen3_omni_moe_thinker"}
    if not is_omni:
        raise ValueError("--target-model must be a Qwen3-Omni checkpoint.")

    thinker_config = config.get("thinker_config")
    if isinstance(thinker_config, dict):
        text_config = thinker_config.get("text_config")
    else:
        text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen3-Omni config does not contain thinker text_config.")

    max_position_embeddings = text_config.get("max_position_embeddings")
    if max_position_embeddings is not None:
        max_position_embeddings = _as_positive_int(
            max_position_embeddings, "target max_position_embeddings"
        )
    hidden_size = _as_positive_int(
        text_config.get("hidden_size"), "target text hidden_size"
    )
    num_attention_heads = _as_positive_int(
        text_config.get("num_attention_heads"),
        "target text num_attention_heads",
    )
    num_key_value_heads = _as_positive_int(
        text_config.get("num_key_value_heads", num_attention_heads),
        "target text num_key_value_heads",
    )
    head_dim = text_config.get("head_dim")
    if head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "Target text config must define head_dim when hidden_size is not "
                "divisible by num_attention_heads."
            )
        head_dim = hidden_size // num_attention_heads
    return TargetTextConfig(
        hidden_size=hidden_size,
        num_hidden_layers=_as_positive_int(
            text_config.get("num_hidden_layers"), "target text num_hidden_layers"
        ),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=_as_positive_int(head_dim, "target text head_dim"),
        vocab_size=_as_positive_int(
            text_config.get("vocab_size"), "target text vocab_size"
        ),
        max_position_embeddings=max_position_embeddings,
    )


def _weight_files(directory: Path) -> list[Path]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index_path}.")
        names = sorted(set(weight_map.values()))
        files = [directory / name for name in names]
    elif (directory / "model.safetensors").is_file():
        files = [directory / "model.safetensors"]
    else:
        files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in {directory}.")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}.")
    return files


def _inspect_tensors(files: list[Path]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                if key in tensors:
                    raise ValueError(f"Duplicate tensor {key!r} in checkpoint shards.")
                shape = tuple(handle.get_slice(key).get_shape())
                tensors[key] = TensorInfo(file=path, shape=shape)
    return tensors


def _find_tensor_key(
    tensors: dict[str, TensorInfo],
    suffix: str,
    *,
    required: bool,
) -> str | None:
    matches = [key for key in tensors if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) > 1:
        raise ValueError(f"Multiple tensors match {suffix!r}: {matches}.")
    if not matches:
        if required:
            raise ValueError(f"Required tensor {suffix!r} is missing.")
        return None
    return matches[0]


def _find_mapping_key(tensors: dict[str, TensorInfo], names: set[str]) -> str | None:
    matches = [key for key in tensors if key.split(".")[-1] in names]
    if len(matches) > 1:
        raise ValueError(f"Multiple token-mapping tensors found: {matches}.")
    return matches[0] if matches else None


def _source_target_layer_ids(config: dict[str, Any]) -> tuple[int, ...]:
    dflash_config = config.get("dflash_config")
    if isinstance(dflash_config, dict):
        layer_ids = dflash_config.get("target_layer_ids")
    else:
        layer_ids = None
    if layer_ids is None:
        layer_ids = config.get("dspark_target_layer_ids")
    if layer_ids is None:
        layer_ids = config.get("target_layer_ids")
    if not isinstance(layer_ids, list) or not layer_ids:
        raise ValueError("Draft config must define non-empty target_layer_ids.")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in layer_ids):
        raise ValueError("Draft target_layer_ids must contain only integers.")
    return tuple(layer_ids)


def _validate_target_layer_ids(
    layer_ids: tuple[int, ...], num_hidden_layers: int
) -> None:
    if not layer_ids:
        raise ValueError("At least one target layer ID is required.")
    if tuple(sorted(set(layer_ids))) != layer_ids:
        raise ValueError("Target layer IDs must be unique and strictly increasing.")
    if layer_ids[0] < 0 or layer_ids[-1] >= num_hidden_layers:
        raise ValueError(
            "Target layer IDs must be in "
            f"[0, {num_hidden_layers - 1}]; got {list(layer_ids)}."
        )


def _reject_unsupported_draft_config(config: dict[str, Any]) -> None:
    architectures = set(config.get("architectures") or ())
    if not architectures & {"Qwen3DSparkModel", "Qwen3OmniDSparkModel"}:
        raise ValueError(
            "--draft-model must declare Qwen3DSparkModel or "
            "Qwen3OmniDSparkModel in architectures."
        )
    if config.get("quantization_config") or config.get("quant_method"):
        raise ValueError(
            "Quantized draft checkpoints are not supported by this smoke converter."
        )
    rope_configs = (config.get("rope_parameters"), config.get("rope_scaling"))
    if config.get("mrope_section") is not None or any(
        isinstance(value, dict) and "mrope_section" in value for value in rope_configs
    ):
        raise ValueError("The DSpark draft must use logical 1-D RoPE, not MRoPE.")


def _attention_projection_keys(
    tensors: dict[str, TensorInfo], suffix: str
) -> tuple[str, ...]:
    return tuple(
        sorted(key for key in tensors if key == suffix or key.endswith(f".{suffix}"))
    )


def _validate_attention_weights(
    tensors: dict[str, TensorInfo],
    *,
    num_hidden_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> frozenset[str]:
    packed_keys = _attention_projection_keys(tensors, "self_attn.qkv_proj.weight")
    if packed_keys:
        raise ValueError(
            "Packed qkv_proj weights are not supported by this converter; "
            "use a checkpoint with separate q_proj/k_proj/v_proj tensors."
        )

    suffixes = {
        "q": "self_attn.q_proj.weight",
        "k": "self_attn.k_proj.weight",
        "v": "self_attn.v_proj.weight",
        "o": "self_attn.o_proj.weight",
    }
    keys = {
        name: _attention_projection_keys(tensors, suffix)
        for name, suffix in suffixes.items()
    }
    for name, projection_keys in keys.items():
        if len(projection_keys) != num_hidden_layers:
            raise ValueError(
                f"Expected {num_hidden_layers} {name}_proj weight tensors; "
                f"found {len(projection_keys)}."
            )

    def prefixes(name: str) -> set[str]:
        suffix = suffixes[name]
        return {key[: -len(suffix)] for key in keys[name]}

    q_prefixes = prefixes("q")
    for name in ("k", "v", "o"):
        if prefixes(name) != q_prefixes:
            raise ValueError(
                f"Attention {name}_proj tensors do not cover the same layers "
                "as q_proj tensors."
            )

    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    expected_shapes = {
        "q": (q_size, hidden_size),
        "k": (kv_size, hidden_size),
        "v": (kv_size, hidden_size),
        "o": (hidden_size, q_size),
    }
    for name, projection_keys in keys.items():
        expected = expected_shapes[name]
        for key in projection_keys:
            if tensors[key].shape != expected:
                raise ValueError(
                    f"{key} must have shape {expected}; got {tensors[key].shape}."
                )

    resize_keys = set().union(*keys.values())
    for prefix in q_prefixes:
        for name in ("q", "k", "v", "o"):
            bias_key = f"{prefix}self_attn.{name}_proj.bias"
            if bias_key in tensors:
                projection_size = {
                    "q": q_size,
                    "k": kv_size,
                    "v": kv_size,
                    "o": hidden_size,
                }[name]
                if tensors[bias_key].shape != (projection_size,):
                    raise ValueError(
                        f"{bias_key} must have shape {(projection_size,)}; got "
                        f"{tensors[bias_key].shape}."
                    )
                resize_keys.add(bias_key)
    return frozenset(resize_keys)


def _make_plan(
    draft_config: dict[str, Any],
    target: TargetTextConfig,
    tensors: dict[str, TensorInfo],
    requested_layer_ids: tuple[int, ...] | None,
) -> ConversionPlan:
    _reject_unsupported_draft_config(draft_config)
    source_layer_ids = _source_target_layer_ids(draft_config)
    target_layer_ids = requested_layer_ids or source_layer_ids
    _validate_target_layer_ids(target_layer_ids, target.num_hidden_layers)

    source_draft_hidden_size = _as_positive_int(
        draft_config.get("hidden_size"), "draft hidden_size"
    )
    source_intermediate_size = _as_positive_int(
        draft_config.get("intermediate_size"), "draft intermediate_size"
    )
    draft_num_hidden_layers = _as_positive_int(
        draft_config.get("num_hidden_layers"), "draft num_hidden_layers"
    )
    source_num_attention_heads = _as_positive_int(
        draft_config.get("num_attention_heads"), "draft num_attention_heads"
    )
    source_num_key_value_heads = _as_positive_int(
        draft_config.get("num_key_value_heads", source_num_attention_heads),
        "draft num_key_value_heads",
    )
    source_head_dim = draft_config.get("head_dim")
    if source_head_dim is None:
        if source_draft_hidden_size % source_num_attention_heads != 0:
            raise ValueError(
                "Draft config must define head_dim when hidden_size is not "
                "divisible by num_attention_heads."
            )
        source_head_dim = source_draft_hidden_size // source_num_attention_heads
    source_head_dim = _as_positive_int(source_head_dim, "draft head_dim")
    attention_projection_keys = _validate_attention_weights(
        tensors,
        num_hidden_layers=draft_num_hidden_layers,
        hidden_size=source_draft_hidden_size,
        num_attention_heads=source_num_attention_heads,
        num_key_value_heads=source_num_key_value_heads,
        head_dim=source_head_dim,
    )
    markov_rank = _as_positive_int(draft_config.get("markov_rank"), "draft markov_rank")
    source_input_vocab = _as_positive_int(
        draft_config.get("vocab_size"), "draft vocab_size"
    )
    _as_positive_int(
        draft_config.get("draft_vocab_size") or source_input_vocab,
        "draft output vocab size",
    )
    source_mask_token_id = draft_config.get("mask_token_id")
    if not isinstance(source_mask_token_id, int) or isinstance(
        source_mask_token_id, bool
    ):
        source_mask_token_id = target.vocab_size - 1
    mask_token_id = min(max(source_mask_token_id, 0), target.vocab_size - 1)
    # The smoke checkpoint uses the full target vocabulary. This avoids a
    # training-only d2t mapping and keeps all NPU runtime dimensions target-led.
    draft_input_vocab = target.vocab_size
    draft_output_vocab = target.vocab_size

    fc_key = _find_tensor_key(tensors, "fc.weight", required=True)
    assert fc_key is not None
    fc_shape = tensors[fc_key].shape
    if len(fc_shape) != 2 or fc_shape[0] != source_draft_hidden_size:
        raise ValueError(
            f"{fc_key} must have shape [{source_draft_hidden_size}, input_size]; "
            f"got {fc_shape}."
        )
    if fc_shape[1] % len(source_layer_ids) != 0:
        raise ValueError(
            f"{fc_key} input size {fc_shape[1]} is not divisible by the source "
            f"feature count {len(source_layer_ids)}."
        )
    source_target_hidden = fc_shape[1] // len(source_layer_ids)
    configured_target_hidden = draft_config.get("target_hidden_size")
    if configured_target_hidden is not None and configured_target_hidden != (
        source_target_hidden
    ):
        raise ValueError(
            "Draft target_hidden_size disagrees with fc.weight: "
            f"{configured_target_hidden} != {source_target_hidden}."
        )

    embed_key = _find_tensor_key(tensors, "embed_tokens.weight", required=False)
    lm_head_key = _find_tensor_key(tensors, "lm_head.weight", required=False)
    markov_w1_key = _find_tensor_key(
        tensors, "markov_head.markov_w1.weight", required=True
    )
    markov_w2_key = _find_tensor_key(
        tensors, "markov_head.markov_w2.weight", required=True
    )
    assert markov_w1_key is not None and markov_w2_key is not None

    _validate_matrix_columns(tensors, embed_key, source_draft_hidden_size)
    _validate_matrix_columns(tensors, lm_head_key, source_draft_hidden_size)
    _validate_matrix_columns(tensors, markov_w1_key, markov_rank)
    _validate_matrix_columns(tensors, markov_w2_key, markov_rank)

    return ConversionPlan(
        source_feature_count=len(source_layer_ids),
        source_target_hidden_size=source_target_hidden,
        target_layer_ids=target_layer_ids,
        target=target,
        source_draft_hidden_size=source_draft_hidden_size,
        source_intermediate_size=source_intermediate_size,
        draft_num_hidden_layers=draft_num_hidden_layers,
        source_num_attention_heads=source_num_attention_heads,
        source_num_key_value_heads=source_num_key_value_heads,
        source_head_dim=source_head_dim,
        draft_input_vocab_size=draft_input_vocab,
        draft_output_vocab_size=draft_output_vocab,
        markov_rank=markov_rank,
        mask_token_id=mask_token_id,
        fc_key=fc_key,
        embed_key=embed_key,
        lm_head_key=lm_head_key,
        markov_w1_key=markov_w1_key,
        markov_w2_key=markov_w2_key,
        d2t_key=_find_mapping_key(tensors, {"d2t", "draft_id_to_target_id"}),
        attention_projection_keys=attention_projection_keys,
    )


def _validate_matrix_columns(
    tensors: dict[str, TensorInfo], key: str | None, columns: int
) -> None:
    if key is None:
        return
    shape = tensors[key].shape
    if len(shape) != 2 or shape[1] != columns:
        raise ValueError(f"{key} must have {columns} columns; got {shape}.")


def _converted_config(source: dict[str, Any], plan: ConversionPlan) -> dict[str, Any]:
    config = copy.deepcopy(source)
    layer_ids = list(plan.target_layer_ids)
    config.update(
        {
            "architectures": ["Qwen3OmniDSparkModel"],
            "model_type": "qwen3",
            "hidden_size": plan.target.hidden_size,
            "num_attention_heads": plan.target.num_attention_heads,
            "num_key_value_heads": plan.target.num_key_value_heads,
            "head_dim": plan.target.head_dim,
            "target_hidden_size": plan.target.hidden_size,
            "num_target_layers": plan.target.num_hidden_layers,
            "target_layer_ids": layer_ids,
            "dspark_target_layer_ids": layer_ids,
            "eagle_aux_hidden_state_layer_ids": [item + 1 for item in layer_ids],
            "use_aux_hidden_state": True,
            "markov_head_type": "vanilla",
            "sample_from_anchor": True,
            "dspark_bonus_anchor": False,
            "enable_confidence_head": False,
            "confidence_head_with_markov": False,
            "vocab_size": plan.draft_input_vocab_size,
            "draft_vocab_size": plan.draft_output_vocab_size,
            "mask_token_id": plan.mask_token_id,
            "dspark_noise_token_id": plan.mask_token_id,
        }
    )
    dflash_config = config.get("dflash_config")
    if not isinstance(dflash_config, dict):
        dflash_config = {}
    dflash_config.update(
        {
            "mask_token_id": plan.mask_token_id,
            "target_layer_ids": layer_ids,
            "use_aux_hidden_state": True,
        }
    )
    config["dflash_config"] = dflash_config
    if plan.target.max_position_embeddings is not None:
        config["max_position_embeddings"] = max(
            int(config.get("max_position_embeddings") or 0),
            plan.target.max_position_embeddings,
        )
    return config


def _resize_rows(tensor: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(
            f"Cannot resize tensor with shape {tuple(tensor.shape)} to "
            f"[{rows}, {columns}]."
        )
    output = torch.zeros((rows, columns), dtype=tensor.dtype)
    copied_rows = min(rows, tensor.shape[0])
    copied_columns = min(columns, tensor.shape[1])
    output[:copied_rows, :copied_columns].copy_(tensor[:copied_rows, :copied_columns])
    return output


def _resize_vector(
    tensor: torch.Tensor, size: int, *, fill_value: float = 0.0
) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(
            f"Cannot resize tensor with shape {tuple(tensor.shape)} to [{size}]."
        )
    output = torch.full((size,), fill_value, dtype=tensor.dtype)
    copied_size = min(size, tensor.shape[0])
    output[:copied_size].copy_(tensor[:copied_size])
    return output


def _resize_fc(tensor: torch.Tensor, plan: ConversionPlan) -> torch.Tensor:
    expected = (
        plan.source_draft_hidden_size,
        plan.source_feature_count * plan.source_target_hidden_size,
    )
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Expected fc.weight shape {expected}; got {tuple(tensor.shape)}."
        )
    source = tensor.reshape(
        plan.source_draft_hidden_size,
        plan.source_feature_count,
        plan.source_target_hidden_size,
    )
    output = torch.zeros(
        (
            plan.target.hidden_size,
            len(plan.target_layer_ids),
            plan.target.hidden_size,
        ),
        dtype=tensor.dtype,
    )
    feature_count = min(plan.source_feature_count, len(plan.target_layer_ids))
    hidden_size = min(plan.source_target_hidden_size, plan.target.hidden_size)
    output_hidden_size = min(plan.source_draft_hidden_size, plan.target.hidden_size)
    output[:output_hidden_size, :feature_count, :hidden_size].copy_(
        source[:output_hidden_size, :feature_count, :hidden_size]
    )
    return output.reshape(plan.target.hidden_size, -1)


def _attention_target_shape(key: str, plan: ConversionPlan) -> tuple[int, ...]:
    q_size = plan.target.num_attention_heads * plan.target.head_dim
    kv_size = plan.target.num_key_value_heads * plan.target.head_dim
    if key.endswith("self_attn.q_proj.weight"):
        return (q_size, plan.target.hidden_size)
    if key.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        return (kv_size, plan.target.hidden_size)
    if key.endswith("self_attn.o_proj.weight"):
        return (plan.target.hidden_size, q_size)
    if key.endswith("self_attn.q_proj.bias"):
        return (q_size,)
    if key.endswith(("self_attn.k_proj.bias", "self_attn.v_proj.bias")):
        return (kv_size,)
    if key.endswith("self_attn.o_proj.bias"):
        return (plan.target.hidden_size,)
    raise ValueError(f"Unknown attention projection tensor: {key}.")


def _resize_attention_tensor(
    key: str, tensor: torch.Tensor, plan: ConversionPlan
) -> torch.Tensor:
    shape = _attention_target_shape(key, plan)
    if len(shape) == 2:
        return _resize_rows(tensor, shape[0], shape[1])
    return _resize_vector(tensor, shape[0])


def _resize_backbone_tensor(
    key: str, tensor: torch.Tensor, plan: ConversionPlan
) -> torch.Tensor:
    hidden_size = plan.target.hidden_size
    intermediate_size = plan.source_intermediate_size
    if key.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return _resize_rows(tensor, intermediate_size, hidden_size)
    if key.endswith("mlp.down_proj.weight"):
        return _resize_rows(tensor, hidden_size, intermediate_size)
    if key.endswith(("mlp.gate_proj.bias", "mlp.up_proj.bias")):
        return _resize_vector(tensor, intermediate_size)
    if key.endswith("mlp.down_proj.bias"):
        return _resize_vector(tensor, hidden_size)
    if key.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
        return _resize_vector(tensor, plan.target.head_dim, fill_value=1.0)
    if key.endswith(("self_attn.q_norm.bias", "self_attn.k_norm.bias")):
        return _resize_vector(tensor, plan.target.head_dim)
    if key.endswith(
        (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "hidden_norm.weight",
            "verifier_norm.weight",
        )
    ) or key in {"norm.weight", "model.norm.weight"}:
        return _resize_vector(tensor, hidden_size, fill_value=1.0)
    if key.endswith("rotary_emb.inv_freq"):
        return _resize_vector(tensor, plan.target.head_dim // 2)
    return tensor.clone()


def _backbone_target_shape(key: str, plan: ConversionPlan) -> tuple[int, ...] | None:
    hidden_size = plan.target.hidden_size
    intermediate_size = plan.source_intermediate_size
    if key.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return (intermediate_size, hidden_size)
    if key.endswith("mlp.down_proj.weight"):
        return (hidden_size, intermediate_size)
    if key.endswith(("mlp.gate_proj.bias", "mlp.up_proj.bias")):
        return (intermediate_size,)
    if key.endswith("mlp.down_proj.bias"):
        return (hidden_size,)
    if key.endswith(
        (
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_norm.bias",
            "self_attn.k_norm.bias",
        )
    ):
        return (plan.target.head_dim,)
    if key.endswith(
        (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "hidden_norm.weight",
            "verifier_norm.weight",
        )
    ) or key in {"norm.weight", "model.norm.weight"}:
        return (hidden_size,)
    if key.endswith("rotary_emb.inv_freq"):
        return (plan.target.head_dim // 2,)
    return None


def _iter_source_tensors(
    files: list[Path], plan: ConversionPlan
) -> Iterator[tuple[str, torch.Tensor]]:
    skipped = {key for key in (plan.d2t_key,) if key is not None}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                leaf = key.split(".")[-1]
                if key in skipped or leaf in {"t2d", "target_id_to_draft_id"}:
                    continue
                if "confidence_head" in key or "mask_embedding" in key:
                    continue
                tensor = handle.get_tensor(key)
                if key == plan.fc_key:
                    tensor = _resize_fc(tensor, plan)
                elif key == plan.embed_key:
                    tensor = _resize_rows(
                        tensor,
                        plan.draft_input_vocab_size,
                        plan.target.hidden_size,
                    )
                elif key == plan.lm_head_key:
                    tensor = _resize_rows(
                        tensor,
                        plan.draft_output_vocab_size,
                        plan.target.hidden_size,
                    )
                elif key == plan.markov_w1_key:
                    tensor = _resize_rows(
                        tensor,
                        plan.draft_input_vocab_size,
                        plan.markov_rank,
                    )
                elif key == plan.markov_w2_key:
                    tensor = _resize_rows(
                        tensor,
                        plan.draft_output_vocab_size,
                        plan.markov_rank,
                    )
                elif key in plan.attention_projection_keys:
                    tensor = _resize_attention_tensor(key, tensor, plan)
                else:
                    tensor = _resize_backbone_tensor(key, tensor, plan)
                yield key, tensor.contiguous()


def _iter_converted_tensors(
    files: list[Path], plan: ConversionPlan, dtype: torch.dtype
) -> Iterator[tuple[str, torch.Tensor]]:
    yield from _iter_source_tensors(files, plan)
    if plan.embed_key is None:
        yield (
            "embed_tokens.weight",
            torch.zeros(
                (plan.draft_input_vocab_size, plan.target.hidden_size), dtype=dtype
            ),
        )
    if plan.lm_head_key is None:
        yield (
            "lm_head.weight",
            torch.zeros(
                (plan.draft_output_vocab_size, plan.target.hidden_size), dtype=dtype
            ),
        )
    if plan.draft_output_vocab_size != plan.target.vocab_size:
        yield "d2t", torch.zeros(plan.draft_output_vocab_size, dtype=torch.int64)


class _ShardWriter:
    def __init__(self, directory: Path, max_shard_size: int) -> None:
        self.directory = directory
        self.max_shard_size = max_shard_size
        self.tensors: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self.total_size = 0
        self.parts: list[tuple[Path, list[str]]] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.tensors or any(key in keys for _, keys in self.parts):
            raise ValueError(f"Duplicate output tensor {key!r}.")
        size = tensor.numel() * tensor.element_size()
        if self.tensors and self.current_size + size > self.max_shard_size:
            self._flush()
        self.tensors[key] = tensor
        self.current_size += size
        self.total_size += size

    def _flush(self) -> None:
        if not self.tensors:
            return
        path = self.directory / f"part-{len(self.parts) + 1:05d}.safetensors"
        keys = list(self.tensors)
        save_file(self.tensors, path, metadata={"format": "pt"})
        self.parts.append((path, keys))
        self.tensors = {}
        self.current_size = 0

    def finish(self) -> None:
        self._flush()
        if not self.parts:
            raise ValueError("No output tensors were produced.")
        if len(self.parts) == 1:
            self.parts[0][0].rename(self.directory / "model.safetensors")
            return

        weight_map: dict[str, str] = {}
        count = len(self.parts)
        for index, (old_path, keys) in enumerate(self.parts, start=1):
            filename = f"model-{index:05d}-of-{count:05d}.safetensors"
            old_path.rename(self.directory / filename)
            weight_map.update(dict.fromkeys(keys, filename))
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": weight_map,
        }
        _write_json(self.directory / "model.safetensors.index.json", index)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _checkpoint_dtype(files: list[Path], plan: ConversionPlan) -> torch.dtype:
    candidates = {plan.fc_key, plan.markov_w1_key, plan.markov_w2_key}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for candidate in candidates.intersection(handle.keys()):
                dtype = handle.get_tensor(candidate).dtype
                if not dtype.is_floating_point:
                    raise ValueError(f"Expected floating-point weights, got {dtype}.")
                return dtype
    raise ValueError("Cannot determine checkpoint dtype.")


def _validate_output_checkpoint(directory: Path, plan: ConversionPlan) -> None:
    tensors = _inspect_tensors(_weight_files(directory))
    expected_shapes = {
        plan.fc_key: (
            plan.target.hidden_size,
            len(plan.target_layer_ids) * plan.target.hidden_size,
        ),
        plan.markov_w1_key: (plan.draft_input_vocab_size, plan.markov_rank),
        plan.markov_w2_key: (plan.draft_output_vocab_size, plan.markov_rank),
    }
    embed_key = plan.embed_key or "embed_tokens.weight"
    lm_head_key = plan.lm_head_key or "lm_head.weight"
    expected_shapes[embed_key] = (
        plan.draft_input_vocab_size,
        plan.target.hidden_size,
    )
    expected_shapes[lm_head_key] = (
        plan.draft_output_vocab_size,
        plan.target.hidden_size,
    )
    for key in plan.attention_projection_keys:
        expected_shapes[key] = _attention_target_shape(key, plan)
    for key in tensors:
        if (shape := _backbone_target_shape(key, plan)) is not None:
            expected_shapes[key] = shape
    for key, expected in expected_shapes.items():
        actual = tensors.get(key)
        if actual is None:
            raise ValueError(f"Converted checkpoint is missing {key!r}.")
        if actual.shape != expected:
            raise ValueError(
                f"Converted tensor {key!r} has shape {actual.shape}; "
                f"expected {expected}."
            )

    mapping_keys = [
        key for key in tensors if key.split(".")[-1] in {"d2t", "draft_id_to_target_id"}
    ]
    expects_mapping = plan.draft_output_vocab_size != plan.target.vocab_size
    if expects_mapping and mapping_keys != ["d2t"]:
        raise ValueError(
            "Converted reduced-vocabulary checkpoint must contain exactly one d2t."
        )
    if not expects_mapping and mapping_keys:
        raise ValueError("Converted full-vocabulary checkpoint must not contain d2t.")


def convert_checkpoint(
    *,
    draft_model: str,
    target_model: str,
    output_dir: Path,
    target_layer_ids: tuple[int, ...] | None = None,
    draft_revision: str | None = None,
    target_revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    max_shard_size_gb: float = 1.5,
) -> dict[str, Any]:
    """Create a shape-compatible Qwen3-Omni DSpark smoke checkpoint."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new path."
        )
    if max_shard_size_gb <= 0:
        raise ValueError("max_shard_size_gb must be positive.")

    draft_directory = _resolve_draft_directory(
        draft_model,
        revision=draft_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    draft_config, _ = _resolve_config(
        str(draft_directory),
        revision=None,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    target_config, _ = _resolve_config(
        target_model,
        revision=target_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    target = _extract_target_text_config(target_config)
    files = _weight_files(draft_directory)
    tensors = _inspect_tensors(files)
    plan = _make_plan(draft_config, target, tensors, target_layer_ids)
    converted_config = _converted_config(draft_config, plan)
    dtype = _checkpoint_dtype(files, plan)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "config.json", converted_config)
        writer = _ShardWriter(
            temporary, max_shard_size=int(max_shard_size_gb * 1024**3)
        )
        for key, tensor in _iter_converted_tensors(files, plan, dtype):
            writer.add(key, tensor)
        writer.finish()
        _validate_output_checkpoint(temporary, plan)
        manifest = {
            "purpose": "Qwen3-Omni DSpark framework smoke test only",
            "architecture": "Qwen3OmniDSparkModel",
            "source_draft_model": draft_model,
            "target_model": target_model,
            "source_target_hidden_size": plan.source_target_hidden_size,
            "source_draft_hidden_size": plan.source_draft_hidden_size,
            "target_hidden_size": target.hidden_size,
            "target_num_hidden_layers": target.num_hidden_layers,
            "target_layer_ids": list(plan.target_layer_ids),
            "source_attention_geometry": {
                "num_attention_heads": plan.source_num_attention_heads,
                "num_key_value_heads": plan.source_num_key_value_heads,
                "head_dim": plan.source_head_dim,
            },
            "target_attention_geometry": {
                "num_attention_heads": target.num_attention_heads,
                "num_key_value_heads": target.num_key_value_heads,
                "head_dim": target.head_dim,
            },
            "tensor_conversion": "deterministic overlap copy with zero padding",
            "draft_input_vocab_size": plan.draft_input_vocab_size,
            "draft_output_vocab_size": plan.draft_output_vocab_size,
            "target_vocab_size": target.vocab_size,
            "mask_token_id": plan.mask_token_id,
            "generated_zero_embedding": plan.embed_key is None,
            "generated_zero_lm_head": plan.lm_head_key is None,
            "token_mapping": "full target vocabulary; no d2t remap",
            "accuracy_expected": False,
        }
        _write_json(temporary / "conversion_manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parse_layer_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--target-layer-ids must be comma-separated integers."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an unquantized Qwen3 DSpark checkpoint into a deterministic, "
            "shape-compatible Qwen3-Omni framework smoke-test checkpoint."
        )
    )
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target-layer-ids",
        type=_parse_layer_ids,
        help="Optional zero-based Omni text-layer IDs, for example 1,12,23,34,45.",
    )
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-shard-size-gb", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_checkpoint(
        draft_model=args.draft_model,
        target_model=args.target_model,
        output_dir=args.output_dir,
        target_layer_ids=args.target_layer_ids,
        draft_revision=args.draft_revision,
        target_revision=args.target_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        max_shard_size_gb=args.max_shard_size_gb,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
