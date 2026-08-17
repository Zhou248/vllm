# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.convert_qwen3_dspark_to_qwen3_omni import convert_checkpoint

pytestmark = pytest.mark.skip_global_cleanup


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_configs(draft_dir: Path, target_dir: Path) -> None:
    _write_json(
        draft_dir / "config.json",
        {
            "architectures": ["Qwen3DSparkModel"],
            "model_type": "qwen3",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 6,
            "markov_rank": 2,
            "mask_token_id": 5,
            "block_size": 3,
            "target_layer_ids": [0, 2],
            "max_position_embeddings": 16,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
        },
    )
    _write_json(
        target_dir / "config.json",
        {
            "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 3,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "vocab_size": 8,
                    "max_position_embeddings": 32,
                }
            },
        },
    )


def _base_weights(include_shared: bool = True) -> dict[str, torch.Tensor]:
    weights = {
        "fc.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        "layers.0.self_attn.q_proj.weight": torch.eye(4),
        "layers.0.self_attn.k_proj.weight": torch.arange(
            16, dtype=torch.float32
        ).reshape(4, 4),
        "layers.0.self_attn.v_proj.weight": torch.arange(
            16, 32, dtype=torch.float32
        ).reshape(4, 4),
        "layers.0.self_attn.o_proj.weight": torch.eye(4),
        "layers.0.self_attn.q_norm.weight": torch.ones(2),
        "layers.0.self_attn.k_norm.weight": torch.ones(2),
        "layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
        "layers.0.mlp.up_proj.weight": torch.ones(8, 4),
        "layers.0.mlp.down_proj.weight": torch.ones(4, 8),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "norm.weight": torch.ones(4),
        "markov_head.markov_w1.weight": torch.arange(12, dtype=torch.float32).reshape(
            6, 2
        ),
        "markov_head.markov_w2.weight": torch.arange(12, dtype=torch.float32).reshape(
            6, 2
        ),
    }
    if include_shared:
        weights["embed_tokens.weight"] = torch.arange(24, dtype=torch.float32).reshape(
            6, 4
        )
        weights["lm_head.weight"] = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    return weights


def _load_weights(directory: Path) -> dict[str, torch.Tensor]:
    output = {}
    files = [directory / "model.safetensors"]
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        files = [directory / name for name in sorted(set(index["weight_map"].values()))]
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                output[key] = handle.get_tensor(key)
    return output


def test_converts_hidden_features_and_vocab_for_omni(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    output_dir = tmp_path / "converted"
    draft_dir.mkdir()
    target_dir.mkdir()
    _make_configs(draft_dir, target_dir)
    source = _base_weights()
    save_file(source, draft_dir / "model.safetensors")

    manifest = convert_checkpoint(
        draft_model=str(draft_dir),
        target_model=str(target_dir),
        output_dir=output_dir,
        max_shard_size_gb=1,
    )

    config = json.loads((output_dir / "config.json").read_text())
    assert config["architectures"] == ["Qwen3OmniDSparkModel"]
    assert config["hidden_size"] == 3
    assert config["num_attention_heads"] == 2
    assert config["num_key_value_heads"] == 1
    assert config["head_dim"] == 2
    assert config["target_hidden_size"] == 3
    assert config["num_target_layers"] == 4
    assert config["target_layer_ids"] == [0, 2]
    assert config["eagle_aux_hidden_state_layer_ids"] == [1, 3]
    assert config["vocab_size"] == 8
    assert config["draft_vocab_size"] == 8
    assert config["mask_token_id"] == 5
    assert config["dflash_config"]["mask_token_id"] == 5
    assert config["dflash_config"]["use_aux_hidden_state"] is True
    assert config["markov_head_type"] == "vanilla"
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False

    weights = _load_weights(output_dir)
    expected_fc = source["fc.weight"].reshape(4, 2, 4)[:3, :, :3].reshape(3, 6)
    torch.testing.assert_close(weights["fc.weight"], expected_fc)
    torch.testing.assert_close(
        weights["embed_tokens.weight"][:6], source["embed_tokens.weight"][:, :3]
    )
    assert torch.count_nonzero(weights["embed_tokens.weight"][6:]) == 0
    assert weights["markov_head.markov_w1.weight"].shape == (8, 2)
    assert weights["lm_head.weight"].shape == (8, 3)
    assert "d2t" not in weights
    assert weights["layers.0.self_attn.q_proj.weight"].shape == (4, 3)
    torch.testing.assert_close(
        weights["layers.0.self_attn.k_proj.weight"],
        source["layers.0.self_attn.k_proj.weight"][:2, :3],
    )
    torch.testing.assert_close(
        weights["layers.0.self_attn.v_proj.weight"],
        source["layers.0.self_attn.v_proj.weight"][:2, :3],
    )
    assert weights["layers.0.self_attn.o_proj.weight"].shape == (3, 4)
    assert weights["layers.0.mlp.gate_proj.weight"].shape == (8, 3)
    assert weights["layers.0.mlp.down_proj.weight"].shape == (3, 8)
    assert weights["layers.0.input_layernorm.weight"].shape == (3,)
    assert manifest["source_attention_geometry"]["num_key_value_heads"] == 2
    assert manifest["target_attention_geometry"]["num_key_value_heads"] == 1
    assert manifest["accuracy_expected"] is False


def test_generates_incompatible_shared_weights_for_smoke_test(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    output_dir = tmp_path / "converted"
    draft_dir.mkdir()
    target_dir.mkdir()
    _make_configs(draft_dir, target_dir)
    save_file(_base_weights(include_shared=False), draft_dir / "model.safetensors")

    manifest = convert_checkpoint(
        draft_model=str(draft_dir),
        target_model=str(target_dir),
        output_dir=output_dir,
        max_shard_size_gb=1,
    )

    weights = _load_weights(output_dir)
    assert weights["embed_tokens.weight"].shape == (8, 3)
    assert weights["lm_head.weight"].shape == (8, 3)
    assert torch.count_nonzero(weights["embed_tokens.weight"]) == 0
    assert torch.count_nonzero(weights["lm_head.weight"]) == 0
    assert manifest["generated_zero_embedding"] is True
    assert manifest["generated_zero_lm_head"] is True


def test_refuses_mrope_draft_checkpoint(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    output_dir = tmp_path / "converted"
    draft_dir.mkdir()
    target_dir.mkdir()
    _make_configs(draft_dir, target_dir)
    config = json.loads((draft_dir / "config.json").read_text())
    config["rope_scaling"] = {"mrope_section": [1, 1, 2]}
    _write_json(draft_dir / "config.json", config)
    save_file(_base_weights(), draft_dir / "model.safetensors")

    with pytest.raises(ValueError, match="logical 1-D RoPE"):
        convert_checkpoint(
            draft_model=str(draft_dir),
            target_model=str(target_dir),
            output_dir=output_dir,
        )


def test_converts_incompatible_query_head_geometry(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    output_dir = tmp_path / "converted"
    draft_dir.mkdir()
    target_dir.mkdir()
    _make_configs(draft_dir, target_dir)
    target_config = json.loads((target_dir / "config.json").read_text())
    target_config["thinker_config"]["text_config"]["num_attention_heads"] = 1
    _write_json(target_dir / "config.json", target_config)
    save_file(_base_weights(), draft_dir / "model.safetensors")

    convert_checkpoint(
        draft_model=str(draft_dir),
        target_model=str(target_dir),
        output_dir=output_dir,
    )

    weights = _load_weights(output_dir)
    assert weights["layers.0.self_attn.q_proj.weight"].shape == (2, 3)
    assert weights["layers.0.self_attn.o_proj.weight"].shape == (3, 2)


def test_writes_loadable_sharded_checkpoint_index(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    output_dir = tmp_path / "converted"
    draft_dir.mkdir()
    target_dir.mkdir()
    _make_configs(draft_dir, target_dir)
    save_file(_base_weights(), draft_dir / "model.safetensors")

    convert_checkpoint(
        draft_model=str(draft_dir),
        target_model=str(target_dir),
        output_dir=output_dir,
        max_shard_size_gb=1e-8,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] > 0
    assert index["weight_map"]["fc.weight"].startswith("model-")
    weights = _load_weights(output_dir)
    assert weights["fc.weight"].shape == (3, 6)
    assert "d2t" not in weights
