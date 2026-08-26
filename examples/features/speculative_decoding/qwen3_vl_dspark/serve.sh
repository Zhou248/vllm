#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_model="${TARGET_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
draft_model="${DRAFT_MODEL:-${script_dir}/artifacts/Qwen3-VL-DSpARK-config-fixture}"
target_tp="${TARGET_TP:-2}"
draft_tp="${DRAFT_TP:-1}"
block_size="${NUM_SPECULATIVE_TOKENS:-16}"
max_model_len="${MAX_MODEL_LEN:-4096}"

exec vllm serve "${target_model}" \
  --tensor-parallel-size "${target_tp}" \
  --max-model-len "${max_model_len}" \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"${draft_model}\",\"num_speculative_tokens\":${block_size},\"draft_tensor_parallel_size\":${draft_tp},\"draft_sample_method\":\"greedy\"}"
