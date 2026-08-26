#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one Qwen3-VL multimodal request with DSpARK enabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
if (REPO_ROOT / "vllm").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.multimodal.utils import fetch_image  # noqa: E402

DEFAULT_DRAFT_MODEL = EXAMPLE_DIR / "artifacts" / "Qwen3-VL-DSpARK-config-fixture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--image-url",
        default=(
            "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
            "multimodal_asset/duck.jpg"
        ),
    )
    parser.add_argument("--question", default="Describe this image briefly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.draft_model / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"DSpARK checkpoint not found at {args.draft_model}; "
            "run build_dspark_checkpoint.py first"
        )

    speculative_config = {
        "method": "dspark",
        "model": str(args.draft_model.resolve()),
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        "draft_sample_method": "greedy",
    }
    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        speculative_config=speculative_config,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_chunked_mm_input=True,
        disable_log_stats=False,
    )

    image = fetch_image(args.image_url)
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholder}{args.question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    print(outputs[0].outputs[0].text)

    for metric in llm.get_metrics():
        if metric.name.startswith("vllm:spec_decode_"):
            print(f"{metric.name}={metric.value}")


if __name__ == "__main__":
    main()
