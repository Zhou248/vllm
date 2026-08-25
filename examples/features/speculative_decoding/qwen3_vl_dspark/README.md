# Qwen3-VL-30B-A3B-Instruct DSpARK 联调样例

该目录完全自包含，不依赖 `msModelSpec`。它负责构造一个未训练的 DSpARK
checkpoint，并通过 vLLM 或 vLLM-Ascend 执行 Qwen3-VL 多模态任务。

`target_config.json` 是官方 `Qwen/Qwen3-VL-30B-A3B-Instruct` 配置快照。
其文本模型为 48 层、hidden size 2048、32 个 attention head、4 个 KV head、
head dimension 128，以及 128 个路由专家。生成的 draft 是一层 dense Qwen3
decoder；target 仍然使用 MoE，并负责运行视觉塔。

默认 DSpARK 契约如下：

- `architectures`：`Qwen3VLDSparkModel`
- block size：4
- target 文本层号（从 0 开始）：`[11, 23, 35, 47]`
- vLLM aux hidden-state 层号（从 1 开始）：`[12, 24, 36, 48]`
- Markov rank：8
- draft 使用逻辑 1-D RoPE，不复制 target MRoPE
- target embedding 和 LM head 在运行时共享

## 构造 checkpoint

在 vLLM 仓库根目录执行：

```bash
.venv/bin/python \
  examples/features/speculative_decoding/qwen3_vl_dspark/build_dspark_checkpoint.py
```

输出目录：

```text
examples/features/speculative_decoding/qwen3_vl_dspark/artifacts/
  Qwen3-VL-30B-A3B-Instruct-DSpARK-fixture/
```

如需覆盖已生成的 fixture，请显式传入 `--force`。训练部门提供真实配置后，
可以通过 `--target-config`、`--target-layer-ids`、`--block-size`、
`--num-draft-layers` 和 `--markov-rank` 替换临时契约。

## 执行离线多模态任务

```bash
.venv/bin/python \
  examples/features/speculative_decoding/qwen3_vl_dspark/run_offline.py \
  --tensor-parallel-size 4
```

该脚本会下载一张小型公开测试图片，执行一次多模态请求，并输出生成文本和
DSpARK 指标。请根据实际硬件调整 TP 和显存参数。

## 启动 API 服务

```bash
bash examples/features/speculative_decoding/qwen3_vl_dspark/serve.sh
```

可以通过 `TARGET_TP`、`DRAFT_TP`、`MAX_MODEL_LEN`、`TARGET_MODEL` 和
`DRAFT_MODEL` 环境变量覆盖默认值。vLLM-Ascend 环境可以使用相同脚本和
checkpoint。

生成的 tensor 是确定性的结构权重，只用于验证配置解析、模型注册、权重加载和
forward 链路，不能用于评估生成质量或 acceptance rate。
