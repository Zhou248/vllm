# Qwen3-VL DSpARK 联调样例

该目录完全自包含，不依赖 `msModelSpec`。它根据训练侧真实
`qwen3_vl_dflash` 配置构造一个未训练的 DSpARK checkpoint，并通过
vLLM 或 vLLM-Ascend 执行 Qwen3-VL 多模态任务。

当前仓库根目录的 `config.json` 对应 `Qwen/Qwen3-VL-2B-Instruct`
的 Dense 文本模型，关键契约为：

- target text：28 层，hidden size 2048，16 个 attention head，8 个 KV head
- draft：2 层 dense Qwen3 decoder
- block size：16
- target 文本层号（从 0 开始）：`[1, 7, 13, 19, 25]`
- vLLM aux hidden-state 层号（从 1 开始）：`[2, 8, 14, 20, 26]`
- Markov rank：256
- mask token id：151669
- 完整词表：151936，target embedding 和 LM head 在运行时共享
- target 保持 M-RoPE；draft 自动使用等价的逻辑 1-D RoPE

vLLM 会将训练侧的 `Qwen3VLForConditionalGenerationDFlash` 和嵌套
`text_config`/`dflash_config` 自动规范化为独立的
`Qwen3VLDSparkModel`，不需要 checkpoint 携带或执行自定义 Python 代码。

## 构造 checkpoint

在 vLLM 仓库根目录执行：

```bash
.venv/bin/python \
  examples/features/speculative_decoding/qwen3_vl_dspark/build_dspark_checkpoint.py \
  --source-config config.json
```

输出目录：

```text
examples/features/speculative_decoding/qwen3_vl_dspark/artifacts/
  Qwen3-VL-DSpARK-config-fixture/
```

脚本会保留原始 `config.json` 结构，并从它推导所有 tensor shape。
如需覆盖已生成的 fixture，显式传入 `--force`。输出的
`fixture_manifest.json` 记录配置哈希、推导后的契约、tensor shape 和权重哈希。

## 执行离线多模态任务

```bash
.venv/bin/python \
  examples/features/speculative_decoding/qwen3_vl_dspark/run_offline.py \
  --target-model Qwen/Qwen3-VL-2B-Instruct \
  --tensor-parallel-size 2
```

target 必须与 draft 配置中的 target 文本几何一致。对其他 Dense 或 MoE
Qwen3-VL 变体，需使用对应训练得到的 DSpARK config 和权重，不能直接
混用本 fixture。

## 启动 API 服务

```bash
bash examples/features/speculative_decoding/qwen3_vl_dspark/serve.sh
```

可以通过 `TARGET_TP`、`DRAFT_TP`、`MAX_MODEL_LEN`、`TARGET_MODEL` 和
`DRAFT_MODEL` 环境变量覆盖默认值。vLLM-Ascend 环境可以使用相同脚本和
checkpoint。

生成的 tensor 是确定性结构权重，只用于验证配置解析、模型注册、权重加载和
forward 链路，不能用于评估生成质量或 acceptance rate。配置中启用的
confidence head 属于训练/动态决策路径，当前 fixture 和 vLLM 推理路径不使用它。
