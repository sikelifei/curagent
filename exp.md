# 新 prompt 全量实验结果

本次结果全部重新运行于当前工作区的新 prompt，原目录中的旧结果不计入。原始结果统一位于 `outputs/expmd_newprompt/`。

## 实验配置

- 本地模型：`/data2/zhangwenjian/model/Qwen3-4B-Instruct-2507`。
- API 模型：`deepseek-v4-flash`。
- WebShop：test split，200 条。
- Oolong-Synth：8K、16K、32K、64K、128K、256K、512K、1M、2M、4M；分别 50、50、50、50、50、50、20、20、20、20 条。
- TextCraft-Synth：easy、medium、hard 各 50 条；无官方本地数据时使用项目既定 generated fallback。
- BrowseComp-Plus：前 50 条 test query，BM25 本地服务，50 条。

## WebShop

| 模型 | 记录 | 完成/失败 | 成功 | 成功率 | 平均 reward | model calls | Agent steps | 递归 episode | spawn calls | child agents | 最大深度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 200 | 200/0 | 0 | 0.0% | 0.0000 | 3315 | 3133 | 0 | 0 | 0 | 0 |
| DeepSeek v4 flash | 200 | 199/1 | 54 | 27.14% | 0.4246 | 4623 | 4556 | 0 | 0 | 0 | 0 |

Qwen 原始结果：`outputs/expmd_newprompt/webshop/qwen3_4b_instruct_2507_valid/`。该批次前 58 条使用 35 agent steps，停止调度器后以 `--resume`、10 agent steps 完成剩余条目；所有 200 条均已完成并纳入同一 summary。DeepSeek 原始结果：`outputs/expmd_newprompt/webshop/deepseek_v4_flash/`。

## Oolong-Synth

`score` 为 Oolong 评测分数，`elapsed` 为该桶所有 episode elapsed_seconds 之和。所有桶记录完整，递归触发均为 0，最大深度均为 0。

| 桶 | 条数 | score | score % | elapsed (s) |
|---:|---:|---:|---:|---:|
| 8K | 50 | 0.3588 | 35.88% | 307.8 |
| 16K | 50 | 0.3973 | 39.73% | 351.8 |
| 32K | 50 | 0.4027 | 40.27% | 344.9 |
| 64K | 50 | 0.3866 | 38.66% | 862.4 |
| 128K | 50 | 0.2744 | 27.44% | 1816.7 |
| 256K | 50 | 0.1856 | 18.56% | 922.0 |
| 512K | 20 | 0.2211 | 22.11% | 328.4 |
| 1M | 20 | 0.3500 | 35.00% | 336.6 |
| 2M | 20 | 0.3000 | 30.00% | 367.9 |
| 4M | 20 | 0.3711 | 37.11% | 349.5 |

原始结果：`outputs/expmd_newprompt/oolong/qwen3_4b_instruct_2507/<bucket>/`。

## TextCraft-Synth

| 难度 | 条数 | runner ok | task success | success rate | 平均 score | 总耗时 (s) | 递归 children | 最大深度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 50 | 49 | 37 | 74.0% | 0.7400 | 1991.4 | 1 | 1 |
| medium | 50 | 31 | 0 | 0.0% | 0.0000 | 5832.9 | 0 | 0 |
| hard | 50 | 42 | 0 | 0.0% | 0.0000 | 4207.6 | 5 | 1 |

medium/hard 使用 10 个 10 条分片并行运行，超时和 runner failure 保留在分片结果并计入 50 条分母。easy 原始结果在 `outputs/expmd_newprompt/textcraft/deepseek_v4_flash/easy/`；分片结果在 `outputs/expmd_newprompt/textcraft/deepseek_v4_flash_shards/{medium,hard}/`。

## BrowseComp-Plus

50/50 条记录完成，其中 runner completed 19、failed 31；本地 judge 正确数为 0/19。平均搜索调用 12.32，9/50 条触发递归，平均子 Agent 数 1.32，最大递归深度 2。总耗时 2000.0 秒。原始结果：`outputs/expmd_newprompt/browsecomp_plus/deepseek_v4_flash/`。

## 验证

- WebShop：两个 summary 均 `recorded_rows=200`。
- Oolong：十个 summary 均达到目标条数，合计 420 条。
- TextCraft：easy、medium、hard 均为 50 条，合计 150 条。
- BrowseComp-Plus：summary `total_questions=50`、`recorded=50`。

## WebShop DeepSeek v4 flash 重跑（2026-07-23）

使用当前 prompt、test split 的 200 条（`instance_id=0..199`）、
`agent-max-steps=35`、`env-max-steps=30`、并发 8。200/200 条完成，
episode-level failure 为 0；成功 74/200（37.0%），平均 reward 0.5930，
总 model calls 3988，agent steps 3965。

递归统计：22/200 个 episode 发生递归，调用 `spawn_subagents` 共 33 次，
产生 106 个 child agent，最大递归深度为 2。递归组成功 5/22（22.7%），
平均 reward 0.5614；非递归组成功 69/178（38.8%），平均 reward 0.5969。

原始结果：`outputs/webshop_200_newprompt_20260723/deepseek_v4_flash_retry_traces.jsonl`
和 `outputs/webshop_200_newprompt_20260723/deepseek_v4_flash_retry_summary.json`。
Qwen3-4B-Instruct-2507 本次未能启动，因 8 张 GPU 均被已有 Ray/vLLM 进程占用。
