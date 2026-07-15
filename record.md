 python -u -m examples.run_webshop_batch --config configs/model_api.local.yaml
  │ --recode-root /data2/zhangwenjian/agent/ReCode --split test --start-index 0 --count 200 --concurrency 8 --seed 233 --env-max-steps 30 --agent-max-steps 35 --max-depth 2
  │ --max-concurrent-subagents 4 --max-run-seconds 900 --trace-jsonl outputs/webshop_200_prompt_v3_traces.jsonl --summary-json outputs/webshop_200_prompt_v3_summary.json
 

## Qwen3-4B prompt fix 20 条验证

使用 `configs/model_api_vllm_fast.yaml`，Qwen3-4B vLLM（GPU2，
`max-num-seqs=16`、`max-num-batched-tokens=16384`、prefix caching），
`count=20`、`concurrency=8`。运行耗时 163.2 秒（约 7.36 条/分钟）。

- 20/20 完成，success=2/20，success_rate=10.0%
- average_reward=0.2992
- model_calls=1181，agent_steps=1157，REPL blocks=556
- episodes_with_variables=20，episodes_with_tool_calls=20
- spawn episode=4，spawn calls=82，child agents=72，最大深度=2
- execution_errors=116，code_parse_errors=37
- `search[keywords]` 字面量执行次数：0；实际搜索 action 均包含具体查询词
- 9/20 为 `forced_final`；有 1 条轨迹在重复 search/back/next 中耗尽 30 步

结论：action 模板修正有效，平均 reward 从旧版 0.0323 提升到 0.2992，
成功率从 1% 提升到 10%，但购买完成率仍低，重复导航和 REPL 解析错误仍是
主要瓶颈。暂不启动 Qwen2.5 和 Qwen3-4B-Instruct；下一步应继续收紧
“一次搜索后候选筛选/购买”的流程约束，并降低无效委派与代码格式错误。

结果文件：`outputs/webshop_20_prompt_fix_qwen34b_traces.jsonl`、
`outputs/webshop_20_prompt_fix_qwen34b_summary.json`
 
 200 条真实 WebShop test 已完成，使用最新 prompt 版本和 8 个并发 worker。

  运行配置：

  concurrency=8
  max_concurrent_subagents=4
  test split
  instance_id=0..199

  耗时约 1596 秒，即 26 分 36 秒，平均约 7.52 条/分钟。200/200 条都正常落盘，没有 episode-level 失败。

   指标                   结果
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━
   成功数               77/200
  ──────────────────  ─────────
   成功率                38.5%
  ──────────────────  ─────────
   平均 reward           0.610
  ──────────────────  ─────────
   environment_done        185
  ──────────────────  ─────────
   forced_final             15
  ──────────────────  ─────────
   模型调用               4557
  ──────────────────  ─────────
   REPL blocks            4220
  ──────────────────  ─────────
   变量快照覆盖        200/200
  ──────────────────  ─────────
   执行错误 episode        100
  ──────────────────  ─────────
   执行错误总数            306

  工具调用：

  act               2836
  observe            976
  available_actions    3

  Subagent 行为：

  发生 spawn 的 episode：47
  spawn 调用：53
  child agents：196
  最大深度：2

  child 实际也执行了环境操作：

  child act 调用：511
  child observe 调用：198

  这符合你刚才确认的设计：child 可以操作环境，但需要通过自己的 prompt 和 parent 委派目标来约束行为。200 条中所有 root 都收到 root prompt，196 个 child 都收到独立的 subagent prompt；
  没有强制 delegation prompt，也没有 child trace 不一致。

  主要执行错误：

  invalid search：131
  invalid click：140
  NameError：9
  TypeError：3
  legacy import：2
  SyntaxError：1
  其他：20

  这些是模型 REPL/action 错误，均被记录进轨迹，没有被 harness 静默修正。



 python -u -m examples.run_webshop_batch --config configs/model_api_vllm.yaml --recode-root /data2/zhangwenjian/agent/ReCode --split test --start-index 0 --count 200 --concurrency 8 --seed 233 --env-max-steps 30 --agent-max-steps 35 --max-depth 2 --max-concurrent-subagents 4 --max-run-seconds 900 --trace-jsonl outputs/webshop_200_prompt_v3_traces_qwen34b.jsonl --summary-json outputs/webshop_200_prompt_v3_traces_qwen34b.json
 

Qwen3-4B vLLM 200 条诊断（按用户要求暂停后续模型）

实际结果文件：outputs/webshop_200_prompt_v3_traces_qwen34b.jsonl

说明：该文件是 resume 合并结果，最早 6 条来自未关闭 thinking 的启动，之后
194 条来自 `model_api_vllm_fast.yaml`；后 194 条也只有 2 条成功，因此低成功率
不是首批 thinking warmup 单独造成的。

- 200/200 episode 运行完成，episode-level failure=0
- success=2/200，success_rate=1.0%
- average_reward=0.0323
- model_calls=8557，agent_steps=8343，REPL blocks=4972
- execution_errors=582，涉及 124 个 episode
- spawn episode=8，spawn calls=186，child agents=152，最大深度=2
- environment_done=21，forced_final=178，completed=1

诊断：

1. 主要 action 解析问题。模型大量原样执行 `act('search[keywords]')`，把
   `search[keywords]` 当作真实查询，而不是把 keywords 替换成商品查询。
   结果页出现 Bible、scanner 等无关商品，随后模型在 `next/back` 之间循环。
   200 条轨迹只出现 17 次 `click[buy now]`，其中只有 2 次成功。

2. 委派在少数 episode 中递归放大。只有 8 个 episode 使用 spawn，但产生了
   186 次 spawn 调用和 152 个 child；child 又产生了 144 次 spawn_subagents
   和 34 次 spawn_subagent。instance 86、102、129 的 model_calls 分别为
   393、422、499，主要时间消耗在重复委派而不是 WebShop 导航。

3. 资源上限触发过多。178/200 进入 forced_final，说明模型经常没有在
   agent-max-steps=35 内完成购买流程。错误主要是 invalid_click=198、
   invalid_search=191，以及 child 侧的 NameError/TypeError。

4. vLLM 本身不是首要瓶颈。GPU2 在运行期间约 100% utilization，batch
   参数和请求都正常；关闭 Qwen3 thinking 后单条速度明显提升，但 action
   模式和委派行为仍然导致任务质量极低。

结论：暂不启动 Qwen2.5-1.5B-Instruct 和 Qwen3-4B-Instruct-2507。优先修正
WebShop prompt/tool interface：明确 `search[keywords]` 是模板，必须替换为
真实查询；限制同一 episode 的重复 search/back/next；child 继续允许 act，
但需要由模型明确 live-operation 委派，并禁止无目的递归 spawn。修正后再做
模型横向测试，否则后续模型结果会被同一个 action/prompt 问题污染。
 
