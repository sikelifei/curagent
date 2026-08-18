# TextCraft-Synth：递归求解路径、Prompt / Few-shot 与采样记录

## 1. 结论

TextCraft-Synth 最适合检验的不是“模型能否一次展开完整配方树”，而是一个通用递归 agent 是否学会下面这个闭环：

1. 只理解当前节点的局部目标；
2. 能直接完成就直接完成；
3. 否则产生严格更小、边界清楚、结果可验证的子任务；
4. 显式传递子任务所需的上下文，不依赖父节点的隐藏状态；
5. 等子任务返回后重新观察真实环境；
6. 父节点完成本层组装并验证结果；
7. 只有根节点提交全局完成。

这个协议是通用的。TextCraft 只是把它具体化成“目标物品 -> 直接原料子任务 -> 当前物品组装”。不需要在 agent 内预先构建整棵图，也不应把 TextCraft 专用规划器写进通用 agent 内核。

本轮用 DeepSeek-V4-Flash、temperature 0 对当前实现完成了 4 条有效轨迹，并记录了 1 条 hard 基础设施失败：

| 难度 | 样本 | 深度 | 结果 | 根步骤 | 子 agent | 最大递归深度 | craft 次数 | 墙钟时间 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| easy | generated-easy-101-0 | 2 | 成功 | 5 | 0 | 0 | 8 | 11.0 s |
| easy | generated-easy-103-0 | 3 | 成功 | 13 | 0 | 0 | 7 | 36.6 s |
| medium | generated-medium-201-0 | 4 | 成功 | 14 | 3 | 1 | 38 | 144.2 s |
| medium | generated-medium-203-0 | 4 | 成功 | 7 | 2 | 1 | 23 | 245.2 s |
| hard | generated-hard-301-0 | 8 | API 连接失败 | 轨迹丢失 | 轨迹丢失 | 轨迹丢失 | 未知 | 232.8 s |

easy 和 medium 都得到满分。hard 的记录是基础设施失败，不是可解释的任务失败：runner 捕获顶层 `ModelCallError` 后只保存 `steps=0` 的错误行，已经执行的局部轨迹和环境状态没有落盘，因此不能用它判断策略成功率。

原始轨迹：

- [`easy_seed101_trace.json`](./easy_seed101_trace.json)
- [`easy_seed103_trace.json`](./easy_seed103_trace.json)
- [`medium_seed201_trace.json`](./medium_seed201_trace.json)
- [`medium_seed203_trace.json`](./medium_seed203_trace.json)
- [`hard_seed301_trace.json`](./hard_seed301_trace.json)

对应摘要：

- [`easy_summary.json`](./easy_summary.json)
- [`easy_seed103_summary.json`](./easy_seed103_summary.json)
- [`medium_seed201_summary.json`](./medium_seed201_summary.json)
- [`medium_seed203_summary.json`](./medium_seed203_summary.json)
- [`hard_seed301_summary.json`](./hard_seed301_summary.json)

## 2. Benchmark 与当前实现

论文中的 TextCraft-Synth 用合成物品和依赖关系替代 Minecraft recipe，通过 crafting depth 控制难度：easy 为 2–3，medium 为 4–6，hard 为 7–9。论文训练只使用 medium，评测允许最大递归深度 12，用于观察对 easy 和 hard 的泛化。

论文公开的 action space 是：

```text
craft(ingredients: dict, target: tuple[str, int]) -> str
get_info(items: list) -> list[dict]
finish(message: str) -> str
launch_subagent(targets: dict) -> str
```

当前仓库实现为：

```text
craft(ingredients: dict[str, int], target: tuple[str, int]) -> str
get_info(items: list[str] | None = None) -> list[dict]
view_inventory() -> dict[str, int]
finish(message: str) -> str
spawn_subagent(task: str, context=None) -> str
spawn_subagents(requests: list[dict]) -> list[str]
```

关键语义如下：

- `craft` 的 `target[1]` 是本次实际产出量，不是 recipe 执行次数；它必须能被 `result_count` 整除。
- `ingredients` 必须等于该产出量对应的精确缩放配方；少一个、多一个都会报错。
- 根任务要求的是 additional quantity。最终目标数为 `initial_target_count + requested_count`。
- 所有递归 agent 共享同一个 live inventory；每个 agent 的 Python REPL 和显式 `context` 则相互隔离。
- `spawn_subagents` 会并发执行多个子任务，并阻塞父节点直到全部返回。
- 当前 child 仍能看到 `finish`。prompt 虽要求 child 不调用，但环境没有强制禁止。

当前本地数据不是论文发布的 canonical test set。默认路径不存在时，loader 使用确定性 generator 生成一棵真正的树：每个路径上的物品名唯一，叶子原料也独立。因此当前生成集适合验证递归行为，但不能直接把分数与论文表格横向比较。

### 当前接口 / 评测实现需要注意的问题

- `get_info()` 不传 items 时只返回全局 task targets，并不返回 child 当前局部目标或全部 recipe。medium 轨迹中 child 因此出现了多次无效查询；few-shot 应始终展示显式 `get_info([item])`。
- `can_craft` 只表示当前库存是否已直接满足某个 recipe，不表示该 item 能否通过递归最终制成。
- `_item_depth` 使用第一个 recipe 计算深度，但 `craft` 接受任意匹配 recipe；包含多 recipe 的外部数据可能出现路由深度与实际选择不一致。
- child 能调用 `finish`，目前仅靠自然语言约束。若后续强化权限边界，通用 harness 应支持按 delegated role 隐藏全局提交工具，而不是把这个规则硬编码进 TextCraft 解题逻辑。
- 论文的 `launch_subagent(targets: dict)` 天然提供结构化局部目标；当前通用 `spawn_subagent(task, context)` 更灵活，但 prompt / few-shot 必须补上 goal schema 与返回 schema，否则容易退化为含糊的自然语言委派。
- 顶层模型 / 网络异常时，runner 不保留 partial trace 和 partial environment report，hard 失败无法归因。
- 早期 TextCraft tests 曾匹配旧 task 文案 `Craft the following items`；该断言和 fake handler 分支现已更新为 additional-quantity 语义。

## 3. 纯递归求解路径

每个 agent 只处理一个局部目标，例如：

```json
{
  "item": "part_A",
  "required_inventory": 4,
  "root": false
}
```

这里使用 `required_inventory`（调用者最终需要库存中至少有多少），比“additional 4”更适合作为父子协议：父节点根据自己本层 recipe 算出绝对需求，child 返回后父节点可以直接检查库存是否达到阈值。

单个 agent 的递归过程：

1. `view_inventory()` 得到当前 `item` 数量。
2. 如果已达到 `required_inventory`，立即返回 satisfied，不重复制造。
3. `get_info([item])` 只读取当前 item 的 recipe、`result_count` 和深度。
4. 计算 shortage，并向上取整为合法产出量：

   ```python
   executions = ceil(shortage / result_count)
   output_count = executions * result_count
   ```

5. 将 recipe 中每个直接 ingredient 乘以 `executions`，得到本层精确需求。
6. 若依赖足够浅，当前 agent 直接制造；否则把直接 ingredient 目标委派出去。子任务必须比当前目标更浅，不能原样复制父任务。
7. 子任务完成后丢弃旧库存假设，重新 `view_inventory()` / `get_info(...)`。
8. 依赖数量满足时，仅执行当前 item 的 `craft`。
9. 验证当前 item 数量，返回结构化报告。child 不调用 `finish`；root 在全局目标全部满足后才调用。

这不是全局 DAG 规划。每个节点只看到自己和直接依赖，递归调用树自然跟随 crafting tree 生长。

### 委派粒度

- easy：通常直接完成，不递归。
- medium：根节点可以把多个直接分支委派给一级 children，自己保留最终组装。
- hard：children 面对仍然较深的目标时重复同一决策，因此自然形成多层递归。
- 不应固定“只要 depth >= 4 就一定委派”。正确判据是：当前节点能否在自己的步骤 / 上下文预算内可靠完成，以及子任务是否真正更小并可独立验证。

### 子任务契约

推荐每个委派请求至少包含：

```python
{
    "task": "Ensure the shared inventory contains at least 4 units of part_A.",
    "context": {
        "goal": {
            "item": "part_A",
            "required_inventory": 4,
            "root": False
        }
    }
}
```

推荐 child 返回紧凑、可机器检查的 JSON，而不是长篇制作过程：

```json
{
  "item": "part_A",
  "required_inventory": 4,
  "final_inventory": 4,
  "satisfied": true,
  "error": null
}
```

父节点不把这个字符串当作事实；它以返回值做索引，然后重新查询 live inventory 验证。

## 4. 推荐的通用递归 Prompt

下面这一段适合放在通用 recursive harness 中。它不包含 TextCraft 配方算法。

```text
You are a recursive agent. Complete the assigned task using direct work and
smaller delegated tasks.

At every node, first identify the node's exact completion condition and the
evidence that can verify it. Solve directly when the task fits reliably within
this node's available steps and context. Otherwise delegate only strict
subtasks: each child must have a narrower objective than the current node, a
self-contained task statement, the minimum required context, and an explicit
return contract.

Delegate independent subtasks concurrently. Keep dependent subtasks serial.
Do not delegate an unchanged copy of the current task. Do not assume a child
can see this node's messages, variables, or reasoning; pass required values in
task or context. A child may recurse under the same rules.

After children return, treat their reports as claims to verify with the
available environment or evidence. Re-observe mutable state before acting.
The parent retains responsibility for integration, verification, and its own
completion condition. Only the root performs the environment's global submit
or finish action unless the delegated task explicitly grants that authority.

Return compact, self-contained child reports. Prefer structured fields for the
goal, result, verification evidence, unresolved work, and errors.
```

## 5. 推荐的 TextCraft 薄适配 Prompt

这一段只解释环境规则和 root / child 权限，可叠加在通用 prompt 后。它刻意不教模型展开整棵树。

```text
### TextCraft-Synth

Craft the requested additional target quantities in the shared live inventory.
Use print(view_inventory()) to observe inventory and print(get_info([...])) to
obtain recipes. A recipe's ingredient counts and result_count are per
execution. craft(ingredients, (item, output_count)) requires exact scaled
ingredients, and output_count must be a multiple of result_count.

For each local crafting goal, determine the required final inventory count,
account for inventory already present, and round production up to complete
recipe executions. Work directly when the local goal is small. Otherwise
delegate exact, strictly smaller intermediate-item goals and keep the current
item's assembly in the parent. Give every child an item and a required final
inventory count. After children return, re-check the shared inventory before
crafting.

All agents share inventory but have isolated message histories and REPL
variables. Concurrent delegation is appropriate only for independent local
goals. A delegated agent returns a compact report for its own goal and must not
call finish. Only the root calls finish, after re-checking every requested
target against its initial count plus the requested additional amount.

Print every tool result that must be observed. Do not emit simulated tool
outputs or plan an entire action sequence before seeing real results.
```

## 6. Few-shot 设计

Few-shot 的目的不是灌输具体物品名，而是展示三种通用决策以及父子通信协议。训练样例中的每个 assistant block 应对应一个真实 model step；不要在一个 response 中伪造后续 observation。

### Few-shot A：小任务直接完成

任务：额外制造 1 个 `tool`。

```repl
initial = view_inventory()
print(initial)
print(get_info(["tool"]))
```

真实 observation 显示 `tool` 当前为 0，recipe 为 2 `ingot` + 1 `wood` -> 1 `tool`。

```repl
print(get_info(["ingot", "wood"]))
```

真实 observation 显示 `wood` 已有 1，`ingot` 可由 2 `ore` 一次产出 2，且 `ore` 足够。局部任务很小，直接执行。

```repl
print(craft({"ore": 2}, ("ingot", 2)))
print(craft({"ingot": 2, "wood": 1}, ("tool", 1)))
print(view_inventory())
```

根节点验证 `tool >= initial["tool"] + 1` 后：

```repl
print(finish("crafted the requested additional tool"))
```

学习点：不为递归而递归；先观察真实工具结果；正确处理 `result_count`。

### Few-shot B：父节点委派独立直接分支

任务：当前节点需要库存中至少 2 个 `machine`。真实 `get_info(["machine"])` 显示一次产出 2，需要 3 `frame` 和 2 `engine`，两个分支均较深。

```repl
result = get_info(["machine"])[0]
print(result)
```

```repl
reports = spawn_subagents([
    {
        "task": "Ensure the shared inventory contains at least 3 frame.",
        "context": {
            "goal": {
                "item": "frame",
                "required_inventory": 3,
                "root": False
            }
        }
    },
    {
        "task": "Ensure the shared inventory contains at least 2 engine.",
        "context": {
            "goal": {
                "item": "engine",
                "required_inventory": 2,
                "root": False
            }
        }
    }
])
print(reports)
```

children 返回结构化报告后，父节点必须重新验证，而不是直接相信报告：

```repl
inventory = view_inventory()
print(inventory)
```

当 `frame >= 3` 且 `engine >= 2`：

```repl
print(craft({"frame": 3, "engine": 2}, ("machine", 2)))
print(view_inventory())
```

学习点：child goal 精确且严格更小；父节点保留本层 assembly；并发只用于独立分支；返回后重新观察共享状态。

### Few-shot C：child 继续递归

child 收到：

```json
{"item": "engine", "required_inventory": 2, "root": false}
```

它不拥有全局 `machine` 任务，也不需要知道父节点的完整历史。

```repl
goal = context["goal"]
inventory = view_inventory()
print(inventory)
print(get_info([goal["item"]]))
```

真实 observation 显示 `engine` 当前为 0，一次产出 1，直接依赖 1 `rotor` 和 2 `coil`，两者仍较深。child 使用相同递归规则：

```repl
reports = spawn_subagents([
    {
        "task": "Ensure the shared inventory contains at least 2 rotor.",
        "context": {"goal": {"item": "rotor", "required_inventory": 2, "root": False}}
    },
    {
        "task": "Ensure the shared inventory contains at least 4 coil.",
        "context": {"goal": {"item": "coil", "required_inventory": 4, "root": False}}
    }
])
print(reports)
print(view_inventory())
```

验证依赖后，child 只组装自己的 `engine`：

```repl
print(craft({"rotor": 2, "coil": 4}, ("engine", 2)))
final_inventory = view_inventory()
print(final_inventory)
```

然后返回，不调用 `finish`：

```repl
answer["content"] = json.dumps({
    "item": "engine",
    "required_inventory": 2,
    "final_inventory": final_inventory.get("engine", 0),
    "satisfied": final_inventory.get("engine", 0) >= 2,
    "error": None
})
answer["ready"] = True
```

学习点：同一个 policy 在 child 中继续使用；任务随深度严格缩小；child 的停止条件是自己的局部目标，而不是全局完成。

### 不建议作为 few-shot 的行为

- 把完整根任务原样交给 child。
- 在 task 字符串中复制整棵 recipe tree。
- child 制造完中间件后调用 `finish`。
- 父节点在 child 返回后不看库存，直接执行 craft。
- 只传自然语言“帮我做这个分支”，不传 item、数量和返回契约。
- 在一个 assistant response 中连续写很多 REPL blocks，并假装已经知道每一步的 observation。
- 为了展示递归，在 easy 上也强制 spawn。

## 7. 本轮轨迹说明

### Easy

两条 easy 分别为 depth 2 和 depth 3，均先查询目标及依赖、按层 craft、最后检查库存并调用 `finish`，没有创建 child。这是合理行为：浅任务不值得支付额外 agent 的调用和通信成本。depth-3 样本用了 13 个根步骤，说明即使不需要递归，工具调用示范仍应强调减少无输出调用和重复查询。

### Medium

第一条 medium 的根节点发现目标的三个直接 `m3` 分支，精确计算每个只需 1 个，然后并发创建 3 个 children；第二条创建 2 个一级 children。两条轨迹都由根节点在 children 返回后负责最终 `m4` 组装。这个职责划分是正确的。

但轨迹也有明显低效：

- 有 child 首先调用 `view_inventory()` / `get_info()` 却没有 `print`，下一步又重新调用。
- 一个 child 在单个 response 中进行了大量重复 `get_info`，输出 token 达 8039。
- 环境记录了两次 `Insufficient ...`，说明根或 child 曾基于过期 / 不完整库存尝试 craft，之后虽恢复成功，但这种错误会在 hard 放大。
- 整条 medium 使用 35 次模型调用、141,539 input tokens、13,923 output tokens；成功但效率不理想。
- 第二条 medium 也成功且无 tool error，但 25 次模型调用仍使用了 166,200 input tokens、25,556 output tokens，墙钟时间反而达到 245.2 秒。这表明只有 final success 不能衡量递归质量。

这说明 few-shot 最应该教的是“每一步观察真实结果、子任务契约、返回后验证”，而不是更详细地列举 TextCraft 规则。

### Hard

depth 8 样本运行约 233 秒后 API connection error。当前 `_run_one` 只拿到异常，没有保存 agent 已构建的 trace 或环境的 partial report，因此摘要显示 `steps=0`。建议在正式采样前增加 checkpoint / finally 落盘：至少保存每个 agent step、每次 craft event、children tree 和异常发生时的 inventory。否则网络失败、模型失败和递归策略失败会混在一起。

## 8. 对 WebShop / Oolong Prompt 的借鉴

WebShop 当前 prompt 最有价值的是权限与串并行边界：环境只能串行操作，child 只分析父节点显式传入的 observation snapshot，父节点回来后重新 observe。这正是通用递归 prompt 应保留的“可变状态必须重新验证”。

Oolong-Synth 当前 prompt 最有价值的是明确的分解单位和 context contract：父节点按完整 record 边界切块，每个 child 只收到 question、dataset intro、chunk text、chunk id，父节点负责合并。它体现了“child 不继承隐式上下文”。

不应泛化进通用 prompt 的部分，是 WebShop 的具体动作序列、Oolong 的固定字符阈值和标签解析格式。通用 agent 只需要学会：根据预算选择直接做或委派、产生严格更小的子任务、明确传递上下文、验证并聚合结果。

## 9. 后续 RL 数据建议

训练样本应覆盖行为选择，而不只是最终成功：

- easy direct-success：奖励不递归也能快速正确完成。
- needless-recursion negative：easy 强制递归，虽然成功但步骤 / token 更差。
- medium one-level：根拆直接分支，child 完成局部目标，根组装。
- hard recursive-child：至少一个 child 再递归，且每层任务严格缩小。
- bad-contract recovery：child 返回含糊报告，父节点通过环境验证发现未完成并修复。
- stale-state recovery：并发返回后库存改变，父节点重新观察再行动。
- overreach negative：child 调用全局 finish 或接管父任务。
- unchanged-delegation negative：child 收到与父节点实质相同的任务。

除 root success 外，可以记录过程标签：`subtask_strictly_smaller`、`context_self_contained`、`child_goal_satisfied`、`parent_verified`、`root_only_finish`、`invalid_craft_count`、`redundant_tool_query`。这些标签适合做 verifier、过滤 SFT few-shot，或作为后续 dense recursive reward 的组成部分。

最重要的是不要奖励“递归次数”。应奖励正确的路由：简单任务直接做，真正超出本节点可靠预算时才拆；child 的局部成功可验证，父节点最终整合成功。

## 10. 2026-08-05 Prompt 迭代与批量轨迹审计

本节续记当前工作区中的后续实验。前文第 1–9 节保留为早期设计与
sample 记录；本节中的 JSONL 和源码是当前迭代的权威证据。

### 10.1 递归质量判据

新增了 TextCraft 专用机械分析器：

- `recursive_agent/envs/textcraft_synth/trace_analysis.py`
- `examples/analyze_textcraft_results.py`
- 聚合输出：`output3/textcraft_prompt_analysis.json`

分析器不仅统计最终分数，还检查：

- depth 2–3 是否发生不必要递归；
- depth >= 4 是否完全没有 child；
- 是否调用并发 `spawn_subagents` 修改共享 inventory；
- child 是否调用 `finish`；
- child task 是否包含 absolute inventory contract；
- 是否原样转发 parent task；
- `get_info` 是否无参数调用或在同一 agent 内重复同一查询；
- observation truncation、execution error、agent 数和最大递归深度。

这些指标是诊断信号，不是 benchmark 官方分数。例如一次重复查询不会使成功
轨迹失效，但会使 `recursion_reasonable=false`，便于在扩大采样前发现退化。

### 10.2 已观察到的 Prompt 版本

表中的 fingerprint 是保存于 trace 中的完整 system prompt SHA-256 前 12 位，
因此同一个结果文件不会被误认为使用了当前源码中的 prompt。

| 版本 | system prompt fingerprint | 数据与结果 | 递归表现 | 决策 |
|---|---|---|---|---|
| P0 | `178e95843ea2` | easy 20/20 成功；medium 9/20 成功，15 条有 trace | easy 0 child 合理；13/15 medium trace 完全没有递归 | 淘汰：deep task 路由太弱 |
| P1 | `6bd8d69c82dc` | medium 5 条记录，4 条完成且成功 | 14 children、最大深度 2，但 4/4 有并发 mutation 风险；593 次重复 `get_info`、44 次无参数查询、31 次截断 | 淘汰：会递归但规划和并发失控 |
| P2 | `54c0615b27fd` | 受限 medium 1 条，3 steps 后 forced-final，0 craft | 186 次 `get_info`、90 次无参数查询、0 child | 淘汰：详细示例诱导全树展开和重复查询 |
| P3（当前） | `1be048e5089c` | 单元测试 16/16 通过；真实回归被 API `429 insufficient_quota` 阻断 | 规则改为 deep target 只委托一个 serial child，禁止并发和预展开 descendants | 保留为待验证候选 |

对应原始文件：

- P0 easy：`output3/textcraft_easy_rerun/results.jsonl`
- P0 medium：`output3/textcraft_medium_rerun/results.jsonl`
- P1 medium：`output3/textcraft_eval_medium_3m/results.jsonl`
- P2 medium：`output3/textcraft_prompt_regression_medium_small/results.jsonl`

### 10.3 当前 P3 Prompt

当前环境 prompt 位于
`recursive_agent/envs/textcraft_synth/prompts.py`，环境段为 1572 字符、241 个
空白分词，SHA-256 前 12 位为 `28d99d22b8ee`。完整 system prompt（通用递归
prompt + TextCraft 环境段 + tool descriptions）为 4387 字符，fingerprint 为
`1be048e5089c`。

P3 保留的核心只有：

1. additional quantity 和精确 recipe batch 语义；
2. depth >= 4 时只检查当前 item 和直接 recipe，立即委托一个严格更小的
   absolute-inventory 子目标；
3. shared inventory 下只允许 serial `spawn_subagent`，禁止
   `spawn_subagents`；
4. child 返回后重新读取 live inventory；
5. child 通过 `answer` 返回，只有 root 调用 `finish`；
6. 禁止无参数或重复的 `get_info`。

长 few-shot、固定物品名、完整 recipe tree 和重复的工具 schema 均未放入当前
环境 prompt。root task prompt 仍只包含本 episode 的 `{targets}`。

### 10.4 当前证据边界与下一轮 gate

P3 尚不能宣称实验完成：最后一次真实 medium 回归在首个 model call 前收到
`429 insufficient_quota`，因此没有 P3 的有效递归 trace。API 恢复后应先运行：

1. 2 条 easy：要求成功、0 child、无 child finish；
2. 2 条 medium：要求至少产生 serial child，`spawn_subagents_calls=0`，child
   task 使用 absolute contract，parent 回来后读取 inventory；
3. 1 条 hard：要求最大递归深度至少 2，且每层 task 不与 parent 相同；
4. 小样本通过后再运行 20 easy / 20 medium / 20 hard。

扩大采样的过程 gate：无 child finish、无并发 mutation、无 unchanged-task
delegation；`get_info` 重复和 observation truncation 应显著低于 P1。最终成功率
与过程 gate 必须同时报告，不能用成功分数掩盖不合理递归。
