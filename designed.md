# Recursive Agent 基础框架设计

## 1. 设计目标

实现一个通用的递归 Agent Harness。每个 Agent 都是相同的 `AgentNode`：它可以直接使用环境工具完成任务，也可以把任务拆给一个或多个 child；child 仍运行同一套 Agent loop，因此可以继续递归。

Harness 只负责模型调用、工具执行、递归调度、环境隔离、预算和轨迹记录，不替模型做策略优化。禁止自动修正 action、补参数、选择相近动作、改写代码或推断答案，保证模型原始决策与 reward 可以直接对应。

## 2. 递归模型

每个节点包含：

```text
AgentNode
├── agent_id / parent_id / depth
├── task                 当前节点的任务
├── context              父节点显式传入的只读信息
├── trajectory           当前节点自己的完整轨迹
├── environment          共享、只读或 clone 环境句柄
└── budgets              节点预算和任务树共享预算
```

父 Agent 自主决定是否分解任务。Harness 不强制递归，也不自动生成子任务：

- 直接调用环境工具：当前节点自己执行；
- `spawn_agent(...)`：启动一个 child 并等待结果；
- `spawn_agents(...)`：并行启动多个 child，等待全部结束；
- `finish(...)`：返回当前节点结果。

child 的输入与返回接口为：

```text
spawn_agent(task, context, expected_output?, access?) -> SubagentResult

SubagentResult = {
  task,
  context,
  status,
  result,
  error
}
```

`context` 必须由父 Agent 显式生成且可序列化，不能引用父节点的隐式变量。父节点只能看到 child 的分发信息和最终 `SubagentResult`，看不到 child 的内部 prompt、observation 和逐步轨迹。

## 3. 统一 Agent Loop

root 和所有 child 执行同一个循环：

```text
while node 未结束且预算充足:
    observation = environment.observe()
    prompt = compose(base_prompt, task_module, node_state)
    response = model(prompt, available_tools)
    record(raw_response)
    tool_call = strict_parse(response)
    result = execute_exactly(tool_call)
    append(tool_call, result) to node.trajectory
return SubagentResult
```

一次模型响应只接受一个 tool call。原生 tool call 是默认协议；不支持原生 tool call 的模型可以使用严格 JSON，但解析后必须得到同一个内部 `ToolCall`，不能从自然语言猜测动作。

## 4. 工具设计

### 4.1 框架工具

所有任务共享三个递归控制工具：

```text
spawn_agent(task, context, expected_output?, access?)
spawn_agents(specs)
finish(result)
```

`spawn_agents` 返回与输入顺序一致的结果列表。单个 child 失败不取消其他 child；所有 child 到达终态后才返回父节点。

### 4.2 环境工具

环境工具由任务模块直接注入，不使用 `act("search[...]")` 这类二次字符串协议。例如：

```text
WebShop: search(query), click(item_id), buy(option)
Search:  search(query), click(result_id), open(url), submit(answer)
```

Harness 按 tool schema 校验参数后原样交给 environment adapter。参数非法时返回真实错误，不修正或替换调用。

### 4.3 Python Capability

Python 不是递归 Agent 的默认输出格式，而是可选工具：

```text
python_exec(code)
```

WebShop、搜索和离散工具任务默认不暴露该工具；代码执行、数据分析或确实需要循环和复杂聚合时才启用。代码按模型传入内容原样执行，Harness 不提取、补全或修复代码。Python executor 默认只做纯计算，不直接持有可写环境工具，避免代码执行一半后报错造成部分副作用；它也不能绕过权限和任务树预算。

## 5. Prompt 设计

Prompt 采用“通用 Base Prompt + 独立 Task Module”，不能为每个 benchmark 复制完整 Agent prompt。

所有任务共用的 `BasePrompt` 只描述：

- 直接执行与递归分解的选择；
- framework tools 的调用规则；
- context 和 child result 的边界；
- 轨迹、预算、异常和终止规则；
- Harness 不修正模型决策的原则。

每个任务只注册一个精简模块：

```text
TaskModule
├── instruction          任务指令
├── observation_spec     observation 说明
├── environment_tools    工具 schema
├── environment_rules    操作限制
└── finish_condition     完成条件
```

Task Module 只能说明环境事实，不能加入候选排序、答案抽取、相似 action 匹配或 benchmark 特定捷径。

每个 step 的模型输入由以下内容组成：当前 `task + context`、Base Prompt、Task Module、当前节点完整轨迹、最新 observation、可用工具和剩余预算。child 不继承父节点完整轨迹，只接收父节点传入的 `task + context`；child result 作为一次 tool result 加入父节点轨迹。

## 6. 环境接口与隔离

所有 benchmark 实现相同接口：

```text
reset(instance) -> Observation
observe() -> Observation
tools(access) -> ToolSchema[]
execute(tool_call, expected_version) -> ExecutionReceipt
reconcile(call_id) -> ExecutionReceipt | None
is_done() -> bool
reward() -> float
capabilities() -> EnvCapabilities
clone() -> Environment | None
close() -> None
```

`Observation` 至少包含 `text`、`version` 和 `metadata`。写操作携带 `expected_version`，避免基于过期 observation 执行。

环境执行统一返回回执，不能只通过是否抛异常猜测状态：

```text
ExecutionReceipt
├── call_id
├── status: success | rejected | failed
├── effect: no_change | committed | unknown
├── result / error
├── version_before / version_after
└── observation
```

Adapter 应先完成参数、权限和版本校验，再产生副作用。每次调用使用唯一 `call_id`；环境支持幂等键时可以透传，但 Harness 仍不能自动重放。无法确定远端操作是否成功时必须返回 `effect=unknown`，不能谎报为失败，并通过 `reconcile(call_id)` 尝试对账。

环境访问遵循明确能力：

- `owner`：root 使用真实环境并负责最终提交；
- `readonly`：child 只能读取或分析；
- `clone`：child 在隔离 session 中操作，结果不自动合并；
- `delegated`：child 获得 single-writer lease 后串行写入。

默认 child 使用 `readonly`。WebShop 等共享状态环境采用 single-writer；支持 clone 的搜索环境可以并行探索。环境不支持请求的 access 时返回错误，不能静默降级。

## 7. 异常处理

异常处理只负责暴露事实和保护副作用，不负责修正模型策略。

### 7.1 错误分类

| 错误 | 处理方式 |
| --- | --- |
| 模型服务超时或连接失败 | 在没有 tool call 的前提下做有限基础设施重试；仍失败则结束节点 |
| tool call 无法解析或参数不符合 schema | 记录原始输出，反馈错误，允许模型重新决策一次 |
| 环境拒绝且 `effect=no_change` | 反馈真实环境错误和最新 observation，允许重新决策一次 |
| stale version | 刷新 observation，禁止重放旧调用，让模型重新决策 |
| 调用成功或 `effect=committed` | 不重试；即使后处理报错，也从最新状态进入下一 step |
| 超时且 `effect=unknown` | 禁止自动重放；先 `observe()` 对账，仍无法确认则以 `uncertain` 结束 |
| 权限、配置、预算或不可恢复 runtime 错误 | 直接结束当前节点 |

模型修复重试是一次新的模型输出，不是 Harness 修改并重放旧调用。默认每个 step 最多修复一次；第二次仍失败，节点返回 `error`。

### 7.2 错误反馈

反馈采用固定结构，只提供事实：

```text
error_type
original_error
failed_tool_call
effect: no_change | committed | unknown
latest_observation
remaining_budget
```

错误反馈中不提供替代 action、不改参数、不生成代码补丁。原有 tool schema 会正常保留，模型自行产生新的完整调用。

### 7.3 递归异常

child 的异常统一转换为 `SubagentResult(status="error")`，不能穿透并使父节点崩溃。`spawn_agents` 等待并收集所有 child 的结果；一个 child 失败不取消其他 child。是否再次分发、改由父节点完成或直接结束，由父模型在下一 step 决定，scheduler 不自动重启 child。

`spawn_agents` 必须先完整解析所有 spec，并一次性检查深度、child 数量和预算，再启动任何 child。启动之后使用 gather barrier 收集全部结果，避免出现因某个 spec 非法而只启动一部分 child 的情况。

达到 `max_depth`、child 数量或共享预算上限时，也返回明确的失败结果，不静默降低 access、不改成直接执行。

## 8. 预算与轨迹

框架至少提供：

```text
max_steps_per_agent
max_retries_per_step = 1
max_model_calls_total
max_tool_calls_total
max_depth
max_children_total
max_concurrency
```

`max_steps_per_agent` 统计已进入执行阶段的决策；格式修复不消耗环境 step，但消耗模型调用预算。root 和全部 descendants 共享任务树总预算，避免递归放大资源。基础设施重试和模型修复重试必须分开计数。

每次决策记录：`agent_id`、`parent_id`、`depth`、prompt、observation、原始模型输出、解析后的 tool call、`ExecutionReceipt`、错误、环境版本、reward 和预算。每次重试是独立轨迹，不能覆盖失败输出。终态至少区分 `ok`、`error`、`uncertain`、`max_steps` 和 `budget_exhausted`。

## 9. 基础代码结构

```text
curagent/
├── core/
│   ├── agent.py          AgentNode 与统一循环
│   ├── types.py          ToolCall、Result、Observation
│   ├── scheduler.py      递归、并发和环境 lease
│   ├── budget.py         任务树共享预算
│   ├── prompt.py         BasePrompt 与模块组合
│   └── trace.py          原始轨迹记录
├── environments/
│   └── base.py           Environment 接口
├── tasks/
│   └── <task_name>.py    Task Module 与环境注册
├── executors/
│   └── python.py         可选 Python capability
└── tests/
```

第一版只实现统一 Agent loop、严格 tool call、递归调度、context/result 传递、共享预算、异常反馈和一个 mock environment。基础闭环稳定后，再接 WebShop 或搜索 benchmark，不在核心层加入任务策略。用git做好版本管理。
实现完成代码之后，可以接入webshop的部分，实现一下基础的推理的评测，webshop的环境可以从/data2/zhangwenjian/agent/ReCode 用，conda用recode的环境就能正常运行，请你完整实现这个，最后给出一个200条的test的指标
