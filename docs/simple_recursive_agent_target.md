# 简化递归 Agent 目标设计

本文件是当前实现应遵循的第一版简化规格。

## 1. 目标

本项目只需要实现一个清晰、可运行的递归 Agent，不需要在第一版解决所有环境隔离、权限租约、clone、事务和多 benchmark 抽象。

Agent 的核心能力只有三件事：

1. 模型决定当前节点直接行动、创建 child，还是结束；
2. child 使用同一个 Agent loop，并接收父模型显式提供的任务和上下文；
3. 每次模型输出后，把实际结果原样放回下一轮 prompt，让模型自行判断如何继续。

目标不是让 Harness 替模型规划，而是让 Harness 成为一个薄执行循环。

## 2. 最终结论

### 2.1 环境是否需要传给 child

如果当前任务树有 environment，child 自动继承父节点的同一个 environment。

```text
root         -> 可以 observe 和执行 environment actions
child        -> 使用同一个 environment，也可以 observe 和执行 actions
grandchild   -> 继续使用同一个 environment
```

父节点不需要显式声明 child 是 readonly、writable 或 delegated。父模型通过 `task/context` 表达希望 child 完成什么：

```json
{
  "task": "判断哪个商品最符合要求",
  "context": {
    "goal": "找到正确商品并打开商品页面",
    "constraints": ["蓝色", "32 oz", "不锈钢"]
  }
}
```

child 进入同一个 Agent loop 后，可以选择分析、继续 spawn，也可以直接调用 `search/click/buy`。child 完成后，父节点重新 observe，看到 child 已经造成的最新环境状态。

核心原则是：

- 父模型决定委派什么任务；
- child 模型决定为完成任务是否需要 action；
- Harness 不分析 task 文本来判断是否允许修改环境；
- Harness 不增加显式 `access` 参数；
- 所有递归节点具有相同的 loop 和相同的环境能力。

这会牺牲细粒度权限隔离，但更符合当前目标：先实现一个真正同构、由模型自主递归和行动的 Agent。

### 2.2 环境是否仍然需要

环境对 WebShop 这类交互任务仍然必要，并属于整棵任务树的共享运行资源，而不是 root 私有资源。

```text
递归机制：AgentNode + spawn + shared budget + depth
环境机制：observe + tools + execute + done/reward
```

纯分析任务可以没有环境；WebShop 任务树中的所有节点共享同一个环境；两者使用同一个 Agent loop。

## 3. 只保留两个任务限制

### `max_total_steps`

root 和所有 child 共享一个总 step 额度。

一次模型输出就是一步，包括：

- 正常 tool call；
- 格式错误；
- 参数错误；
- 未知工具；
- 环境执行失败；
- `spawn_agent`、`spawn_agents` 和 `finish`。

模型服务完全没有返回内容时，不算一步，因为没有可反馈给模型的输出；Harness 释放预留额度并结束当前节点。

### `max_depth`

root 深度为 0。child 深度为父节点加 1。超过深度时不启动 child，只把简单失败结果返回父模型。

不设置：

- child 数量预算；
- 每个 agent 的 step 预算；
- model/tool 分开预算；
- retry 次数；
- repair 次数；
- 环境 effect 分类。

第一版不需要 `child_concurrency`。`spawn_agent` 等待一个 child，`spawn_agents` 按输入顺序逐个运行 child。递归是完整的，只是不并行，因此共享 environment 不需要额外写锁。

## 4. Agent loop

root 和 child 完全相同：

```text
while shared steps remain:
    如果共享 environment 已结束，返回环境结果
    预留一个 shared step
    获取当前 observation（任务树没有 environment 时跳过）
    组合 prompt
    调用模型

    如果没有模型输出：释放 step，结束当前节点
    如果有模型输出：提交 step

    尝试解析一个 tool call
    解析失败 -> 记录普通结果，进入下一轮
    解析成功 -> 执行一次，记录普通结果，进入下一轮

    finish -> 返回当前节点结果
```

Harness 不根据错误类型决定“这次是否值得重试”。只要还有 shared step，模型就可以自行继续。

## 5. Prompt

每个节点的 prompt 只包含：

```json
{
  "task": "当前节点任务",
  "context": {},
  "trajectory": [
    {
      "model_output": {},
      "execution_result": "..."
    }
  ],
  "observation": {},
  "tools": [],
  "remaining_steps": 12
}
```

规则：

- 所有节点的 `observation` 都来自同一个共享 environment；
- 整棵任务树没有 environment 时省略 `observation` 或设为 `null`；
- child 不继承父 trajectory；
- child 接收父模型显式传入的 task/context，同时像父节点一样读取当前 environment；
- prompt 不放 `error_type`、`effect`、`retry_count`、`failed_call` 等额外字段；
- prompt 不放 reward，reward 由 WebShop evaluator 记录；
- `remaining_steps` 是整棵树共享剩余额度。

System prompt 只需要表达最小协议：

```text
You are a recursive agent node.
Use one available tool, spawn child agents when useful, or call finish.
Tool results are returned in trajectory; decide the next step yourself.
```

不需要在 system prompt 中逐项解释 repair、effect、权限、错误分类、候选选择或环境恢复策略。

## 6. 工具

第一版只需要四类工具：

```text
spawn_agent(task, context, expected_output?)
spawn_agents(specs)
finish(result)
shared environment tools
```

### Environment tools

只要任务树拥有 environment，root、child 和 grandchild 都能看到同一组环境工具。WebShop 中就是：

```text
search(query)
click(target)
buy()
```

child 与父节点拥有相同工具能力，并且还可以继续创建自己的 child。父节点通过任务文本约束 child 的目标，而不是由 Harness 删除 action tools。

### 删除的复杂参数

从 `spawn_agent` 和 `spawn_agents` schema 删除：

```text
access: owner/readonly/clone/delegated
```

删除 `access` 并不意味着禁止 child action，而是所有 child 自动继承父节点当前已有的 environment 和工具。Harness 不再维护显式权限树。

## 7. 异常处理

异常处理只有一条策略：**把实际结果直接反馈给模型。**

### 7.1 解析失败

模型输出不是一个合法 tool call：

```json
{
  "model_output": "I think B001 is best",
  "execution_result": "No executable tool call was found"
}
```

该输出已经消耗一个 step。下一轮模型自己决定如何修正。

### 7.2 参数或未知工具错误

```json
{
  "model_output": {
    "name": "click",
    "arguments": {"target": 7}
  },
  "execution_result": "arguments.target must be a string"
}
```

Harness 不把它转换成 `ToolSchemaError`、`rejected` 或 repair prompt。

### 7.3 环境执行异常

```json
{
  "model_output": {
    "name": "search",
    "arguments": {"query": "blue bottle"}
  },
  "execution_result": "WebShop request timed out"
}
```

Harness 不自动重放。模型可以先观察当前 root 状态，再决定是否再次调用。

### 7.4 child 异常

child 的异常只作为 child 的普通最终结果返回父节点：

```json
{
  "result": null,
  "error": "child could not complete the analysis"
}
```

父模型自行决定重新分发、自己完成，还是结束。

如果在本轮模型调用之前无法取得 root observation 或无法构造 prompt，则没有模型输出可供反馈；这种情况只记录到外部 trace 并结束当前节点，不伪造一个模型决策，也不消耗 step。

### 7.5 长结果

发给模型的单个 `execution_result` 最多保留约 1000 token：

```text
<前 1000 token> [truncated after approximately 1000 tokens]
```

完整异常和完整结果可以保存在外部 trace，但不塞进下一轮 prompt。无法解析的超长原始模型输出也应按同样规则截断。

## 8. 数据结构

### AgentLimits

```python
@dataclass(frozen=True)
class AgentLimits:
    max_total_steps: int
    max_depth: int
```

### SubagentSpec

```python
@dataclass(frozen=True)
class SubagentSpec:
    task: str
    context: JSONValue
    expected_output: str | None = None
```

### SubagentResult

父模型只需要看到简短结果：

```python
{
    "result": JSONValue | None,
    "error": str | None,
}
```

`agent_id`、`parent_id`、`depth` 等调试信息可以保存在外部 trace，不进入 parent prompt。

## 9. Environment 最小接口

Environment 是任务树可选的共享资源：

```python
class Environment(Protocol):
    async def observe(self) -> JSONValue: ...
    def tools(self) -> Sequence[ToolSchema]: ...
    async def execute(self, call: ToolCall) -> Any: ...
    def is_done(self) -> bool: ...
    async def close(self) -> None: ...
```

`reset`、`reward`、`success` 和 benchmark report 放在 `webshop_eval.py`，不进入通用 Agent loop。root 创建时如果传入 environment，所有 descendants 自动继承同一个引用；如果 root 没有 environment，整棵树都只使用 framework tools。

## 10. 需要修改的代码

当前实现与本目标的主要差异：

| 当前实现 | 问题 | 目标 |
| --- | --- | --- |
| AgentNode 强制接收 Environment | 纯分析任务也必须伪造环境 | root environment 可选；存在时由全树继承 |
| readonly child 看得到环境但没有 action tools | 节点能力不再真正同构 | child 与父节点拥有相同环境工具 |
| 模型可在 spawn 中选择 access | 模型可以请求 delegated/clone 等能力 | spawn schema 不含权限字段 |
| ResourceManager + delegated writer lease | 权限协议复杂且锁覆盖整个 child 生命周期 | 第一版顺序运行 child，删除整套写锁 |
| SubagentResult 包含大量调试字段 | 多 child 结果容易膨胀并被截断 | parent 只接收 result/error |
| broad model try 统一释放 step | 已收到畸形 provider 输出时可能不计 step | 区分无响应和已有原始响应 |
| 截断实际按 1000 bytes | 与 1000 token 不一致，中文截断过早 | tokenizer 或明确的近似 token 限制 |
| trace 强制读取 reward | benchmark 概念进入 core | reward 只在 evaluator |
| Python async 方法调用同步 subprocess | 阻塞其他 agent | 异步 subprocess 或默认关闭 |

### `curagent/core/agent.py`

1. `environment` 改为可选。
2. child 创建时自动传入 `self.environment`。
3. `_available_tools()` 只有 environment 不为空时才加入环境工具。
4. root、child、grandchild 使用相同的 `_available_tools()` 和 `_execute()`。
5. 删除 reward 写入通用 trace 的逻辑。
6. 保留 shared step、depth、spawn 和 finish。
7. 将 parse、schema、execute 异常统一变成 `execution_result`，继续 loop。

### `curagent/core/tools.py`

1. 删除 spawn schema 的 `access` 字段。
2. 保留严格的“一次输出最多一个 call”规则。
3. 未知工具和非法参数只返回可序列化错误文本。
4. 不生成额外错误分类对象。

### `curagent/core/scheduler.py`

1. 删除 AccessMode、delegated writer lease 和 `parent_access`。
2. 删除 clone 和环境权限分配逻辑。
3. 只保留 child id、深度检查和结果顺序。
4. `spawn_agent` 等待 child 完成；`spawn_agents` 按输入顺序逐个运行。
5. 后续确认需要并行时，再增加有界 worker queue 和具体环境的并发策略。

### `curagent/core/types.py`

1. 删除 `AccessMode`。
2. 删除 `SubagentSpec.access`。
3. 保留 `AgentLimits(max_total_steps, max_depth)`。
4. 将 `Observation` 改成普通 JSON 值，或只在 WebShop adapter 内保留结构化 Observation。
5. 将 `SubagentResult` 缩成 result/error，调试字段移到 trace。

### `curagent/core/trace.py`

1. trajectory 只保留 model output 和 execution result。
2. 1000 token 截断同时应用于 execution result 和 malformed raw output。
3. 外部 trace 可以保留完整值。
4. 删除通用 trace 对 `float reward` 的强制依赖。

### `curagent/models/openai_compatible.py`

1. 区分“连接失败、完全没有响应”和“已经收到但结构畸形的响应”。
2. 无响应时释放 step；已有原始响应时提交 step，并把解析问题作为普通结果反馈模型。
3. 保留 provider raw response 到外部 trace，但 prompt 侧受长度限制。

### `curagent/environments/base.py`

Environment 由 root 创建，并由所有 descendants 自动共享。Environment 本身不判断当前调用来自 root 还是 child。

### `curagent/environments/recode_webshop.py`

继续保留 WebShop-specific 的页面解析和动作翻译。root 和 child 都可以调用它；因为递归调度顺序运行，同一时刻只有一个节点操作环境。共享权限 lease、clone 和 delegated access 不再进入通用 core。

### `curagent/harness/webshop_eval.py`

负责 reset、reward、success、episode close 和 benchmark report。AgentNode 只负责完成任务循环。

## 11. 递归示例

root 当前页面 observation 为：

```text
搜索结果包含 B001、B002、B003
```

模型决定把“找出并打开正确商品”整个子任务交给 child：

```json
{
  "name": "spawn_agent",
  "arguments": {
    "task": "从候选商品中找出最符合要求的商品，并在 WebShop 中打开它",
    "context": {
      "instruction": "购买蓝色 32 oz 保温不锈钢水瓶",
      "observation": "B001... B002... B003..."
    },
    "expected_output": "打开正确商品后返回商品 id 和理由"
  }
}
```

child 继承当前 WebShop environment。它先分析 observation，然后自己调用：

```json
{"name":"click","arguments":{"target":"B001"}}
```

环境进入 B001 商品页面后，child 调用 `finish`：

```json
{
  "result": {
    "item_id": "B001",
    "reason": "符合颜色、容量和材质"
  },
  "error": null
}
```

父节点收到结果后重新 observe，直接看到当前已经位于 B001 商品页面。父节点可以继续选择 Blue、32 oz 和 buy，也可以把后续动作继续分配给另一个 child。

这里没有“child 是否允许 action”的额外协议。父模型通过任务决定委派范围，child 为完成任务自行决定是否调用环境工具。

## 12. 实施顺序

### 第一阶段：先改递归边界

1. 删除 spawn 的 access 参数。
2. child 自动继承 `self.environment` 和环境工具。
3. 增加测试：child 可以调用 search/click/buy，父节点随后能看到更新后的 observation。
4. 增加 root -> child -> grandchild 连续修改同一环境的测试。

### 第二阶段：改异常反馈

1. 所有解析、schema、执行异常都变成普通 execution result。
2. 每个有输出的模型响应都消耗一个 shared step。
3. 删除 repair/retry/effect/reconcile 分支。
4. 对结果和 malformed output 做约 1000 token 截断。

### 第三阶段：清理 core

1. 移除 AccessMode、ResourceManager、writer lease、clone 和 delegated write。
2. 移除通用 trace 的 reward 依赖。
3. environment 对任务树可选；存在时由所有节点继承。
4. `spawn_agents` 改为按顺序运行，删除 child concurrency 配置。
5. 修复 Python executor 的事件循环阻塞问题。

### 第四阶段：验证

1. root 直接完成 Mock WebShop。
2. root -> child -> grandchild 递归完成纯分析任务。
3. child 和 grandchild 可以执行环境 action。
4. 错误后模型可以继续多轮决策直到 shared step 用尽。
5. shared total steps 在递归树中不超限。
6. 第二个非 WebShop 的纯分析例子无需修改 AgentNode。

## 13. 验收标准

实现满足以下条件即可认为达标：

1. root 和 child 使用完全相同的 Agent loop。
2. 模型可以自行决定是否 spawn，以及 child 的 task/context。
3. child 自动继承父 environment 和 action tools，不需要显式 access。
4. child 修改环境后，父节点下一轮可以看到最新 observation。
5. 所有普通异常都直接进入下一轮 prompt。
6. 任意有模型输出的响应都消耗一个 shared step。
7. 任务树只受 `max_total_steps` 和 `max_depth` 限制。
8. 不需要为新增递归分析任务修改 core。

## 14. 最终设计判断

最简单且足够的方案是：

```text
模型决定递归
    -> spawn 只传 task/context
    -> child 自动继承相同 environment 和 tools
    -> child 自己决定是否执行 action 或继续递归
    -> child result 返回父节点
    -> 父节点从最新 environment 状态继续
    -> 所有异常作为普通结果反馈模型
```

不需要在第一版实现 readonly/writable、clone、delegated lease、effect 对账或复杂权限树。环境能力直接随递归节点继承，任务范围由父模型的 task/context 表达，Harness 不替模型判断这个 child 应不应该行动。
