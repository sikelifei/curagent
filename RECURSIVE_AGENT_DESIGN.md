# 通用递归 Agent 改造设计

状态：审阅稿，当前只描述设计，不修改 `/data2/zhangwenjian/agent/rlm`。

## 1. 目标

基于现有 `rlm` 的模型客户端、持久 Python REPL、变量、工具注入和迭代循环，在 `curagent` 中实现一个更小、更通用的递归 Agent。

核心原则是：**harness 只提供能力和机械执行，不替模型做决策。**

- 每个 Agent 都运行完全相同的循环，具有相同的 REPL、custom tools 和递归能力。
- 不再区分 planner、solver、RLM subagent 和普通 LLM subagent。
- 模型自行决定当前任务是直接分析、执行 REPL/custom tool、启动一个 subagent，还是并发启动多个 subagent。
- harness 不负责拆任务，不判断是否应该递归，不判断共享环境是否适合并发，也不自动改成串行。
- system prompt 只描述能力、接口和必要的并发常识，训练与评测负责让模型学会正确决策。
- 不再围绕长文本、强制 chunking 或“主模型只能规划”来设计。

参考代码保持只读。后续审阅通过后，建议在 `curagent` 中实现新包，而不是直接覆盖原始 `rlm`。

## 2. 现有实现中保留和删除的部分

### 2.1 保留

- `clients/` 中的模型 provider 适配和 usage 统计。
- 每个 Agent 独立、跨 step 持久的 Python namespace。
- 从模型输出提取 fenced `repl` 代码块并执行的流程。
- `print(...) -> stdout -> 下一轮 user observation` 的交互闭环。
- `context`、`SHOW_VARS()` 和 `answer = {"content": "", "ready": False}`。
- custom tool 的现有注册形式：普通值，或 `{"tool": value, "description": "..."}`。
- custom tool 名称冲突校验、名称和描述注入 system message。
- 清理、日志、usage、超时和取消等基础机制。

现有链路已经接近目标：

```text
模型输出 repl 代码块
  -> 提取 Python 代码
  -> exec(code, namespace, namespace)
  -> 调用 namespace 中的 observe()/act()/其他 custom tool
  -> print() 写入本轮 stdout
  -> stdout 作为 user observation 追加到完整消息历史
  -> 下一次模型生成
```

### 2.2 删除

- `llm_query`、`llm_query_batched`。
- `rlm_query`、`rlm_query_batched`。
- `other_backends` 和根据 depth 切换模型的行为。
- `custom_sub_tools`。parent 和 child 使用同一个 tool registry。
- `orchestrator=True` 和 `ORCHESTRATOR_ADDENDUM`。
- “必须先读长文本”“必须拆分上下文”“sub-LLM 能处理多少字符”等长文本特化提示。
- 递归到叶子后退化为一次普通 LLM completion 的行为。
- 每轮重复拼接原始问题，或重复插入大段操作说明。

### 2.3 首版暂不保留

为了先把统一递归语义做正确，首版建议只承诺本地 Python REPL：

- 暂不实现 Docker、Modal、Prime、Daytona、E2B 等多套行为不一致的环境。
- 暂不实现 compaction 和 persistent multi-conversation。
- 暂不实现 provider native tool calling。所有 Agent 行为统一走 `repl` 代码块。

后续如需远程 REPL，应在同一接口下增加 adapter，不能让 system prompt 宣称一个运行时不存在的工具。

## 3. 对模型暴露的唯一递归接口

REPL 中只增加以下两个递归函数：

```python
def spawn_subagent(task: str, context: Any | None = None) -> str:
    """同步启动一个同构 child Agent，只返回 child 的最终文本结果。"""


def spawn_subagents(
    requests: list[dict[str, Any]],
) -> list[str]:
    """并发启动多个同构 child Agent，结果按输入顺序返回。"""
```

`spawn_subagents` 的输入格式固定为：

```python
requests = [
    {"task": "分析候选 A", "context": candidate_a},
    {"task": "分析候选 B", "context": candidate_b},
]
results = spawn_subagents(requests)
```

接口刻意不提供以下参数：

- `model`：child 使用与 parent 相同的模型配置。
- `agent_type`、`planner`、`solver`：不存在角色类型。
- `access`、`readonly`、`parallel_safe`：harness 不替模型判断环境语义。
- `parallel`：单个接口同步执行，批量接口就是并发执行。
- parent history 或 namespace：child 只能看到明确传入的 `task` 和 `context`。

具体契约：

- `task` 必须是非空字符串。
- `context` 可以是任意可被 `copy.deepcopy` 的值。启动 child 前创建独立深拷贝，并放入 child REPL 的 `context` 变量。
- parent 和 child 对各自 `context` 的修改互不影响；需要共享的实时环境对象应注册成 custom tool，而不是塞入 `context`。
- child 不自动继承 parent 的消息历史、普通变量或中间推理。
- child 继承同一 system prompt、模型配置、custom tools 和两个 spawn 函数。
- 每个 child 有独立消息历史、独立 `answer` 和独立 Python namespace。
- custom tool callable 仍是同一注册对象。因此绑定到同一个 benchmark env 的工具会访问同一环境状态。
- `spawn_subagents([])` 返回 `[]`；多个结果严格保持输入顺序。
- parent 只看到 child 的最终字符串，不看到 child 轨迹、step 或 usage。完整信息仅进入内部 trace。
- child 失败时返回简短错误字符串，例如 `Error: subagent timed out`，不把内部异常堆栈塞进 parent prompt。

## 4. 并发语义：由模型判断，不由 harness 干预

`spawn_subagents` 对合法请求直接并发执行。harness 不检查 custom tool 名称，不识别任务类型，不加环境锁，不申请 lease，不自动 clone，也不因为环境有状态而降级成串行。

模型需要从 task、context 和工具描述中自行判断：

- 独立文本处理、候选比较、互不依赖的分析可以使用 `spawn_subagents`。
- 多个 child 若同时操作同一个有状态交互环境，可能让 observation 和 action 相互交错并造成状态不一致。
- 这类共享环境任务应由模型选择自己串行操作，或只启动一个 `spawn_subagent`。

这是模型策略，不是 harness policy。对应行为应通过训练样本和评测塑造，而不是增加运行时兜底。

实现仍可保留 `max_concurrent_subagents` 作为纯机器资源上限。它只作用于当前这一次 `spawn_subagents` 调用，多出的请求在该 batch 内排队；它不分析任务，也不改变输入顺序或 Agent 决策。嵌套 batch 使用各自的 worker pool，首版不实现整棵递归树的全局调度器。

## 5. 统一 Agent 生命周期

公开入口建议为：

```python
agent = RecursiveAgent(
    backend="openai",
    backend_kwargs={...},
    tools=tools,
    max_steps=20,
    max_depth=4,
    max_concurrent_subagents=4,
    max_observation_chars=8000,
    termination_check=env.status,  # 可选
)

result = agent.run(task=instruction, context=initial_context)
```

其中：

```python
@dataclass
class AgentResult:
    answer: str
    status: Literal["completed", "forced_final", "environment_done"]
    steps: int
    usage: UsageSummary
    trace: AgentTrace | None = None
```

REPL 内部的 `spawn_subagent(s)` 只取 `AgentResult.answer` 返回给 parent。

### 5.1 每个 Agent 都运行同一个循环

```text
创建模型 client 和独立 REPL namespace
  -> 构造 [system, initial user task]
  -> 最多执行 max_steps 个工作 step
       -> 使用完整 message history 调用模型
       -> 追加 assistant response
       -> 顺序执行 response 中的 repl blocks
       -> 代码可以本地计算、调用 custom tool 或 spawn subagent
       -> printed output 或执行错误经过字符上限处理后作为 user observation 追加历史
       -> answer.ready 或 env done 时结束
  -> 未结束则额外执行一次 forced-final completion
  -> 清理当前 Agent 资源并返回 AgentResult
```

parent 和 child 的唯一区别是树中的 `depth` 和各自的 task/context，执行代码完全相同。

### 5.2 一个 step 的定义

- 一次正常模型 completion 计为当前 Agent 的一个 step。
- 同一 response 中执行一个或多个 `repl` block 不额外计 step。
- child 自己有同样的 `max_steps`，其 step 不计入 parent 的本地 step。
- `max_steps` 耗尽后的 forced-final completion 是额外一次收尾调用，不计工作 step，但必须计入 usage 和总耗时。
- `max_steps` 必须大于 0，初始化时校验。

### 5.3 递归边界

`max_depth` 只是防止无限递归的硬资源边界，不改变 Agent 类型：

- 所有 depth 的 Agent 都能看到同样的两个 spawn 函数。
- 达到 `max_depth` 后再次调用 spawn，函数返回清晰的 limit error。
- 当前 Agent 仍可继续本地分析、调用 custom tools 和提交答案。
- 绝不在叶子节点偷偷退化成普通一次性 LLM，因为这会破坏能力边界一致性。

## 6. 消息与 prompt 设计

### 6.1 System prompt

system prompt 对 root 和所有递归 child 完全相同，不指定 planner 身份，不强制拆任务。它由通用能力、环境 addendum 和注册工具描述组成：

```text
You are a general recursive agent. Complete the task using your own reasoning,
the persistent Python REPL, the available tools, and subagents. Decide for
yourself whether to solve directly, execute code or tools, or delegate work.

Run Python by writing ```repl``` blocks. Variables persist across steps. Only
printed stdout is returned as an observation, so use print(...) when you need
to inspect a value.

Built-ins:
- spawn_subagent(task, context=None) -> str: run one child agent with the same
  capabilities and return only its final result. It can return an Error string
  if a recursion or resource limit prevents the child from running.
- spawn_subagents(requests) -> list[str]: run independent child requests
  concurrently and return results in input order. Each request is a dict with
  a "task" and an optional "context". Independent text or analysis tasks can
  be parallelized. Do not let multiple child agents operate the same stateful
  environment concurrently, because their actions can make the environment
  state inconsistent.
- SHOW_VARS() -> str: list persistent REPL variables.
- answer: set answer["content"] and then answer["ready"] = True when finished.

Every agent has the same capabilities. A newly delegated agent starts its own
message history and receives only its delegated task and a private copy of the
context passed to it, not its caller's messages or REPL variables. Any agent
may delegate recursively within the configured limits.

{custom_tools}
```

这里的并发说明只告诉模型事实，不触发任何 harness 检查或自动调度。

### 6.2 初始 user message

root 的数据集任务只显式出现一次：

```text
Task:
{task}
```

child 使用相同 system prompt，但第一条 user message 标明任务来源和 context 语义：

```text
Delegated task:
{task}

This task was supplied by another agent. A private copy of the context it
supplied is available in the REPL variable `context`.
```

- `task` 不再同时作为隐藏的 REPL context。
- `context` 无论大小都只放进 REPL，不在每轮复制。
- child 的 `task` 进入 child 的初始 user message，显式 context 值放进其独立 REPL。
- root 和 child 的能力、环境 prompt、工具和循环完全一致；差异只在首次任务消息。
- 不再追加字符数、长文本容量、chunk size 等 metadata。

### 6.3 后续消息

后续依靠完整消息历史保持原始 task 可见，不再次拼接 task。

- 有代码输出或执行错误时追加：`REPL output:\n{output_or_error}`。
- 代码执行但无输出时追加：`REPL output:\n(no output)`。
- response 没有 `repl` block 时，只追加一个最短的 `Continue.`，让模型可以先在文本中分析再自行决定下一步。
- 不再每轮追加 `Turn i/n`、任务复述或操作建议。

模型可见的单轮 observation 默认限制为 8,000 字符，由
`max_observation_chars` 配置。超限时 harness 保留输出首尾、截断标记和错误摘要。
完整 stdout/error 仍进入 execution trace；trace 另外记录模型实际看到的
`model_observation` 与 `observation_truncated`。该限制按字符计算，不假设不同
provider 使用相同 tokenizer；设置为 `None` 可关闭。

### 6.4 强制最终答案

当 `max_steps` 用尽且 `answer["ready"]` 仍未设置时，追加一条 **user** 消息：

```text
No working steps remain. Return the best final answer now as plain text.
Do not use the REPL, tools, or subagents.
```

然后额外调用同一个模型一次：

- 这次 response 直接作为最终答案。
- 不提取或执行其中的 `repl` block。
- 不要求它再次设置 `answer["ready"]`。
- 完整 history 中仍有初始 task，因此不需要重贴问题。

现有实现把这条要求错误地放成 `assistant` role，新实现必须使用 `user` role。

## 7. REPL 与 custom tools

### 7.1 单一持久 namespace

建议使用一个 namespace，而不是容易产生覆盖差异的 globals/locals 双字典：

```python
namespace = {
    "__builtins__": allowed_builtins,
    "context": copied_context,
    "answer": {"content": "", "ready": False},
    "SHOW_VARS": show_vars,
    "spawn_subagent": spawn_one,
    "spawn_subagents": spawn_many,
    **registered_tools,
}

exec(code, namespace, namespace)
```

每个 code block 后先读取并锁定有效的 `answer["ready"]` 和 `answer["content"]`。若尚未结束，再恢复被删除或破坏的内置 spawn 函数、`answer` scaffold 和 custom tool 名称；不能清空一个仍然有效的 `answer` dict。普通变量继续保留。

`answer["ready"]` 只在严格等于 `True` 时结束。若一个 response 有多个 code block，按顺序执行；某个 block 一旦提交答案或环境结束，立即停止后续 block，避免结束后继续产生 action。

### 7.2 stdout

为了让并发 child 的 REPL 输出不互相截获，首版不应像现有 LocalREPL 一样反复替换进程级 `sys.stdout` 或 `os.chdir`。

最小实现是在每次 exec 前向该 namespace 注入一个绑定本地 buffer 的 `print`，捕获模型直接调用的 `print(...)`，并把未捕获异常格式化成 error observation。这样各 Agent 的 Python namespace 和 observation buffer 相互独立。它是逻辑隔离，不是安全 sandbox；导入库直接写进程级 stdout/stderr 的行为首版不保证捕获。

### 7.3 Tool 注册

沿用现有简单格式：

```python
tools = {
    "observe": {
        "tool": env.observe,
        "description": "Return the current environment observation. Print the result.",
    },
    "act": {
        "tool": env.act,
        "description": "Execute one environment action and return the new observation.",
    },
    "target": {
        "tool": sample.target,
        "description": "Target information for the current sample.",
    },
}
```

- callable 放入 REPL global namespace，可直接调用 `observe()`。
- 非 callable 作为普通全局数据使用。
- 工具名必须是合法 Python identifier，且不能覆盖内置 scaffold。
- system prompt 中只注入工具名称和 description，不注入 callable repr、secret 或大对象内容。
- parent 和所有 child 注册完全相同的 tools，不再有 `custom_sub_tools`。
- harness 不根据工具名称或 description 推断它是否有状态、是否可并发。

## 8. Benchmark 接入

每条 benchmark 数据创建一个独立任务运行：

```python
sample = benchmark.load(index)
env = benchmark.create_env(sample)

agent = RecursiveAgent(
    backend=...,
    backend_kwargs=...,
    tools={
        "observe": {"tool": env.observe, "description": "..."},
        "act": {"tool": env.act, "description": "..."},
    },
    termination_check=env.status,
)

result = agent.run(
    task=sample.instruction,
    context=sample.context,
)
```

`termination_check` 是可选的最小终止接口：

```python
@dataclass
class EnvironmentStatus:
    done: bool
    final_answer: str | None = None
    reason: str | None = None
```

每个 code block 后按以下顺序检查：

- 若 `answer["ready"] is True`，优先接受模型已提交的答案并结束。
- 否则 `done=False` 时正常进入下一 step。
- 否则 `done=True` 且有 `final_answer` 时直接结束。
- 否则 `done=True` 但没有 `final_answer` 时停止执行后续 code block，并走一次 forced-final completion。

这个接口只报告 benchmark 自己是否结束，不参与任务分解或并发判断。它只能在一个 code block 返回后生效；若环境需要禁止 terminal state 后的 action，应由 benchmark tool 自己拒绝。没有该接口时，只使用 `answer["ready"]` 和 `max_steps`。

不同数据项可由 benchmark runner 在外层并发；它们各自绑定不同 env。单条数据内部是否递归和并发，仍完全由模型决定。

## 9. 错误与限制语义

错误处理保持简单、可预测：

| 情况 | 行为 |
| --- | --- |
| 非法配置、非法 tool 名 | 启动前直接抛出配置异常 |
| Python 语法、变量或 custom tool 异常 | 捕获为本轮 error observation；超限时截断正文并优先保留错误摘要，让模型自行修正 |
| 单个 child 失败 | `spawn_subagent` 返回简短 `Error: ...` 字符串 |
| batch 中部分 child 失败 | 失败位置返回错误字符串，其他结果保留且整体仍按输入顺序 |
| 达到 `max_depth` | spawn 返回 limit error，当前 Agent 继续运行 |
| `max_steps` 耗尽 | 额外一次 forced-final completion |
| 模型调用失败 | 抛 `ModelCallError`，携带最近的文本 response（若有） |
| 共享 run timeout | 阻止新的 model/spawn 调用，尽力取消未开始的 child，并抛 `TimeoutExceededError` |
| 用户取消 | 设置 cancellation event，尽力取消未开始的 child，并在当前同步调用返回后清理 |

首版只需要以下限制：

- `max_steps`：每个 Agent 的工作 step 上限。
- `max_depth`：递归深度硬上限。
- `max_concurrent_subagents`：每次 batch 调用自己的物理 worker 上限，不是全树调度策略。
- `max_run_seconds`：root 和整棵递归树共享的 cooperative deadline。
- `max_observation_chars`：每个 Agent 每轮模型可见 REPL/tool feedback 的字符上限；不裁剪 trace 中的原始 execution 输出。

线程内已经运行的同步 Python、custom tool 或 provider 请求不能被可靠硬中止。timeout 和 cancel 在首版是 cooperative/best-effort 语义：在同步调用返回后检查并停止后续工作。若将来需要硬终止，必须改用可杀掉的独立进程。

token 和 dollar budget 先做 usage 统计，不在首版实现并发下容易产生竞态的硬预算。需要时再基于整棵 run 的统一计数增加，而不是给每个 child 复制一份看似独立的预算。

## 10. 内部结构建议

```text
curagent/
  recursive_agent/
    __init__.py
    agent.py          # RecursiveAgent.run、统一 step loop、child 创建
    config.py         # AgentConfig 与参数校验
    types.py          # AgentResult、AgentTrace、EnvironmentStatus、usage
    prompts.py        # 短 system prompt、initial user、forced-final user
    repl.py           # namespace、repl block 执行、stdout、answer
    tools.py          # tool 解析、校验、prompt 格式化和 namespace 注入
    clients/          # 从现有 rlm 复用并精简的模型客户端
    exceptions.py
  tests/
  examples/
```

不单独增加 planner、solver、scheduler、access policy 或 environment lock 模块。`spawn_subagents` 的线程池可以直接封装在 `agent.py` 或一个很小的内部 helper 中。

内部需要一个轻量 `RunContext`，只保存：

- root run id、parent/child trace 关系。
- 共享 deadline 和 cancellation event。
- usage 汇总。

`RunContext` 不读取 task，不分析工具，不决定是否并发，也不做环境权限控制。

## 11. 从现有 rlm 到新实现的映射

| 现有位置 | 处理 |
| --- | --- |
| `core/rlm.py` 的 completion loop | 精简为 `RecursiveAgent.run`，parent/child 共用 |
| `_subcall` | 改为统一 `_spawn_child`，永远运行完整 Agent loop |
| `_fallback_answer` | 删除 |
| `core/lm_handler.py` 的 raw LLM subcall 路由 | 删除；主 Agent 直接使用 client |
| `environments/local_repl.py` | 保留 namespace/变量思想，重写四种 query 为两个 spawn |
| `environments/base_env.py` 的 custom tool 解析 | 保留并收敛到 `tools.py` |
| `utils/prompts.py` | 用第 6 节短 prompt 全量替换 |
| `utils/parsing.py` | 保留 repl block 提取和 observation 格式化，修正 ready 后停止 |
| `RLMChatCompletion/RLMIteration` | 精简为 `AgentResult/AgentStep/AgentTrace` |
| `max_iterations` | 重命名为语义清楚的 `max_steps` |
| `completion(prompt, root_prompt)` | 改为 `run(task, context=None)` |

当前 `curagent` 配置里的 `planner` 概念也应在实现阶段改成单一 `model` 配置；不再单独配置 planner model 和 subagent model。

## 12. 测试与验收标准

### 12.1 Prompt 与历史

- task 只在初始 user message 中出现一次。
- 后续 step 不重复 task，只依靠完整 history。
- system prompt 不含 planner/orchestrator、长文本 chunking 或强制 delegation。
- custom tool 名称和 description 出现在 system message，工具值本身不泄露。
- forced-final 指令是 user role。

### 12.2 自主决策

- 模型可以完全不 spawn，直接分析和作答。
- 模型可以只操作 custom env tools。
- 模型可以调用一个或多个 child。
- harness 不因工具叫 `observe`、`act` 或绑定有状态对象而改变模型选择。
- `spawn_subagents` 不会自动串行化环境任务；是否避免该调用属于训练/eval 行为，不是 deterministic unit test。

### 12.3 递归一致性

- parent 和 child 的 system prompt、内置函数和 custom tools 集合相同。
- child task 正确进入 child 初始 user message。
- child 只收到显式 context，不会看到 parent history 或变量。
- 达到 depth 上限时不会调用 raw LLM fallback。
- nested child 仍可继续调用相同的两个 spawn 函数。

### 12.4 并发

- 多个 child 实际并发运行，而不是顺序调用。
- 返回顺序与 request 顺序一致，不受完成顺序影响。
- 一个 child 失败不会丢失其他 child 结果。
- `max_concurrent_subagents` 只限制单次 batch 的 worker 数，不重排结果。
- 不同 child 的 REPL namespace、`answer` 和 captured stdout 不串扰。

### 12.5 终止和错误

- `answer["ready"] is True` 后立即停止，且不执行后续 code block。
- 正常 ready 不产生额外模型调用。
- 未 ready 时恰好执行 `max_steps` 次，再额外生成一次最终答案。
- forced-final 的 usage 和耗时被统计。
- env done 有答案时直接返回，无答案时只生成一次最终答案。
- Python/tool 错误只作为 observation 返回，不触发额外的自动放弃策略。
- timeout、取消、模型异常和 child 局部失败遵守已声明的 best-effort 清理语义。

## 13. 实现顺序

审阅通过后建议按以下顺序实施：

1. 搭建新包，复用一个模型 client，完成短 prompt 和单 Agent step loop。
2. 实现持久本地 REPL、custom tool 全局注入、stdout observation 和 `answer`。
3. 实现同构 `spawn_subagent`，确认 child task/context 和能力一致。
4. 实现保序并发 `spawn_subagents`，不加入环境策略。
5. 实现 max step、max depth、forced-final、cooperative timeout、取消和错误 observation。
6. 增加 benchmark 示例：独立文本任务并发，以及共享交互环境由模型选择串行。
7. 完成 prompt、递归、并发、终止和异常测试后，再评估远程 REPL adapter。

## 14. 审阅时需要确认的决策

1. 新实现是否确认写入 `curagent/recursive_agent`，保持原始 `rlm` 只读。
2. 首版是否确认只支持本地 Python REPL，不同步移植七种旧环境。
3. `spawn_subagents` 是否确认采用 `[{"task": ..., "context": ...}]` 这一种输入格式。
4. 达到 `max_depth` 时是否确认返回错误给当前 Agent，而不是降级成一次普通 LLM。
5. 是否确认共享环境并发完全交给模型判断，harness 不加锁、不自动串行、不做权限或 clone 策略。
