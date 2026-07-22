# Curagent 架构介绍

## 1. 总体概括

Curagent 是一个“模型驱动、REPL 执行、递归委派”的通用 Agent 框架。

核心调用链：

    任务
      -> RecursiveAgent
      -> 构造 system prompt
      -> 调用大模型
      -> 解析 repl 代码
      -> 在持久化 Python REPL 中执行
      -> 将执行结果作为 observation
      -> 必要时创建 subagent
      -> 返回最终答案

核心实现文件：

- recursive_agent/agent.py
- recursive_agent/repl.py
- recursive_agent/prompts.py
- recursive_agent/types.py
- recursive_agent/envs/runner.py

核心特点：

1. root agent 和 child agent 使用同一套 Agent 逻辑。
2. 每个 Agent 拥有独立且持久化的 Python REPL。
3. 模型通过 repl 代码控制工具、变量和子 Agent。
4. 子 Agent 通过显式传入的 context 获取数据。
5. REPL 变量和消息历史不会自动跨 Agent 共享。
6. runtime 统一管理递归、并发、超时、取消和 trace。

## 2. RecursiveAgent 的构造

典型配置：

    agent = RecursiveAgent(
        backend="openai",
        backend_kwargs={...},
        tools={...},
        max_steps=20,
        max_depth=4,
        max_concurrent_subagents=4,
        max_run_seconds=300,
        max_observation_chars=8000,
    )

初始化位于 recursive_agent/agent.py:103，主要完成：

1. 校验 Agent 参数；
2. 解析工具并生成工具描述；
3. 组装 system prompt；
4. 保存环境终止检查函数；
5. 保存 REPL 安全配置；
6. 创建模型 client 工厂。

| 参数 | 含义 |
|---|---|
| backend | 模型后端 |
| tools | 注入 REPL 的自定义工具 |
| max_steps | 单个 Agent 的最大模型 step 数 |
| max_depth | 最大递归深度 |
| max_concurrent_subagents | child 最大并发数 |
| max_subagents_per_agent | 单个 Agent 最大直接 child 数 |
| max_run_seconds | root 和 child 共享的总时间 |
| max_observation_chars | observation 字符上限 |

调用 agent.run(task, context) 后，Agent 创建共享的 RunContext、root trace、模型 client、消息历史和 ReplSession，然后循环执行“模型调用 -> repl 解析 -> Python 执行 -> observation 反馈”。核心循环位于 recursive_agent/agent.py:289。

RunContext 保存 run_id、全局 deadline、取消事件、usage accumulator 和 trace 锁。root 与所有 child 共享同一个运行时预算。

## 3. Prompt 构造

最终 prompt 由三层组成：

    system prompt
      +-- 通用 Agent prompt
      +-- 环境专用 prompt
      +-- custom tools 描述

    user prompt
      +-- Task: 原始任务
      或
      +-- Delegated task: 子任务

    后续消息
      +-- REPL output
      +-- Continue.
      +-- forced-final 指令

通用 prompt 位于 recursive_agent/prompts.py:71，要求模型：

- 使用 repl block 执行 Python；
- 通过 print 输出希望观察的值；
- 使用持久化变量完成多轮工作；
- 使用 context 读取私有输入；
- 使用 custom tools；
- 在必要时创建 subagent；
- 使用 answer 标记完成。

可用内置能力：

    spawn_subagent(task, context=None)
    spawn_subagents(requests)
    SHOW_VARS()
    answer
    context

最终提交方式：

    answer["content"] = "最终答案"
    answer["ready"] = True

Prompt 要求模型把任务分成 DIRECT 或 DECOMPOSABLE。对于可拆分任务，通常需要创建 2 到 4 个独立请求、收集 child 报告并综合。模型负责拆分决策，runtime 负责强制资源限制。

环境通过 prompt_addendum 注入领域规则，runner 在 recursive_agent/envs/runner.py:71 负责接线。WebShop、Oolong、BrowseComp-Plus 和 Oolong-Synthetic 分别注入交互、长文本、检索和分块处理规则。

Root 的初始消息是 Task: 原始任务；child 的初始消息是 Delegated task: 子任务。Child 只接收 delegated task 和显式传入的 context，看不到父 Agent 的消息历史和 REPL 变量。

## 4. REPL 解析与执行

模型可以在输出中放置 repl 代码块，例如：

    x = 40
    print(x)

Agent 使用 find_repl_blocks(response) 解析，位置为 recursive_agent/agent.py:304。解析器支持 Markdown fenced repl block 和 XML 风格 repl block，大小写不敏感，支持多个 block，并按照出现顺序执行。没有 block 时发送 Continue.。

每个 Agent 创建一个 ReplSession。namespace 包含：

    __builtins__
    __name__
    context
    answer
    spawn_subagent
    spawn_subagents
    custom tools

namespace 是持久化的，因此第一轮创建的变量可以在第二轮继续使用。

普通赋值不会自动返回结果，必须显式调用 print。ReplSession.execute() 会重定向 print 到 StringIO，并生成 CodeExecutionTrace，记录代码、stdout、error、耗时和执行后变量名。

多个 block 的结果合并为 REPL output，再作为 user message 发回模型。如果某个 block 设置 answer["ready"] = True，Agent 立即返回，不再执行后续 block。

## 5. 错误处理

### Python 执行错误

REPL 捕获普通 Exception，并转换为：

    Error: ZeroDivisionError: division by zero

错误不会立即终止整个 Agent，而是作为 observation 返回，模型可以根据错误修正下一轮代码。完整错误仍写入 CodeExecutionTrace.error。

### Observation 截断

默认每轮 observation 上限为 8000 字符。长输出保留头部、尾部、截断标记、原始长度和 execution error。模型看到截断后的 model_observation，完整 stdout 仍保存在 trace 中，并通过 observation_truncated 标记。

### 模型调用错误

模型 client 抛异常时包装为 ModelCallError，并保存失败前的 last_response：

    ModelCallError(
        "Model call failed: ...",
        last_response=latest_response,
    )

不支持的 client 返回类型也会被视为模型调用错误。

### 超时和取消

runtime 在模型调用前后、REPL 执行后、child 创建前和并发调度过程中检查共享 RunContext。

    agent.cancel()

会设置取消事件并最终抛出 CancellationError；超时抛出 TimeoutExceededError。

### Environment 终止

环境可提供 termination_check=environment.status。每个 REPL block 后检查：

- 模型显式设置 answer["ready"] = True 时，模型答案优先；
- 环境 done 且有 final_answer 时，返回环境答案；
- 环境 done 但没有 final_answer 时，再执行一次 forced-final completion。

## 6. Subagent 机制

Subagent 不是特殊类，而是再次调用 _run_agent(...)。Child 与 root 使用相同的模型配置、system prompt、REPL、custom tools 和递归接口，但拥有新的 depth、消息历史、REPL namespace、context 和 trace。

递归树：

    root Agent
    ├── child Agent 1
    │   ├── grandchild Agent 1
    │   └── grandchild Agent 2
    └── child Agent 2

单个 child：

    child = spawn_subagent(
        "Analyze candidate A",
        {"candidate": "A"},
    )

child 最终返回 child.answer 字符串，完整 trace 挂到父 trace 的 children 字段。

批量 child：

    reports = spawn_subagents([
        {"task": "Analyze A", "context": {"candidate": "A"}},
        {"task": "Analyze B", "context": {"candidate": "B"}},
    ])

runtime 使用 ThreadPoolExecutor 并发执行，返回顺序与输入顺序一致。单个 child 失败不会自动导致其他 child 失败，而是返回 Error: subagent ...。

runtime 不会自动给共享环境加锁，也不会把并发操作改成串行，所以批量并发适合独立、只读任务。

资源限制：

| 参数 | 含义 |
|---|---|
| max_depth | 最深递归层数 |
| max_concurrent_subagents | 同时运行的 child 数 |
| max_subagents_per_agent | 单个 Agent 的直接 child 数 |
| max_steps | 单个 Agent 的模型 step 数 |
| max_run_seconds | 整个递归树共享的总时间 |

如果 max_depth=2，则 root depth=0、child depth=1、grandchild depth=2。超过限制返回 Error: maximum recursion depth reached。

## 7. context 如何传递

### 7.1 context 不是 prompt，也不是消息历史

调用：

    agent.run(
        task="分析问题",
        context={"data": "..."},
    )

context 不会自动拼接到 user prompt，也不会自动作为模型消息发送。它会进入 Agent 的 REPL 变量 context，模型需要通过：

    print(context)
    print(context["data"])

访问。

三个通道不同：

    task     = 显式文本输入
    context  = REPL 私有数据输入
    messages = 模型对话历史

### 7.2 Root context

环境 runner 首先复制环境 context：

    initial_context = copy.deepcopy(environment.context)

然后调用 agent.run(task=task_prompt, context=initial_context)。RecursiveAgent.run() 再次执行：

    private_context = copy.deepcopy(context)

传递链：

    Environment._context
      -> deepcopy
      -> EnvironmentRunResult.initial_context
      -> deepcopy
      -> Root Agent context
      -> Root ReplSession.namespace["context"]

因此环境的原始 context 不会直接暴露给 Agent 修改。

### 7.3 Root REPL 中的 context

Root 创建 ReplSession 时将 context 放入 namespace。多个 step 之间可以继续读取和修改：

    context["counter"] = 1
    print(context["counter"])

### 7.4 默认情况下 child 不继承 parent context

如果模型写：

    child = spawn_subagent("分析子问题")

默认参数是 context=None，因此 child 收到的是 None，不是父 Agent 的 context。

Child 不会自动获得父 Agent 的 context、REPL 变量、消息历史和已执行的 Python 代码。

### 7.5 显式传递 child context

如果 child 需要数据，父 Agent 必须显式传递：

    child_context = {
        "candidate": "A",
        "evidence": evidence_a,
    }
    child = spawn_subagent("分析 candidate A", child_context)

批量传递：

    reports = spawn_subagents([
        {"task": "分析 candidate A", "context": {"candidate": "A"}},
        {"task": "分析 candidate B", "context": {"candidate": "B"}},
    ])

runtime 创建 child 前执行 child_context = copy.deepcopy(context)，所以 child 修改自己的 context 不会反向修改父 Agent 的 context。

例如父 Agent 有 context=[1]，child 显式获得该 context 后执行 context.append(9)，child 看到 [1, 9]，父 Agent 仍看到 [1]。Child 也访问不到父 Agent 创建的其他 REPL 变量。

批量 child 的每个 context 独立复制。但如果 context 内部包含外部共享对象，或者 custom tool 访问共享环境，仍可能发生外部状态冲突。因此 context 隔离不等于外部资源隔离。

## 8. context 与 Environment 的区别

环境中的 context 通常是初始只读快照，不是整个环境状态本身。

例如 WebShop context 包含环境名、实例编号、购物指令、初始 observation 和初始合法动作；实时状态则通过 observe()、act(action) 和 available_actions() 获得。

因此：

    context = 初始任务和初始数据
    tools   = 操作或查询变化中的环境
    status  = 判断环境是否终止

Oolong 的 context 可以包含完整长文本 context["context_window_text"]。BrowseComp-Plus 的 context 主要包含 query 信息，搜索结果由 search(query) 获取。

## 9. 最终状态与 Trace

三种最终状态：

- completed：模型在 REPL 中设置 answer["ready"] = True；
- environment_done：环境返回 done=True 和 final_answer；
- forced_final：达到最大 step，或环境结束但没有 final_answer，额外调用一次模型生成纯文本答案。

AgentTrace 记录：

- agent_id、parent_id、depth、task；
- system_prompt；
- 每个 step 的模型 response；
- 每个 REPL block 的代码、stdout、error 和变量；
- observation 及其截断状态；
- child trace；
- usage、status、answer、error 和耗时。

因此一次递归运行可以还原成完整的执行树，并统计 root 和 child 的模型调用次数、输入 token、输出 token 以及 cost。

## 10. 汇报总结稿

> Curagent 采用统一的递归 Agent 架构。Root Agent 和所有 Child Agent 使用相同的模型接口、system prompt、持久化 Python REPL、custom tools 和递归能力。模型通过输出 repl 代码与运行时交互，runtime 负责解析代码、执行代码、捕获 stdout 和异常，并将结果作为下一轮 observation 返回给模型。每个 Agent 都有独立的消息历史和 REPL namespace，子 Agent 不会自动继承父 Agent 的变量和消息，只会接收父 Agent 显式传递、并经过 deepcopy 的 context。这样既保证了递归 Agent 之间的隔离，又允许 root 按需把长文本、任务分片和中间数据传给 child。运行时通过递归深度、并发数、直接子 Agent 数量、最大 step、全局超时和 observation 截断控制资源，并使用 AgentTrace 记录完整的执行树、模型调用、REPL 执行、错误、子 Agent 和 token usage。整体上，模型负责规划和综合，REPL 负责工作记忆和工具调用，runtime 负责执行、安全边界和生命周期管理。

---

# 附录：各部分 Agent 提示词中文翻译

本附录对应当前源码中的实际提示词。代码中的函数名、变量名、字段名和输出格式保留英文，因为它们必须与 runtime 接口一致。

## A. 通用 SYSTEM_PROMPT

源码：recursive_agent/prompts.py 的 SYSTEM_PROMPT。

### 身份和基本目标

你是一个通用递归 Agent。请使用推理、持久化 Python REPL、可用工具以及在有帮助时使用的 subagent，完成初始用户消息中的任务。

### REPL 规则

- 在 repl 代码块中运行 Python。
- 变量会跨多个 step 持久化。
- 只有打印出来的 stdout 会返回给你；需要查看值时必须使用 print(...)。

### 内置能力

- spawn_subagent(task, context=None) -> str：运行一个 child Agent，并返回它的最终结果。
- spawn_subagents(requests) -> list[str]：并发运行相互独立的 child 请求，并按照输入顺序返回结果。每个请求包含 task，可选 context。
- SHOW_VARS() -> str：列出当前持久化 REPL 变量。
- answer：完成任务时，先设置 answer["content"]，再设置 answer["ready"] = True。

### context 和 child 隔离

REPL 变量 context 保存启动当前 Agent 时传入的私有 context，也可能是 None。

新创建的 child 只会收到：

- 它自己的 delegated task；
- 父 Agent 显式传入的 context 的私有副本。

Child 不会收到调用方的消息历史或 REPL 变量。注册的工具和环境说明对每个 Agent 都可用。

部分工具可能访问共享外部状态，因此修改有状态环境时必须协调，不能让并发 child 产生冲突。

### 任务分类

在开始解决前，先分类：

1. DIRECT

   适用于短计算、简单解释或单个明确动作。直接完成，不要无谓委派。

2. DECOMPOSABLE

   适用于包含至少两个真正独立子任务，并且并行分析有可能改善最终答案的任务。

对于 DECOMPOSABLE 任务，第一个 REPL block 必须：

1. 创建 2 到 4 个相互独立的请求；
2. 使用 spawn_subagents(requests)；
3. 收集所有 child 结果；
4. 综合结果后给出最终答案。

不要只在普通文本中描述委派计划，必须在 REPL block 中实际执行委派。

只有独立或只读工作可以并行。会修改同一个环境或外部状态的操作必须顺序执行。

每个 worker 只解决自己负责的子任务，并返回简洁、自包含的报告。只有当 worker 自己的任务仍然包含多个独立子任务且继续委派确实有收益时，才允许继续递归。

如果 child 返回错误、不完整结果或相互冲突的结论，继续使用自己的推理和可用工具处理，不要盲目相信 child。

## B. 通用初始消息和 forced-final

### Root 初始消息

    Task:
    {task}

### Child 初始消息

    Delegated task:
    {task}

    该任务由另一个 Agent 提供。它传入的 context 的私有副本位于 REPL 变量 context 中，也可能是 None。调用方的消息历史和 REPL 变量不可用，除非它们被明确写入 task 或 context。你拥有与其他 Agent 相同的工具、REPL 和委派能力，并需要向调用方返回自包含的结果。

### 通用 forced-final

    不再有工作步骤。现在以普通文本返回最佳最终答案。
    不要使用 REPL、工具或 subagent。

## C. WebShop 环境提示词

源码：recursive_agent/envs/webshop/prompts.py。

### 环境和工具使用

- WebShop 工具已经注册为 REPL 全局变量。直接调用，不要 import WebShop、ReCode 或旧版辅助模块。
- 选择动作前先调用 observe()。
- 只能使用当前 observe() 返回的 valid_actions。
- valid_actions 中的 search[keywords] 是模板，不是字面动作。必须把 keywords 替换成实际搜索词。
- 不要执行 act("search[keywords]")，而应该执行类似 act("search[dip powder kit gentle nude]")。
- click 目标必须完全复制当前 observation 中的文本。

### 商品约束和导航

从购物指令中提取所有硬约束：

- 产品类型；
- 数量；
- 包装或件数；
- 尺寸；
- 颜色；
- 材质；
- 兼容性；
- 价格。

在搜索结果页先比较可见候选，再点击。在商品页先选择所有必需的可见选项，再点击 Buy Now。

只有当某个必需属性不清楚时，才打开 Description 或 Features。

实时环境动作必须串行执行。动作报错后重新 observe，不要重复使用已经过期的动作。没有新证据时，不要在搜索、Back、Next 之间来回跳转。最多尝试一次备用搜索，然后选择当前可见的最佳候选。

### WebShop 中的递归委派

当有多个可见候选或多个独立约束需要分析时，可以把 observe() 的副本传给 spawn_subagents。

每个请求必须明确 child 做的是：

- snapshot analysis：只分析快照；
- live environment operation：操作实时环境。

快照分析 child：

- 不得调用 act；
- 返回候选商品；
- 返回已匹配和缺失的要求；
- 返回证据；
- 返回一个当前合法的推荐动作。

实时操作 child：

- 根据需要调用 observe() 和 act()；
- 每个动作前立即检查 valid_actions；
- 一次只做一个改变状态的调用；
- 返回操作后的环境状态。

Agent 可以继续递归，但要避免重复分析同一个 observation 或形成委派循环。除非已经明确协调，否则多个 Agent 不能并发操作同一个 session。

### WebShop 示例含义

并行候选分析的示例流程是：

1. root 调用 state = observe()；
2. 把同一个只读 state 副本分别传给 candidate A 和 candidate B；
3. child 只分析快照并返回 matched、missing、evidence 和推荐动作；
4. root 打印并综合结果。

普通导航流程是：

1. observe；
2. act("search[实际搜索词]")；
3. 检查当前结果；
4. act("click[当前可见的候选]")；
5. 选择必要选项；
6. act("click[buy now]")。

示例中的搜索词、商品名、点击标签和选项标签只是示意，必须替换为当前 observation 中的值。

### WebShop task prompt

完成本次 WebShop 购物 episode。

购物指令是：

    {instruction}

使用 observe() 检查当前页面和合法动作。每次只执行一个合法的 act(action)，打印动作结果，并持续执行，直到 WebShop 在 click[Buy Now] 后进入终态，或者达到环境 step limit。

在环境进入终态之前，不得声称任务完成。委派是可选的，只有能增加有效工作时才使用。

### WebShop forced-final

不再有工作步骤。现在以简洁的普通文本返回本次 WebShop 购物 episode 的状态，并说明请求商品是否成功购买。

实际购买动作是 act("click[Buy Now]")；buy[...] 和 [FINISH] 都是无效动作。

只有环境真正进入 Buy Now 终态时才能声称成功。forced-final 阶段不得使用工具或 subagent，也不要使用 BrowseComp 的答案格式。

## D. Oolong-real 环境提示词

源码：recursive_agent/envs/oolong/prompts.py。

### 模式规则

- 初始 root Agent 拥有整个 episode。
- 如果初始 user 消息以 Delegated task: 开头，则当前 Agent 是 delegated agent。
- delegated agent 是只读的，禁止调用 observe、episode_report、submit_answer、spawn_subagent 和 spawn_subagents。
- delegated agent 必须通过最后一个 REPL block 设置 answer["content"] 和 answer["ready"] = True，返回一个紧凑 JSON 报告。
- 只有 root Agent 可以提交答案，并且必须恰好提交一次。

### 私有 context

context 是一个 dict：

- 完整 transcript 位于 context["context_window_text"]；
- 问题位于 context["question"]；
- child 会收到复制后的 chunk；
- child context 还包含 context["mapping"]、context["episode_index"] 和 context["chunk_index"]。

禁止打印完整 transcript。Root 必须把 transcript 切成不重叠、保留行边界的 chunk，每块大约不超过 12000 字符。这样可以降低 child context 长度，并让每个 auditor 的语义扫描更容易完成。

只能统计真正 episode block 中的文本。episode marker 必须是独立行：

    [START OF EPISODE]
    [END OF EPISODE]

前言中可能会内嵌提到这些字符串，所以必须使用多行正则识别独立 marker 行，不能简单使用 rfind、find 或 split(..., 1)。

如果 context 有多个 episode：

- episode N：选择第 N 个 episode；
- cumulative-through-episode-N：选择第 1 到 N 个 episode；
- across all、each episode、all episodes：选择全部 episode；
- this episode：只有一个 block 时选择该 block。

忽略问题、指令、mapping 说明、背景故事、广告以及选中 episode 之外的文本。

### Roll 统计规则

- 一个真实的骰子或检定结果只统计一次；
- 结果必须和 check、save、attack、skill、initiative、damage 或其他 D&D roll 绑定；
- encouragement 中的 roll、询问之前结果的 roll、重复叙述和普通 prose 不计数；
- modifier 或最终总值不能当作 natural die value；
- advantage/disadvantage 明确产生两个骰子结果时，两个都记录，但不要再次统计后续 check total；
- 结果不明确时标记 uncertain，不要猜；
- 角色问题使用 context["mapping"]，不能从不含 mapping 的 chunk 中自行推断；
- roll-type 使用 transcript 中明确写出的类型；
- natural-value 只统计明确的未修正骰子值。

### Child JSON 报告 schema

合法 child report 是 JSON object，包含：

- chunk_index；
- episode_index；
- rolls；
- spells；
- uncertain。

rolls 是紧凑对象，包含整数 total 和以下计数映射：

- by_player；
- by_character；
- by_type；
- by_value；
- by_natural_value。

total 统计真实骰子或检定结果，不统计单词 roll 的出现次数。还可以在问题要求特定值时提供 relevant_values，每项包含 speaker、character、type、value、natural_value 和 evidence。

spells 是紧凑对象，包含 total、by_player、by_character、by_name 以及按时间顺序排列的 ordered spell events。每个 ordered event 包含 name、speaker、character、level、base_level；未知 level 使用 JSON null。

uncertain 最多包含三个短片段，每段最多 120 字符。不能复制整段背景故事或 transcript。JSON 外不能有 prose。格式错误的报告不是证据，必须被 aggregate child 忽略。

### Root 执行手册

Root 执行三个逻辑阶段：

1. 初始化 chunks；
2. 扇出只读 child 报告；
3. 聚合并立即提交。

使用 spawn_subagent(task, context) 的位置参数形式。独立 chunk 使用 spawn_subagents([{"task": ..., "context": ...}])。

扇出返回后，不要打印、重新打开或手动调试单个报告。下一个 REPL block 必须直接聚合并提交。aggregate child 需要解析合法 JSON、保持 chunk 所有权不重叠、应用问题过滤，并返回包含整数或字符串 candidate 的 JSON。

aggregate child 同样是只读的，必须以 json.dumps(result) 写入 answer["content"] 并设置 ready。

Root 解析 candidate 后立即调用 submit_answer(r'\boxed{...}')。如果 candidate 不可用，应提交最佳的、明确有证据支持的候选，不要再次进入探索循环。不要返回只有 prose 的答案，也不要留下未闭合的 REPL block。

### Oolong task prompt

解决这个 Oolong-real benchmark example。

问题是：

    {question}

完整 transcript 只位于私有 REPL 变量 context["context_window_text"]。必须遵循 Oolong JSON child-report 工作流：不打印 transcript，切分 chunk，分发只读 auditor，然后立即聚合和提交。

不要返回 prose answer。聚合后 root 必须执行 submit_answer(r'\boxed{YOUR_ANSWER}')，以便官方 scorer 解析。

## E. Oolong-Synthetic 环境提示词

源码：recursive_agent/envs/oolong_synth/prompts.py。

### 数据和角色

完整数据集只在私有 REPL context 中可用。记录没有标签；不存在 labels 文件、隐藏 label 字段或可以直接揭示答案的工具。

这是一个扁平的 root/worker 工作流，并覆盖通用的任务路由规则，包括通用的 2 到 4 个子任务规则。所有 Agent 收到同一个 prompt，由 context["oolong_role"] 决定角色：

- root：选择直接处理或切成 64K chunk，合并结果，并且是唯一允许调用 submit_answer 的 Agent；
- worker：处理一个指定 chunk，不得继续委派，不得调用 submit_answer，通过 answer 返回一个 JSON 报告。

### 首次 REPL 操作

只使用可执行 repl block。每个 block 后等待真实 observation，不要自行编造 REPL output。

第一个 response 必须测量：

- context_window_text 的长度；
- 完整记录数量；
- 当前角色；
- 问题。

合法记录是一整行：

    Date: ... || User: ... || Instance: ...

不能拆分记录。第一条记录之前的文本是 dataset introduction。

### Root workflow

1. 使用测得的 context_chars。如果不超过 64K，在 root 中直接处理完整任务，并用连续、有边界的页面读取记录，避免 observation 截断。
2. 如果超过 64K，按记录边界、按原顺序贪心切 chunk。每个 chunk 中 len(row) + 1 的总和不超过 64K；超长单条记录可以单独成 chunk；每条记录必须恰好出现一次。
3. 使用 spawn_subagents 分批发送 chunk。task 只写“处理指定 Oolong-Synthetic chunk 并返回 JSON 报告”，不要把数据行放入 task 文本。worker context 只包含角色、唯一 chunk_id、expected_rows、该 chunk 的 rows、dataset_intro、全局 question 和 dataset。
4. 对每个 child 结果执行 json.loads；验证每个 chunk id 只出现一次、rows_seen == expected_rows，并检查问题需要的 labels 或 grouping keys。只对缺失或格式错误的 chunk 重试。合并验证后的结果，完成排名、比较和算术，并恰好调用一次 submit_answer(...)。

### Worker workflow

Worker 必须逐条读取自己的记录。如果一个 observation 放不下，就缩小页面并只重读该页。

分类必须来自逐条阅读 Instance 的语义，不能使用关键词、正则、词表、label 名称匹配或猜测类别平衡。Python 可以解析精确的 Date/User 元数据，但 label 必须是 Agent 阅读后明确赋予的。

Worker 对自己的 rows 应用问题中的 Date/User 过滤，并返回紧凑 JSON：

- chunk_id；
- rows_seen；
- counts；
- totals。

所有候选 label 都要包含，即使数量为零。rows_seen 必须等于 expected_rows。counts 用于可加的局部结果，totals 用于总体数量、分母、before/after 或比例问题。

Worker 设置 answer["content"] = json.dumps(report) 和 answer["ready"] = True。不得返回 prose 或 Markdown，不得调用 submit_answer，也不得创建另一个 Agent。

### 语义分类规则

标签只能来自阅读每个 Instance：

- trec_coarse 的 abbreviation：缩写或其展开；
- entity：具体对象、组织、产品、语言、事件、动物或物质；
- description 或 abstract concept：定义、原因、方式、解释、目的或含义；
- human being：个人或群体；
- location：地点；
- numeric value：数量、金额、日期、年龄、距离、价格、时长、百分比或其他数字。

Root 不得用关键词分类器或猜测替代失败的 worker 覆盖。只有 root 提交全局答案，worker 只报告自己的 chunk。

### Oolong-Synthetic task prompt

解决这个 Oolong-Synthetic benchmark task。

问题是：

    {question}

完整的无标签数据只位于私有 REPL 变量 context["context_window_text"]。遵循环境规定的 64K routing workflow，并以问题要求的精确格式调用 submit_answer(...)。

### Oolong-Synthetic forced-final

不再有工作步骤。以问题要求的 answer format 返回 Oolong-Synthetic 的最佳最终答案。不要添加分析、Markdown fence 或无依据的备选答案。不要再使用工具、subagent 或 submit_answer。

## F. BrowseComp-Plus 环境提示词

源码：recursive_agent/envs/browsecomp_plus/prompts.py。

### 基本检索规则

使用固定的 BrowseComp-Plus 语料库和注册的 search(query) 工具回答问题。该工具返回官方 Top-5 BM25 snippets。

查询应当短而具体，包含有区分度的短语、姓名、日期、组织、地点或标题。后续查询应利用之前结果中发现的新实体，而不是重复几乎相同的改写。

每个检索结果只是候选证据。必须用重要线索验证领先答案，明确计算问题要求的日期或数字关系，并引用支持证据的 document ID。所有 Agent 共享同一个 search budget。

工具只能在 Python 风格的 repl block 中执行。检索时必须实际调用 search 并等待结果，不能只在普通文本中描述搜索。

最终输出必须恰好包含三行：

    Explanation: 带引用的简短解释，例如 [12345]
    Exact Answer: 最短且无歧义的答案
    Confidence: 0-100%

最终答案也必须通过 repl block 设置 answer["content"] 和 answer["ready"]。题目和检索结果是唯一可用的 benchmark 信息，gold answer、labels、qrels 和 evaluator 数据不可用。

### 检索和递归策略

每个 Agent、每个递归深度，都必须在第一次 search 前判断当前任务是否可拆分。这是内部路由决策，不要花 search call 去解释，也不要输出形式化约束表。

当任务包含至少两个可以独立调查的 clue、evidence branch、event、document、claim、time period、organization 或 candidate family 时拆分。后续搜索必须依赖前一搜索发现的实体时，仍然属于同一条链接证据链，不应强行拆分。

如果可拆分：

1. 搜索前创建 2 到 4 个互不重叠的 request；
2. 让这些 request 覆盖所有独立分支；
3. 每个 request 指定一个独立的局部证据输出；
4. 在第一个 REPL block 中调用 spawn_subagents 并收集报告；
5. 不要重复执行 child 已经负责的搜索。

如果没有两个有价值的独立分支，就直接解决链接任务。不能只为了把原问题外包给一个 child，或重复原问题而创建 child。

每个 delegated task 必须严格小于 parent task：

- 只有一个调查目标；
- 只包含该目标所需的线索、实体、约束、lead 和排除项；
- 排除分配给 sibling 的约束；
- 说明预期的局部输出；
- 要求返回支持证据、反驳证据、未解决点、document IDs 和已经尝试的 query family。

如果 request 复制完整原问题、要求相同最终答案、重复 parent 的目标或 query plan、与 sibling 重叠，或者不能说明会增加什么独立证据，则 request 无效，委派前必须重写。

Child 在第一次 search 前也必须做同样的拆分判断。如果局部任务仍有多个独立分支，就继续拆分；否则直接解决。不能把自己的完整目标或 query plan 原封不动传给下一级。

### Search discipline

- 从最有区分度的 source-like clue 开始，例如精确短语、罕见事件、日期、标题、组织、专名或罕见术语组合；
- 一次只发起一个 search call，读取结果后再决定下一个查询；
- 新查询必须引入新实体、新 source phrase、新日期、新候选或真正不同的检索路径；
- 同一查询的近似改写最多两次；
- 连续检索没有产生新候选、新文档或新关键词时，停止该路径并返回 unresolved；
- 不要直接搜索一个计算后的关系，先检索基础日期或数值，再自行计算。

Delegated report 应包含：

    local result | candidates | supported claims | contradicted claims |
    unresolved claims | docids | tried query families

收到报告后，比较各分支的候选和证据。在应该指向同一实体的分支之间求交集，排除被重要线索反驳的候选，只做连接、验证或排除报告所需的 targeted search。

不要重新开始宽泛搜索，也不要在没有新实体或新短语的情况下重复 child 已失败的 query family。每个报告都只是证据线索，不是最终真相；最终答案必须满足多个有区分度的线索，尤其是最能确定身份的困难线索。

### BrowseComp 输出校验

设置 answer["ready"] = True 前，必须确认 answer["content"] 恰好是三行：

    Explanation: 带 document citation 的简短解释
    Exact Answer: 最短且无歧义的答案
    Confidence: 0-100%

三个字段之间必须是实际换行，不能把多个字段放到一行。最后不能留下待执行的搜索、未执行的 REPL block 或进一步调查计划。

### BrowseComp task prompt

使用固定的 BrowseComp-Plus BM25 语料库回答 evidence-seeking question。

问题是：

    {query}

找到并验证答案，然后按照 Explanation / Exact Answer / Confidence 格式返回，不添加其他 section。

### BrowseComp forced-final

不再有工作步骤。该响应会被直接解析，不会执行代码。即使证据不完整，也要立即选择证据支持最强的答案。

整个响应必须恰好三行，并且第一个字符必须是 Explanation 中的 E：

    Explanation: 带引用的简短解释，例如 [12345]
    Exact Answer: 最短且无歧义的答案
    Confidence: 0-100%

不得提及 search call、REPL block、subagent、Markdown fence、进一步调查或其他 section。

## G. Prompt 在运行时的组合关系

环境 prompt 并不是替换通用 prompt，而是追加到通用 SYSTEM_PROMPT 后面。工具描述再次追加到最后。

因此 root 和 child 的 system prompt 具有相同的三层结构：

    通用规则
      + 环境规则
      + 工具说明

不同环境可以通过 prompt 覆盖通用行为。例如 Oolong-Synthetic 明确覆盖通用的 2 到 4 子任务规则，WebShop 强调实时动作必须串行，Oolong 强制 delegated agent 只读，BrowseComp-Plus 则强制独立证据分支和三行答案格式。

这说明 Curagent 的 prompt 设计不是单纯给模型提供背景，而是同时承担：

1. Agent 能力说明；
2. 任务路由策略；
3. context 使用协议；
4. child 输入输出协议；
5. 环境安全约束；
6. 最终答案格式约束。
