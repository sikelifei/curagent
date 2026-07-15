# curagent

一个小型、通用的递归 Agent。root 与所有 child 使用完全相同的模型配置、短 system prompt、持久 Python REPL、custom tools 和递归接口；harness 不拆任务，也不替模型判断是否应该递归或并发。

## 安装

```bash
python -m pip install -e .
python -m pip install -e '.[all-providers]'  # 仅在需要 Anthropic/Gemini 时
```

首版只支持本地 Python REPL。模型端保留 RLM 的 `openai`、`vllm`、`openrouter`、`vercel`、`portkey`、`azure_openai`、`anthropic` 和 `gemini` 路由；Anthropic 与 Gemini SDK 通过对应 optional dependency 安装。

## 基本用法

```python
from recursive_agent import RecursiveAgent

agent = RecursiveAgent(
    backend="openai",
    backend_kwargs={
        "model_name": "your-model",
        "api_key": "...",
        "base_url": "https://example.com/v1",
        "sampling_args": {"temperature": 0.2, "max_tokens": 2048},
    },
    tools={
        "lookup": {
            "tool": lambda key: {"x": 42}.get(key),
            "description": "Look up a value by key. Print the result to inspect it.",
        }
    },
    max_steps=20,
    max_depth=4,
    max_concurrent_subagents=4,
    max_run_seconds=300,
    max_observation_chars=8000,
)

result = agent.run(
    task="Find the value of x and explain it.",
    context={"source": "example"},
)
print(result.answer)
print(result.status, result.steps, result.usage.to_dict())
```

模型在 REPL 中可以使用：

```python
child = spawn_subagent("Analyze candidate A", context=candidate_a)
results = spawn_subagents([
    {"task": "Analyze candidate A", "context": candidate_a},
    {"task": "Analyze candidate B", "context": candidate_b},
])
answer["content"] = "final text"
answer["ready"] = True
```

`spawn_subagents` 总是并发执行合法请求并按输入顺序返回。共享有状态环境是否适合并发由模型判断，runtime 不加环境锁、不 clone 环境，也不自动降级为串行。

## Observation 截断

REPL、tool 和环境返回值只有在模型执行 `print(...)` 后才进入下一轮消息。为防止
一次超长 stdout 或异常内容占满后续 context，harness 默认将每轮发给模型的
`REPL output` 限制为 8,000 字符：

- 短输出保持原样。
- 长输出保留首尾，并加入 `truncated by harness` 标记、原始字符数和上限。
- 本轮 execution error 会优先保留在 `Preserved execution errors` 段。
- `AgentTrace` 中的 `code_executions[].output/error` 仍保存完整原始值；
  `model_observation` 保存模型实际看到的截断版本，`observation_truncated` 记录
  本轮是否截断。

可通过 `max_observation_chars` 修改上限，或设置为 `None` 关闭：

```python
agent = RecursiveAgent(
    ...,
    max_observation_chars=12000,
)
```

这是字符边界，不声称等于某个模型的精确 token 数；不同 provider/model 的
tokenizer 不同。

## YAML 配置

配置采用单一 `model`，不区分 planner 和 subagent model：

```yaml
model:
  type: api
  api:
    api_key: "..."
    base_url: "https://example.com/v1"
    model: "your-model"
    temperature: 0.2
    max_tokens: 2048
    timeout: 120
```

```python
agent = RecursiveAgent.from_config("configs/model_api.local.yaml")
```

真实接口 smoke test：

```bash
python -m examples.smoke_from_config --config configs/model_api.local.yaml
```

## Environment 终止

可选的 `termination_check` 在每个 REPL code block 后调用：

```python
from recursive_agent import EnvironmentStatus

def status():
    return EnvironmentStatus(done=env.done, final_answer=env.final_answer)
```

模型显式提交的 `answer` 优先于 environment 状态；environment 已结束但没有答案时，Agent 只执行一次 forced-final completion。

## 测试

```bash
python -m unittest discover -v
```

测试覆盖 prompt/history、持久 REPL、tool 恢复、递归隔离、nested child、depth limit、保序并发、局部 child 失败、observation 截断与原始 trace 保留、终止顺序、forced final、usage、timeout 和 cancel。

## 环境插件

环境统一放在 `recursive_agent/envs/`，不是写死在 Agent loop 中：

```text
recursive_agent/envs/
  base.py                 # AgentEnvironment 通用契约
  registry.py             # 环境名称与 factory 注册
  runner.py               # 模型配置、tools、termination_check 接线
  webshop/
    environment.py        # ReCode WebShop 适配
    dataset.py            # train/test split 与 sample
    prompts.py            # 可编辑的 dataset task prompt
    tools.py              # observe/act 等 tool 注册
```

新增环境时实现 `AgentEnvironment` 并调用 `register_environment("name")` 即可。公共 runner 不需要增加环境分支。

### ReCode WebShop

WebShop 直接复用 `/data2/zhangwenjian/agent/ReCode` 的环境、商品数据、目标集和 Lucene 索引。由于 ReCode WebShop 的依赖固定在 Python 3.10 环境中，使用对应 conda 环境运行：

```bash
source /data2/zhangwenjian/miniconda3/etc/profile.d/conda.sh
conda activate recode

python -m examples.run_webshop \
  --config configs/model_api.local.yaml \
  --recode-root /data2/zhangwenjian/agent/ReCode \
  --split test \
  --instance-id 0 \
  --max-observation-chars 8000
```

也可以设置 `RECODE_ROOT`，省略 `--recode-root`。适配层注册以下 tools：

- `observe()`：当前 observation、合法 actions、history、reward 和 terminal 状态。
- `act(action)`：执行一个合法的 `search[...]` 或当前页面已有的 `click[...]`。
- `available_actions()`：返回当前可用 action 字符串。
- `episode_report()`：返回 reward、success、steps 和 trajectory。
- `shopping_instruction`：当前样本的只读购物指令。

`act` 不接受 `[FINISH]`，WebShop 只能通过合法购买动作或环境 step limit 进入终态。环境不会给共享 session 加并发锁；dataset prompt 会告知模型只能让一个 Agent 操作实时环境。

默认 prompt 位于 `recursive_agent/envs/webshop/prompts.py`。也可通过包含 `{instruction}` 的 UTF-8 模板文件覆盖：

```bash
python -m examples.run_webshop --prompt-file /path/to/webshop_prompt.txt
```

程序化入口：

```python
from recursive_agent.envs import create_environment, run_environment

environment = create_environment(
    "webshop",
    recode_root="/data2/zhangwenjian/agent/ReCode",
    split="test",
    instance_id=0,
    seed=233,
)
run = run_environment(
    environment,
    model_config="configs/model_api.local.yaml",
    agent_kwargs={"max_steps": 30, "max_depth": 2},
)
print(run.agent_result.answer)
print(run.environment_report)
```
