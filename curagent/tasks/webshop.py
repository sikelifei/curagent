"""Environment facts for WebShop; no benchmark policy is embedded here."""

from curagent.core.prompt import TaskModule
from curagent.core.types import ToolSchema


WEBSHOP_ENVIRONMENT_TOOLS = (
    ToolSchema(
        name="search",
        description="Submit the exact query on the WebShop search page.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        is_environment_tool=True,
        is_write=True,
    ),
    ToolSchema(
        name="click",
        description="Click one exact visible product, option, or navigation target.",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}},
            "required": ["target"],
            "additionalProperties": False,
        },
        is_environment_tool=True,
        is_write=True,
    ),
    ToolSchema(
        name="buy",
        description="Click Buy Now on the current product page.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        is_environment_tool=True,
        is_write=True,
    ),
)


WEBSHOP_TASK_MODULE = TaskModule(
    instruction="Complete the shopping instruction in the current WebShop session.",
    observation_spec=(
        "Observation.text is the rendered current page. Observation.metadata contains the shopping "
        "instruction, reward, done flag, and exact visible targets when the adapter can identify them."
    ),
    environment_tools=WEBSHOP_ENVIRONMENT_TOOLS,
    environment_rules=(
        "search submits a query only when the current page exposes search.",
        "click accepts one exact visible target such as a product id, option, or navigation label.",
        "buy submits Buy Now on the current product page and may terminate the episode.",
        "WebShop is single-writer. The root owns the real session; readonly children cannot mutate it.",
        "A rejected action has no effect. A stale-version action is never replayed automatically.",
    ),
    finish_condition=(
        "The environment is complete after a terminal purchase. A readonly analysis child completes "
        "by calling finish with its requested JSON result."
    ),
)
