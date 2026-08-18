"""WebShop environment tool schemas."""

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
    ),
    ToolSchema(
        name="buy",
        description="Click Buy Now on the current product page.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        is_environment_tool=True,
    ),
)
