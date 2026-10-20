"""Response parsing, separate from chat templates and memory replay.

Following SGLang's reasoning/function-call split, reasoning_parser removes the
reasoning section for display and tool_call_parser extracts validated calls.
This client keeps its existing parsing behavior without a server dependency;
Flash uses the checkpoint's native completion parser. Parsing never rewrites
original generated token IDs retained by MemTSession.
"""

import json
import re
from copy import deepcopy
from uuid import uuid4

from memt_sglang.client.chat_format import token_ids

_TOOL_PATTERN = (
    r"<tool_call>\s*<function=(?P<name>[^<>\s]+)>\s*(?P<arguments>.*?)"
    r"\s*</function>\s*</tool_call>"
)
_PARAMETER_PATTERN = r"<parameter=(?P<name>[^<>\s]+)>\n?(?P<value>.*?)\n?</parameter>"


class ResponseParser:
    """Non-streaming reasoning and function-call parsing for generated output.

    Instances contain configuration only; each parse() call uses fresh local
    state, so the returned parser can be shared by sequential sessions.
    """

    def __init__(self, tokenizer, *, eos_token_ids=None):
        self.tokenizer = tokenizer
        self.eos_token_ids = frozenset(
            {tokenizer.eos_token_id} if eos_token_ids is None else eos_token_ids
        )

    def parse(self, ids, prompt, *, thinking=None, tools=None):
        generated = list(token_ids(ids))
        visible = generated[:-1] if generated[-1] in self.eos_token_ids else generated
        body = self.tokenizer.decode(visible, skip_special_tokens=False)
        opening = self.thinking_opening
        if self.tokenizer.decode(list(prompt), skip_special_tokens=False).endswith(
            opening
        ):
            body = opening + body
        answer = self.reasoning_parser(body) if thinking is False else body
        calls = self.tool_call_parser(self.reasoning_parser(body), tools or [])
        message = {"role": "assistant", "content": body}
        return answer, calls, message

    @staticmethod
    def reasoning_parser(value):
        if "</think>" in value:
            value = value.split("</think>", 1)[1].lstrip("\n")
        return re.sub(r"<think>.*?(?:</think>|$)", "", value, flags=re.DOTALL).lstrip(
            "\n"
        )

    def tool_call_parser(self, content, tools):
        raise NotImplementedError


class QwenResponseParser(ResponseParser):
    thinking_opening = "<think>\n"

    def tool_call_parser(self, content, tools):
        if not tools or "<tool_call>" not in content:
            return []
        matches = list(re.finditer(_TOOL_PATTERN, content, flags=re.DOTALL))
        if not matches or "<tool_call>" in re.sub(
            _TOOL_PATTERN, "", content, flags=re.DOTALL
        ):
            raise ValueError("Incomplete or malformed tool request")
        functions = {tool["function"]["name"]: tool["function"] for tool in tools}
        calls = []
        for match in matches:
            function = functions.get(match["name"])
            if function is None:
                raise ValueError(f"Unknown tool: {match['name']}")
            if re.sub(
                _PARAMETER_PATTERN, "", match["arguments"], flags=re.DOTALL
            ).strip():
                raise ValueError("Malformed tool parameters")
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            arguments = {}
            for argument in re.finditer(
                _PARAMETER_PATTERN, match["arguments"], flags=re.DOTALL
            ):
                name, value = argument["name"], argument["value"]
                if name in arguments:
                    raise ValueError(f"Duplicate tool argument: {name}")
                if (
                    name not in properties
                    and parameters.get("additionalProperties") is False
                ):
                    raise ValueError(f"Unknown tool argument: {name}")
                if properties.get(name, {}).get("type") != "string":
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        if properties.get(name, {}).get("type") is not None:
                            raise ValueError(
                                f"Invalid JSON tool argument: {name}"
                            ) from None
                arguments[name] = value
            if set(parameters.get("required", [])) - arguments.keys():
                raise ValueError("Missing required tool arguments")
            calls.append(
                {
                    "id": "call_" + uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": match["name"],
                        "arguments": arguments,
                    },
                }
            )
        return calls


class DeepSeekResponseParser(ResponseParser):
    thinking_opening = "<think>"

    def __init__(self, tokenizer, *, eos_token_ids=None, parse_message=None):
        super().__init__(tokenizer, eos_token_ids=eos_token_ids)
        self.parse_message = parse_message

    def tool_call_parser(self, content, tools):
        if not tools or "<｜DSML｜tool_calls>" not in content:
            return []
        if self.parse_message is None:
            raise ValueError("DeepSeek tools require the checkpoint's native parser")
        if content.startswith("<｜DSML｜tool_calls>"):
            content = "\n\n" + content
        parsed = self.parse_message(content + "<｜end▁of▁sentence｜>", "chat")
        calls = parsed.get("tool_calls") or []
        functions = {tool["function"]["name"]: tool["function"] for tool in tools}
        for call in calls:
            function = call["function"]
            declared = functions.get(function["name"])
            if declared is None:
                raise ValueError(f"Unknown tool: {function['name']}")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            parameters = declared.get("parameters", {})
            if set(parameters.get("required", [])) - arguments.keys():
                raise ValueError("Missing required tool arguments")
            if parameters.get("additionalProperties") is False and (
                arguments.keys() - parameters.get("properties", {}).keys()
            ):
                raise ValueError("Unknown tool arguments")
            function["arguments"] = arguments
            call["id"] = "call_" + uuid4().hex
        return calls

    def parse(self, ids, prompt, *, thinking=None, tools=None):
        answer, calls, message = super().parse(
            ids, prompt, thinking=thinking, tools=tools
        )
        if calls:
            body = message["content"]
            message["content"] = (
                self.reasoning_parser(body).split("<｜DSML｜tool_calls>")[0].rstrip()
            )
            if body.startswith("<think>") and "</think>" in body:
                message["reasoning_content"] = body[len("<think>") :].split(
                    "</think>", 1
                )[0]
            message["tool_calls"] = deepcopy(calls)
            for call in message["tool_calls"]:
                call["function"]["arguments"] = json.dumps(
                    call["function"]["arguments"], ensure_ascii=False
                )
        return answer, calls, message
