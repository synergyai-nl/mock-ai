import json
import logging
import random
from time import time
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError

from mockai.dependencies import ResponseFile
from mockai.models.json_file import PreDeterminedResponse

_log = logging.getLogger(__name__)


def _encode_arguments(arguments: Any) -> str:
    """Encode tool call arguments the way the OpenAI wire format requires.

    `function.arguments` is a JSON-encoded *string*, not an object. Responses
    files declare it as an object, so it is encoded here — the one place both
    the streaming and non-streaming paths pass through. Values that are
    already strings are returned unchanged, so callers that supply their own
    encoded JSON are not double-encoded.
    """
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def json_response(content: str | None, model: str, tool_calls: list[dict] | None):
    response = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time()),
        "model": model,
        "system_fingerprint": "mock",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "logprobs": None,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return response


def _stream_chunk(id: str, model: str, delta: dict, finish_reason: str | None = None):
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time()),
        "model": model,
        "system_fingerprint": "mock",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def streaming_response(content: str | None, model: str, tool_calls: list[dict] | None):
    id = f"chatcmpl-{uuid4().hex}"

    def chunk(delta: dict, finish_reason: str | None = None):
        return f"data: {json.dumps(_stream_chunk(id, model, delta, finish_reason))}\n\n"

    if content is not None:
        # `role` is announced once, in the opening chunk, and omitted after —
        # repeating it on every delta is not what the API does.
        for position, character in enumerate(content):
            delta = {"content": character, "tool_calls": None}
            if position == 0:
                delta["role"] = "assistant"
            yield chunk(delta)
        yield chunk({}, finish_reason="stop")
    elif tool_calls is not None:
        # Each tool call is streamed on its own `index`: one header delta
        # introducing id/type/name with empty arguments, then deltas carrying
        # only the argument fragments. Clients accumulate fragments by index,
        # so the index — not the position in this list — is what ties them
        # together, and repeating id/type/name would corrupt the accumulation.
        for index, tool_call in enumerate(tool_calls):
            function = tool_call["function"]
            header = {
                "content": None,
                "tool_calls": [
                    {
                        "index": index,
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {"name": function["name"], "arguments": ""},
                    }
                ],
            }
            if index == 0:
                header["role"] = "assistant"
            yield chunk(header)
            for character in function["arguments"]:
                yield chunk(
                    {
                        "tool_calls": [
                            {"index": index, "function": {"arguments": character}}
                        ]
                    }
                )
        yield chunk({}, finish_reason="tool_calls")
    else:
        raise ValueError("Either content or tool_calls must not be None")

    yield "data: [DONE]\n\n"


def response_struct_to_openai_format(response: PreDeterminedResponse):
    if response.type == "text":
        content = response.output
        tool_calls = None
    elif response.type == "function":
        content = None

        if isinstance(response.output, str):
            raise ValueError("Impossible state")

        tool_calls = response.output._to_dict_list()

        for tool_call in tool_calls:
            tool_call["id"] = str(uuid4())
            tool_call["type"] = "function"
            function = {
                "name": tool_call.pop("name"),
                "arguments": _encode_arguments(tool_call.pop("arguments")),
            }
            tool_call["function"] = function
    else:
        raise ValueError("unreachable")

    return content, tool_calls


async def generate_openai_completion_response(
    payload: dict,
    responses: ResponseFile,
    mock_response: str | None,
):
    model = payload["model"]
    stream = payload.get("stream")
    content = None
    for message in payload["messages"][::-1]:
        if message["role"] == "user":
            content = message["content"]
            break
    if content is None:
        content = payload["messages"][-1]["content"]

    if content is None:
        raise ValueError("Content from last message cannot be None")
    tool_calls = None

    if isinstance(content, list):
        for obj in content:
            if obj["type"] == "text":
                content = obj["text"]
                break
        else:
            raise ValueError(
                "Content array must include at least one object with 'type' = 'text'",
            )
    found_predetermined_response = False
    if responses is not None:
        response = responses.find_matching_or_none(origin="openai", payload=payload)
        if response is not None:
            content, tool_calls = response_struct_to_openai_format(response)
            found_predetermined_response = True

    _log.info(
        "predetermined response %s found",
        "not" if not found_predetermined_response else "",
    )

    if mock_response is not None:
        if found_predetermined_response:
            _log.info(
                "Overriding predetermined response with mock response from header"
            )
        else:
            _log.info("Using mock response from header")
        try:
            is_function = mock_response[:2] == "f:"

            r_type = "function" if is_function else "text"

            if is_function:
                output = json.loads(mock_response[2:])
            else:
                output = mock_response

            header_mock_response = PreDeterminedResponse(
                type=r_type, input="None", output=output
            )
            content, tool_calls = response_struct_to_openai_format(header_mock_response)
        except (ValidationError, json.JSONDecodeError) as e:
            content = str(e)

    content = cast(str, content)

    if stream is None or stream is False:
        return json_response(content, model, tool_calls)
    else:
        return streaming_response(content, model, tool_calls)


async def generate_openai_embeddings_response(embedding_size: int, payload: dict):
    if isinstance(input := payload["input"], str):
        input_list = [input]
    else:
        input_list = input

    embedding_range = range(embedding_size)
    input_range = range(len(input_list))

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [random.uniform(-1, 1) for _ in embedding_range],
                "index": number,
            }
            for number in input_range
        ],
        "model": payload["model"],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
