#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, fourteen free models.

Reads the brief at $MAKEFUN_PROMPT_FILE (the free-model table with the
house's quality ratings, and the six-prompt manifest), serves the client's
OpenAI / Anthropic requests by dialling the best free OpenRouter model that
meets each prompt through the grader's gateway, fails over on refusal, and
exits 0 when the env says the workload is done. Standard library only.

How it differs from the -01 entry this author wrote: -01 routed across ten
simulated providers on three wires with per-provider budgets; -02 has a
single provider (OpenRouter) behind one gateway route, so there is no
provider directory, no probe, and no wire matrix. Routing is a single
rating-ordered list per prompt, filtered by the capabilities the prompt
needs (JSON mode, tools, vision). Every upstream call is a Chat
Completions POST to $OPENAI_BASE_URL, whatever wire the client used.

Design notes (from the -02 kit's RULES.txt / competition.yaml and the
practice brief, read before writing this):
- Quality is 60% of the score and pays the rating of the model that really
  served each prompt, over the best rating meeting that prompt's
  requirements. So the candidate order is rating descending, and the top
  model is retried on later prompts after a single refusal: a 429 is
  usually a per-minute rate limit that clears, and giving up the best
  model for the whole trial would throw away quality on every later
  prompt. A model is only dropped after two refusals in the trial.
- The provider's own answer body must carry the prompt's token for the
  quality credit, so the upstream answer is passed through verbatim: the
  text is never rewritten, trimmed, or summarised, and JSON-mode requests
  keep their response_format so the model answers in JSON.
- The upstream call is always non-stream JSON, whatever the client asked.
  The answer (text, tool calls, usage, stop reason) is then rendered in
  the client's wire, JSON or SSE. One renderer per wire keeps the
  translation to a request-side rewrite plus a response-side render.
- Determinism: fixed created timestamps, no wall clock in any reply, no
  randomness; a route is a function of the brief, the requests seen, and
  the gateway's answers so far. The poll loop only exits once the env says
  done AND no request is still in flight, so the last answer is never cut
  off by the exit.
- The catalog is built from the brief alone and printed byte-for-byte by
  --list: one provider, openrouter, exactly the fourteen dialable free ids.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.5
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT_S = 12
MAX_ATTEMPTS_PER_PROMPT = 4
DEFAULT_MAX_TOKENS = 512

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"


def log(message):
    sys.stderr.write("aggregator-02: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    if isinstance(v, bool):
        return None
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


# ---------------------------------------------------------------- brief ---

def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except ValueError:
                pass
        start = text.find("{", start + 1)
    return out


def _model_of(v):
    if not is_record(v):
        return None
    mid = str_of(v.get("id"))
    if mid == "" or v.get("free") is not True:
        return None
    return {
        "id": mid,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": True,
    }


def _prompt_of(v):
    if not is_record(v):
        return None
    pid = str_of(v.get("id"))
    if pid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    return {
        "id": pid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": {
            "tools": requires.get("tools") is True,
            "vision": requires.get("vision") is True,
            "json": requires.get("json") is True,
        },
    }


def read_brief(path):
    """The brief at `path`: the free-model table and the prompt manifest."""
    brief = {"models": [], "prompts": [], "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    models_doc = next(
        (d for d in docs if d.get("schema") == MODELS_SCHEMA and isinstance(d.get("models"), list)),
        None,
    )
    if models_doc is None:
        brief["error"] = "the brief at %s holds no %s document" % (path, MODELS_SCHEMA)
        return brief
    brief["models"] = [m for m in (_model_of(v) for v in models_doc["models"]) if m is not None]
    prompts_doc = next(
        (d for d in docs if d.get("schema") == PROMPTS_SCHEMA and isinstance(d.get("prompts"), list)),
        None,
    )
    if prompts_doc is not None:
        brief["prompts"] = [p for p in (_prompt_of(v) for v in prompts_doc["prompts"]) if p is not None]
    return brief


# --------------------------------------------------------------- catalog ---

def build_catalog(brief):
    """The catalog from the brief alone: one provider, openrouter, exactly
    the brief's free models, ids sorted for a stable byte-for-byte --list."""
    models = [{
        "id": m["id"],
        "free": True,
        "context_length": m["context_length"],
    } for m in sorted(brief["models"], key=lambda m: m["id"])]
    return {"providers": [{
        "id": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "wire": "openai",
        "models": models,
    }]}


def served_models(brief):
    """The free models, ids sorted: what GET /v1/models lists, either wire."""
    return [{"id": m["id"], "display_name": m["id"]} for m in sorted(brief["models"], key=lambda m: m["id"])]


# --------------------------------------------------------------- routing ---

def meets_requirements(requires, model):
    if requires.get("tools") is True and model["tools"] is not True:
        return False
    if requires.get("vision") is True and model["vision"] is not True:
        return False
    if requires.get("json") is True and model["json"] is not True:
        return False
    return True


class Router:
    """Rating-ordered candidates per prompt, with trial-wide refusal state.

    A model that refused once stays a candidate: a 429 is usually a
    per-minute limit that clears before the next prompt. A model that
    refused twice is dropped for the rest of the trial.
    """

    def __init__(self, brief):
        self.brief = brief
        self.refusals = {}

    def candidates(self, requires):
        out = [m for m in self.brief["models"]
               if meets_requirements(requires, m)
               and self.refusals.get(m["id"], 0) < 2]
        out.sort(key=lambda m: (-m["rating_milli"], m["id"]))
        return out

    def note_refusal(self, model_id):
        self.refusals[model_id] = self.refusals.get(model_id, 0) + 1


# ------------------------------------------------------------ wire: text ---

def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        b["text"] for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _openai_part_of(block):
    """One Anthropic content block as an OpenAI content part, or None."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _openai_part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            },
        })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return out


def openai_request_for(body, model):
    """The client body as the gateway's Chat Completions body: OpenAI
    bodies pass through with the model swapped, Anthropic ones translate."""
    upstream = dict(body)
    upstream["model"] = model
    upstream["stream"] = False
    upstream.pop("stream_options", None)
    if not isinstance(upstream.get("max_tokens"), (int, float)):
        upstream["max_tokens"] = DEFAULT_MAX_TOKENS
    return upstream


# --------------------------------------------------------------- upstream ---

def _error_message(payload):
    if not is_record(payload):
        return ""
    err = payload.get("error")
    if is_record(err) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    return ""


def _error_code(status, payload):
    if not is_record(payload):
        return str(status)
    err = payload.get("error")
    if is_record(err):
        for key in ("code", "type"):
            value = err.get(key)
            if isinstance(value, str) and value != "":
                return value
            if isinstance(value, int):
                return str(value)
    return str(status)


def _http_json(method, url, headers, body, timeout=UPSTREAM_TIMEOUT_S):
    """One JSON request; returns (status, parsed body or None, raw text)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw), raw
            except ValueError:
                return resp.status, None, raw
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        try:
            return err.code, json.loads(raw), raw
        except ValueError:
            return err.code, None, raw
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return 0, None, "unreachable: %s" % err


def call_gateway(model, client_api, body):
    """One upstream inference call on the gateway's Chat Completions route.

    Returns (ok, text, tool_calls, finish, pin, pout, status, code): ok is
    True only on a 2xx answer with a choices array; otherwise status/code
    name the refusal. The gateway's own answer names the model without the
    openrouter/ prefix; callers put the dialled id back on every reply."""
    if client_api == "anthropic":
        upstream = anthropic_to_openai(body, model)
    else:
        upstream = openai_request_for(body, model)
    upstream["stream"] = False
    upstream.pop("stream_options", None)
    if not isinstance(upstream.get("max_tokens"), (int, float)):
        upstream["max_tokens"] = DEFAULT_MAX_TOKENS
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY != "":
        headers["Authorization"] = "Bearer " + OPENAI_API_KEY
    status, payload, _raw = _http_json(
        "POST", OPENAI_BASE_URL + "/chat/completions", headers, upstream)
    if 200 <= status < 300 and is_record(payload):
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        usage = payload.get("usage") if is_record(payload.get("usage")) else {}
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "stop"
        pin = num_or_none(usage.get("prompt_tokens"))
        pout = num_or_none(usage.get("completion_tokens"))
        return True, text_of(message.get("content")), tool_calls, finish, pin, pout, status, ""
    return False, "", [], "stop", None, None, status, _error_code(status, payload)


def usage_counts(text, pin, pout):
    """Whole usage counts for a reply: the provider's own numbers when it
    gave them, else a character-based estimate that still satisfies the
    wire check (input >= 0, output >= 1)."""
    if pin is None:
        pin = 0
    if pout is None or pout < 1:
        pout = max(1, len(text) // 4)
    return int(pin), int(pout)


# ------------------------------------------------------- client rendering ---

def _split_text(text, size=64):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def _openai_tool_calls(tool_calls):
    out = []
    for call in tool_calls:
        if not is_record(call):
            continue
        fn = call.get("function") if is_record(call.get("function")) else {}
        args = fn.get("arguments")
        out.append({
            "id": call.get("id") if isinstance(call.get("id"), str) else "",
            "type": "function",
            "function": {
                "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                "arguments": args if isinstance(args, str) else json.dumps(arguments_of(args)),
            },
        })
    return out


def openai_json_answer(model, text, tool_calls, finish, pin, pout):
    message = {"role": "assistant", "content": text}
    calls = _openai_tool_calls(tool_calls)
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "chatcmpl-agg02",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish if finish in ("stop", "length", "tool_calls", "content_filter") else "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout},
    }


def openai_sse_answer(model, text, tool_calls, finish, pin, pout):
    """The provider's whole text as an OpenAI event stream, ending with the
    usage chunk and [DONE]."""
    head = {"id": "chatcmpl-agg02", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model}
    frames = []

    def frame(chunk):
        merged = dict(head)
        merged.update(chunk)
        return "data: %s\n\n" % json.dumps(merged)

    frames.append(frame({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}))
    for piece in _split_text(text):
        frames.append(frame({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
    calls = _openai_tool_calls(tool_calls)
    stop = "tool_calls" if calls else (finish if finish in ("stop", "length", "content_filter") else "stop")
    last = {"index": 0, "delta": {}, "finish_reason": stop}
    if calls:
        last["delta"] = {"tool_calls": calls}
    frames.append(frame({"choices": [last]}))
    frames.append(frame({"choices": [], "usage": {
        "prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def _anthropic_blocks(text, tool_calls):
    blocks = []
    if text != "":
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        if not is_record(call):
            continue
        fn = call.get("function") if is_record(call.get("function")) else {}
        args = fn.get("arguments")
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": arguments_of(args),
        })
    return blocks


def anthropic_json_answer(model, text, tool_calls, finish, pin, pout):
    stop = "tool_use" if tool_calls else (finish if finish in ("end_turn", "max_tokens", "stop_sequence") else "end_turn")
    return {
        "id": "msg_agg02",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _anthropic_blocks(text, tool_calls),
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": pin, "output_tokens": pout},
    }


def anthropic_sse_answer(model, text, tool_calls, finish, pin, pout):
    """The provider's whole text as the Anthropic event sequence."""
    frames = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data))

    frames.append(event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg02", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": pin, "output_tokens": 0},
        },
    }))
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    index = 1
    for block in _anthropic_blocks("", tool_calls):
        frames.append(event("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": block,
        }))
        frames.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    stop = "tool_use" if tool_calls else (finish if finish in ("end_turn", "max_tokens", "stop_sequence") else "end_turn")
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": pout},
    }))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": "openrouter"}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def openai_error(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


# ------------------------------------------------------------- serve one ---

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
ROUTER = Router(BRIEF)
MANIFEST_INDEX = 0
MANIFEST_LOCK = threading.Lock()
IN_FLIGHT = 0
IN_FLIGHT_LOCK = threading.Lock()


def requirements_for(body):
    """The requirements of the request now arriving: the manifest's row by
    arrival order, else the body's own fields."""
    global MANIFEST_INDEX
    with MANIFEST_LOCK:
        manifest = BRIEF.get("prompts") or []
        entry = manifest[MANIFEST_INDEX] if MANIFEST_INDEX < len(manifest) else None
        MANIFEST_INDEX += 1
    if entry is not None:
        return dict(entry["requires"])
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def serve_request(client_api, body):
    """Route one client request, failing over on refusal.

    Returns (status, wire, payload, failures): wire is 'openai' or
    'anthropic', payload is a JSON object or, for a stream, the SSE text,
    and failures is the ordered refusal trail for the optional
    x-makefun-upstream-failures header (empty when nothing refused)."""
    requires = requirements_for(body)
    stream = body.get("stream") is True
    failures = []
    candidates = ROUTER.candidates(requires)
    for model in candidates[:MAX_ATTEMPTS_PER_PROMPT]:
        ok, text, tool_calls, finish, pin, pout, status, code = call_gateway(
            model["id"], client_api, body)
        if ok:
            pin, pout = usage_counts(text, pin, pout)
            if client_api == "openai":
                if stream:
                    return 200, "openai", openai_sse_answer(model["id"], text, tool_calls, finish, pin, pout), failures
                return 200, "openai", openai_json_answer(model["id"], text, tool_calls, finish, pin, pout), failures
            if stream:
                return 200, "anthropic", anthropic_sse_answer(model["id"], text, tool_calls, finish, pin, pout), failures
            return 200, "anthropic", anthropic_json_answer(model["id"], text, tool_calls, finish, pin, pout), failures
        ROUTER.note_refusal(model["id"])
        failures.append({"status": status, "code": code, "model": model["id"]})
        log("refused: %s HTTP %s (%s)" % (model["id"], status, code))
    if client_api == "anthropic":
        return 503, "anthropic", anthropic_error("overloaded_error", "no free model meets this request"), failures
    return 503, "openai", openai_error("no free model meets this request", "no_free_model", None), failures


# ---------------------------------------------------------------- server ---

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if is_record(value) else None

    def _send_json(self, status, payload, failures=None):
        text = json.dumps(payload)
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._send_failures_header(failures)
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text, failures=None):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(data)))
        self._send_failures_header(failures)
        self.end_headers()
        self.wfile.write(data)

    def _send_failures_header(self, failures):
        """The optional refusal trail: at most 8 rows, under 8 KiB."""
        if not failures:
            return
        trail = json.dumps(failures[:8], separators=(",", ":"))
        if len(trail.encode("utf-8")) > 8192:
            return
        self.send_header("x-makefun-upstream-failures", trail)

    def _send_error(self, wire, status, type_, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error(type_, message))
        else:
            self._send_json(status, openai_error(message, type_, None))

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/v1/models":
            models = served_models(BRIEF)
            if isinstance(self.headers.get("x-api-key"), str):
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % url.path)

    def do_POST(self):
        global IN_FLIGHT
        url = urllib.parse.urlparse(self.path)
        if url.path not in ("/v1/chat/completions", "/v1/messages"):
            self._send_error("anthropic" if url.path == "/v1/messages" else "openai",
                             404, "not_found", "no route POST %s" % url.path)
            return
        client_api = "anthropic" if url.path == "/v1/messages" else "openai"
        body = self._read_json()
        if body is None:
            self._send_error(client_api, 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        with IN_FLIGHT_LOCK:
            IN_FLIGHT += 1
        try:
            status, wire, payload, failures = serve_request(client_api, body)
        except Exception as err:  # never drop a request on an internal error
            log("serve failed: %r" % err)
            self._send_error(client_api, 500, "internal_error", "the server failed to answer")
            return
        finally:
            with IN_FLIGHT_LOCK:
                IN_FLIGHT -= 1
        if isinstance(payload, str):
            self._send_sse(payload, failures)
        else:
            self._send_json(status, payload, failures)


# --------------------------------------------------------- register/poll ---

def register(port):
    status, payload, raw = _http_json(
        "POST", ENV_URL + "/register", {"Content-Type": "application/json"},
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        timeout=5,
    )
    if not (200 <= status < 300):
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (ENV_URL, status, raw[:500]))
    if is_record(payload) and isinstance(payload.get("requests"), (int, float)):
        log("registered: the workload holds %s requests" % payload["requests"])


def poll_until_done(server):
    """Exit 0 once the env says done and no request is still in flight."""
    while True:
        try:
            status, payload, _raw = _http_json("GET", ENV_URL + "/work", {}, None, timeout=2)
            if status == 200 and is_record(payload) and payload.get("done") is True:
                with IN_FLIGHT_LOCK:
                    busy = IN_FLIGHT
                if busy == 0:
                    server.shutdown()
                    return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def catalog_text():
    return json.dumps(CATALOG, separators=(",", ":")) + "\n"


def start():
    if BRIEF.get("error"):
        log(BRIEF["error"])
    if "--list" in sys.argv:
        sys.stdout.write(catalog_text())
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if OPENAI_BASE_URL == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%s with %s free models"
        % (port, len(BRIEF["models"])))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port)
    poll_until_done(server)
    sys.exit(0)


if __name__ == "__main__":
    try:
        start()
    except Exception as err:
        log("fatal: %r" % err)
        sys.exit(1)
