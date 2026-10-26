from __future__ import annotations
import argparse, hashlib, json, os, re, urllib.request
from typing import Mapping, Any

def norm(value: str) -> str:
    return " ".join(re.sub(r"[^\w.]+", " ", str(value).casefold()).split())

def completion_endpoint(args: argparse.Namespace) -> str:
    endpoint = args.local_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/completions"
    elif not endpoint.endswith("/v1/completions"):
        endpoint += "/v1/completions"
    return endpoint

def completion_request(body: Mapping[str, Any], args: argparse.Namespace) -> dict:
    request = urllib.request.Request(
        completion_endpoint(args), data=json.dumps(dict(body)).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def chat_completion_request(body: Mapping[str, Any], args: argparse.Namespace) -> dict:
    endpoint = args.local_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/chat/completions"
    elif not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    request = urllib.request.Request(
        endpoint, data=json.dumps(dict(body), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def suffix_logprobs(left: str, right: str, candidates: list[str],
                    args: argparse.Namespace) -> dict[str, float]:
    """Score public right-context compatibility after inserting each proposal."""
    right_window = right[:512]
    prefixes = [left + candidate for candidate in candidates]
    prompts = [prefix + right_window for prefix in prefixes]
    body = json.dumps({"model": args.local_model, "prompt": prompts, "max_tokens": 0,
                       "echo": True, "logprobs": 1, "temperature": 0.0}).encode("utf-8")
    request = urllib.request.Request(completion_endpoint(args), data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    by_index = {int(choice["index"]): choice for choice in payload.get("choices", [])}
    scores: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        choice = by_index[index]
        info = choice.get("logprobs") or {}
        offsets = info.get("text_offset") or []
        tokens = info.get("tokens") or []
        token_scores = info.get("token_logprobs") or []
        start = len(prefixes[index])
        selected = [float(score) for offset, token, score in zip(offsets, tokens, token_scores)
                    if score is not None and int(offset) + len(str(token)) > start]
        scores[candidate] = sum(selected) / len(selected) if selected else 0.0
    return scores


def stable_seed(seed, *parts):
    digest = hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def local_json(system, payload, args, max_tokens=800, temperature=0.):
    response = chat_completion_request(dict(model=args.local_model, messages=[
        dict(role="system", content=system), dict(role="user", content=json.dumps(payload, ensure_ascii=False))],
        temperature=temperature, max_tokens=max_tokens, chat_template_kwargs={"enable_thinking": False}), args)
    c = response["choices"][0]
    if c.get("finish_reason") == "length": raise ValueError("Truncated JSON")
    s = c["message"]["content"]
    return json.loads(s[s.find("{"):s.rfind("}")+1])
