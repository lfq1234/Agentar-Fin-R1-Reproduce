"""本地 qwen3-0.6b 的 OpenAI 兼容推理服务（联调用，非生产后端代码）。

用途：为 backend 的 local 模式（app/model/local/vllm_local.py -> OpenAI SDK）提供
一个最小可用的 OpenAI 兼容 /v1 端点。backend 只调用 chat.completions，因此本服务
实现 /v1/chat/completions 与 /v1/models 即可；embedding 端点暂不实现，因为当前
app/agent/system.py 的 run() 中 RAG 为占位实现，不会调用 get_embedder()。

运行（需 anaconda python，已含 torch+cuda / transformers / fastapi / uvicorn）：
    set MODEL_PATH=D:/models/Qwen3-0.6B
    /d/Program Files/anaconda3/python.exe -m uvicorn tools.qwen_server:app ^
        --host 0.0.0.0 --port 9000

健康检查：curl http://localhost:9000/v1/models
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "D:/models/Qwen3-0.6B").strip()
MODEL_ID = os.environ.get("MODEL_ID", "qwen3-0.6b").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))

print(f"[qwen_server] loading {MODEL_PATH} on {DEVICE} ...", flush=True)
_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
_MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
print(f"[qwen_server] loaded model_id={MODEL_ID} device={DEVICE}", flush=True)

app = FastAPI(title="qwen3-0.6b local server (OpenAI-compatible)")


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    try:
        return _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 关闭思考链，直接给答案
        )
    except TypeError:
        return _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _generate(messages: List[Dict[str, str]], temperature: float) -> str:
    prompt = _build_prompt(messages)
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    do_sample = temperature > 0
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    # 仅解码新生成的 token
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Dict[str, Any]:
    body = await req.json()
    messages = body.get("messages", [])
    temperature = float(body.get("temperature", TEMPERATURE))
    stream = bool(body.get("stream", False))
    content = _generate(messages, temperature)

    if stream:
        # 简化：整体作为单个 chunk 返回（backend 当前固定 stream=false，此为兼容兜底）
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
