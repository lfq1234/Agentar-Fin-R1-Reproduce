"""联调诊断脚本：在进程内直接跑 02 流水线，逐 agent 计时，定位耗时/卡点。

用法（managed venv）：
    cd backend
    ../../.workbuddy/binaries/python/envs/default/Scripts/python.exe tools/_diag.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# 把 backend/ 加入 sys.path（脚本位于 backend/tools/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.agent.system as sysmod

# 给 _call_agent 加计时，不改源码
_orig_call = sysmod._call_agent


async def _timed_call_agent(agent, msg):
    name = getattr(agent, "name", "?")
    t = time.time()
    print(f"[agent {name}] start", flush=True)
    try:
        r = await _orig_call(agent, msg)
    except Exception as e:  # noqa: BLE001
        print(f"[agent {name}] ERROR after {round(time.time()-t,1)}s: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[agent {name}] done {round(time.time()-t,1)}s -> {str(r)[:80]!r}", flush=True)
    return r


sysmod._call_agent = _timed_call_agent


async def main() -> None:
    t0 = time.time()
    print("=== diag start ===", flush=True)
    try:
        result = await asyncio.wait_for(
            sysmod.run(message="存款保险的最高偿付限额是多少？", scene="Banking", structured=False),
            timeout=400,
        )
    except asyncio.TimeoutError:
        print(f"!!! TIMEOUT after {round(time.time()-t0,1)}s (still inside pipeline)", flush=True)
        return
    except Exception as e:  # noqa: BLE001
        print(f"!!! PIPELINE ERROR after {round(time.time()-t0,1)}s: {type(e).__name__}: {e}", flush=True)
        return
    print(f"=== diag DONE in {round(time.time()-t0,1)}s ===", flush=True)
    print("reply:", repr(result.reply[:300]), flush=True)
    print("compliance_notes:", result.compliance_notes, flush=True)
    print("risk_flags:", result.risk_flags, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
