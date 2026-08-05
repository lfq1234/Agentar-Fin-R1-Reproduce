"""07-隐私脱敏（PII redact）。

落库前对敏感字段做掩码（手机号/身份证/银行卡/邮箱），规则可配，白名单避免误伤
（如合规条款编号、产品代码）。脱敏仅作用于 07 落库副本，不影响实时 ``reply``（需求 FR8）。

覆盖范围：所有写路径——user/assistant 主消息、trace_events.summary_in/out、meta_json、
final_result JSON 中的字符串叶子都在 store 写入前统一过一遍（评审 S2）。
"""
from __future__ import annotations

import re

_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_IDCARD = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
_BANK = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# 默认白名单：避免把合规条款编号（如「第138条」）、产品代码等误伤为手机号/银行卡。
# 形如「第1234567890123456号」的整体用更精确的边界正则已规避大部分；此处留扩展点。
_DEFAULT_WHITELIST: list[str] = []


def _mask_phone(v: str) -> str:
    return v[:3] + "****" + v[-4:]


def _mask_idcard(v: str) -> str:
    return v[:6] + "********" + v[-4:]


def _mask_bank(v: str) -> str:
    return v[:4] + "****" + v[-4:]


def _mask_email(v: str) -> str:
    user, _, domain = v.partition("@")
    if len(user) <= 2:
        return "***@" + domain
    return user[:2] + "***@" + domain


def redact_text(text: str, *, whitelist: list[str] | None = None) -> str:
    """对单段文本做 PII 掩码；空值原样返回。"""
    if not text:
        return text
    wl = whitelist if whitelist is not None else _DEFAULT_WHITELIST
    if wl:
        # 白名单条目整体跳过（不替换）
        protected: list[tuple[int, int]] = []
        for item in wl:
            start = text.find(item)
            while start != -1:
                protected.append((start, start + len(item)))
                start = text.find(item, start + len(item))
        if protected:
            out: list[str] = []
            i = 0
            for s, e in sorted(protected):
                out.append(text[i:s])
                i = e
            out.append(text[i:])
            base = "".join(out)
        else:
            base = text
    else:
        base = text

    base = _PHONE.sub(lambda m: _mask_phone(m.group(0)), base)
    base = _IDCARD.sub(lambda m: _mask_idcard(m.group(0)), base)
    base = _BANK.sub(lambda m: _mask_bank(m.group(0)), base)
    base = _EMAIL.sub(lambda m: _mask_email(m.group(0)), base)
    return base


def redact_value(value: object, *, whitelist: list[str] | None = None) -> object:
    """递归脱敏任意结构中的字符串叶子（用于 final_result / meta 字典）。"""
    if isinstance(value, str):
        return redact_text(value, whitelist=whitelist)
    if isinstance(value, list):
        return [redact_value(v, whitelist=whitelist) for v in value]
    if isinstance(value, dict):
        return {k: redact_value(v, whitelist=whitelist) for k, v in value.items()}
    return value
