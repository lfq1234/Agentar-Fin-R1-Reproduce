"""08 个人文档：文本解析（本期仅纯文本类，二进制显式拒绝、不抛 500）。

约定（技术文档 §5.3）：
- ``.txt`` / ``.md`` / ``.markdown`` / ``.html`` / ``.htm`` → 解析为纯文本；
- 其它扩展名（``.pdf`` / ``.docx`` / ``.xlsx`` …）→ 抛 :class:`UnsupportedDocument`，
  调用方据此把该文档标记为 ``status=error`` 并给出明确文案，**不影响同批次其它文件**。

HTML 去标签逻辑与 06 ``store.ingest_document`` 保持一致（同一套正则），
避免两处解析行为漂移。
"""
from __future__ import annotations

import os
import re

SUPPORTED_EXTS = (".txt", ".md", ".markdown", ".html", ".htm")


class UnsupportedDocument(Exception):
    """文档类型不受支持（调用方转 status=error，非 HTTP 500）。"""


class EmptyDocument(Exception):
    """文档解析后无有效文本（空文件 / 全空白）。"""


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTS


def strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    return re.sub(r"\s+", " ", text)


def parse_bytes(filename: str, raw: bytes) -> str:
    """把上传文件字节解析为纯文本。

    Raises:
        UnsupportedDocument: 扩展名不在 :data:`SUPPORTED_EXTS` 内。
        EmptyDocument: 解析结果为空。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise UnsupportedDocument(
            f"暂不支持的文档类型：{ext or '(无扩展名)'}"
            f"（本期支持 {'/'.join(SUPPORTED_EXTS)}）"
        )
    text = raw.decode("utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        text = strip_html(text)
    if not text.strip():
        raise EmptyDocument("文档内容为空，无可解析文本。")
    return text


def make_summary(text: str, limit: int = 120) -> str:
    """取正文首段压缩为一行摘要，供前端「基于此文档提问」展示。"""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("…" if len(flat) > limit else "")
