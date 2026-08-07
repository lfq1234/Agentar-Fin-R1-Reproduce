"""08 个人文档：文本解析（纯文本 / HTML / PDF，二进制显式拒绝、不抛 500）。

约定（技术文档 §5.3）：
- ``.txt`` / ``.md`` / ``.markdown`` / ``.html`` / ``.htm`` → 解析为纯文本；
- ``.pdf`` → 用 ``pypdf`` 抽取每页文本（依赖 ``pypdf``，缺失时给出明确提示）；
- 其它扩展名（``.docx`` / ``.xlsx`` …）→ 抛 :class:`UnsupportedDocument`，
  调用方据此把该文档标记为 ``status=error`` 并给出明确文案，**不影响同批次其它文件**。

HTML 去标签逻辑与 06 ``store.ingest_document`` 保持一致（同一套正则），
避免两处解析行为漂移。
"""
from __future__ import annotations

import os
import re

SUPPORTED_EXTS = (".txt", ".md", ".markdown", ".html", ".htm", ".pdf")


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
        UnsupportedDocument: 扩展名不在 :data:`SUPPORTED_EXTS` 内（或 PDF 依赖缺失）。
        EmptyDocument: 解析结果为空（空文件 / 全空白 / 扫描件无文本层）。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise UnsupportedDocument(
            f"暂不支持的文档类型：{ext or '(无扩展名)'}"
            f"（本期支持 {'/'.join(SUPPORTED_EXTS)}）"
        )
    if ext == ".pdf":
        return _parse_pdf(raw)
    text = raw.decode("utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        text = strip_html(text)
    if not text.strip():
        raise EmptyDocument("文档内容为空，无可解析文本。")
    return text


def _parse_pdf(raw: bytes) -> str:
    """用 ``pypdf`` 抽取每页文本；缺库或扫描件时给明确报错。

    单页抽取异常不阻断整本；全本无文本层（图片型/扫描件 PDF）抛
    :class:`EmptyDocument` 交由调用方降级为 ``status=error``。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # 依赖缺失提示，而非 500
        raise UnsupportedDocument("PDF 解析依赖 pypdf 未安装（pip install pypdf）") from exc
    import io

    reader = PdfReader(io.BytesIO(raw))
    if not reader.pages:
        raise EmptyDocument("PDF 无页面内容。")
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")  # 单页失败不阻断整本
    text = "\n".join(p.strip() for p in parts).strip()
    if not text:
        raise EmptyDocument("PDF 未提取到可解析文本（可能是扫描件/图片型 PDF）。")
    return text


def make_summary(text: str, limit: int = 120) -> str:
    """取正文首段压缩为一行摘要，供前端「基于此文档提问」展示。"""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("…" if len(flat) > limit else "")
