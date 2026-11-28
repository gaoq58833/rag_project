"""表格子块回填：大表被按行窗口切开后，检索命中任一子块时还原整表。

分块器切表时每个子块都带 `【表格 · 第a-b行/共N行】` 前缀和重复表头，
相邻子块还有一行重叠。本模块只做纯文本合并，便于单测。
"""
from typing import List, Optional

# 表格整表回填的最大字符数：超过则截断，避免上下文被一张超大表占满
TABLE_CONTEXT_MAX_CHARS = 6000

_SEPARATOR_CHARS = set("|:- ")


def _is_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and set(s) <= _SEPARATOR_CHARS


def _strip_prefix(lines: List[str]) -> List[str]:
    if lines and lines[0].startswith("【表格"):
        return lines[1:]
    return lines


def _first_pipe_index(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            return i
    return 0


def merge_table_chunks(contents: List[str], max_chars: Optional[int] = TABLE_CONTEXT_MAX_CHARS) -> str:
    """把同一张表的子块按顺序合并成整表：表头只保留一份，去掉行级重叠。"""
    merged: List[str] = []
    for idx, content in enumerate(contents):
        lines = _strip_prefix([line.rstrip() for line in content.split("\n")])
        if not lines:
            continue
        if idx == 0:
            keep = lines
        else:
            # 后续子块：跳过重复的表头与分隔线，只保留数据行
            start = _first_pipe_index(lines) + 1
            if start < len(lines) and _is_separator(lines[start]):
                start += 1
            keep = lines[start:]
        for line in keep:
            if not line.strip():
                continue
            # 相邻子块带上一块的最后一行，连续重复时只留一份
            if merged and line.lstrip().startswith("|") and merged[-1] == line:
                continue
            merged.append(line)

    text = "\n".join(merged)
    if max_chars and len(text) > max_chars:
        out: List[str] = []
        size = 0
        for line in merged:
            if size + len(line) + 1 > max_chars:
                break
            out.append(line)
            size += len(line) + 1
        text = "\n".join(out) + "\n（表格过长，已截断，仅保留前部分行）"
    return text
