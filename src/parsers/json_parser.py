"""JSON 解析器：把 JSON 摊平成"一条记录一个 chunk"，供 BM25 倒排索引使用

JSON 结构强、字段名可精确匹配，因此不走向量索引（INDEX_TYPE = "bm25"）：
- 顶层数组的每个元素、顶层对象的每个 key（值为对象数组时按元素展开）各成一个 chunk
- 字段渲染为 `key: value`，camelCase / snake_case 额外拆出空格分词形式，保证 BM25 能按词命中
- section 记录 JSON 路径（如 users[3]），便于回答时溯源

分块直接由解析器给出（ParseResult.chunks），入库时不再交给 TextChunker 二次切分，
避免多条记录被合并进同一个 chunk 稀释 ts_rank 排序。
"""
import json
import re
from pathlib import Path
from typing import Any, Generator, List, Tuple

from src.parsers.base import BaseParser, ParseResult

MAX_DEPTH = 10          # 嵌套渲染深度上限，防止超深 JSON 撑爆单块
MAX_VALUE_LEN = 1000    # 单个字符串值的长度上限（base64 / 长文截断）
MAX_CHUNK_SIZE = 1000   # 单块字符上限，超出的记录按字段行继续切

# camelCase 边界：小写/数字后跟大写时插空格
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


class JSONParser(BaseParser):
    """JSON 解析器：只建倒排索引，不做向量化"""

    INDEX_TYPE = "bm25"

    def __init__(self, image_dir: str = "data/images"):
        # JSON 没有图片，image_dir 仅为接口统一
        self.image_dir = image_dir

    def parse(self, file_path: str) -> ParseResult:
        raw = Path(file_path).read_bytes()
        return self._build(raw, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        return self._build(file_bytes, filename, filename)

    @staticmethod
    def _decode(raw: bytes) -> str:
        """JSON 标准是 UTF-8，但常带 BOM，按 utf-8-sig → utf-8 → gb18030 回退"""
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _build(self, raw: bytes, filename: str, file_path: str) -> ParseResult:
        text = self._decode(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：第 {e.lineno} 行第 {e.colno} 列，{e.msg}") from e
        except RecursionError as e:
            raise ValueError("JSON 嵌套层级过深，无法解析") from e

        chunks: List[dict] = []
        unit_count = 0
        for section, value in self._iter_units(data):
            unit_count += 1
            self._emit(section, self._render(value), chunks)

        raw_text = "\n\n".join(
            f"## {c['section']}\n{c['content']}" if c["section"] else c["content"]
            for c in chunks
        )

        metadata = {
            "parser": "json",
            "index_type": self.INDEX_TYPE,
            "file_path": file_path,
            "root_type": type(data).__name__,
            "unit_count": unit_count,
        }
        if isinstance(data, dict):
            metadata["top_level_keys"] = [str(k) for k in list(data.keys())[:50]]

        return ParseResult(
            filename=filename,
            total_pages=0,
            total_chunks=len(chunks),
            table_count=0,          # 记录渲染成文本，不产出 Markdown 表格
            chunks=chunks,
            raw_text=raw_text,
            tables=[],
            metadata=metadata,
        )

    def _iter_units(self, data: Any) -> Generator[Tuple[str, Any], None, None]:
        """把 JSON 拆成 (section, 记录) 单元：一条记录一个 chunk

        - 顶层数组：元素含对象/数组时逐个展开，纯标量数组整体算一条
        - 顶层对象：每个 key 一条；值为对象/数组元素组成的数组时按元素展开
        """
        if isinstance(data, list):
            if any(isinstance(x, (dict, list)) for x in data):
                for i, item in enumerate(data):
                    yield f"root[{i}]", item
            else:
                yield "root", data
            return

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
                    for i, item in enumerate(value):
                        yield f"{key}[{i}]", item
                else:
                    yield str(key), value
            return

        yield "root", data

    def _render(self, value: Any, path: str = "", depth: int = 0) -> List[str]:
        """递归渲染成行：叶子是 `path: value`，空容器也给出占位，避免信息丢失"""
        if depth > MAX_DEPTH:
            return [f"{path}: ...(嵌套过深，已省略)" if path else "...(嵌套过深，已省略)"]

        if isinstance(value, dict):
            if not value:
                return [f"{path}: {{}}" if path else "{}"]
            lines: List[str] = []
            for key, child in value.items():
                child_path = f"{path}.{self._key_label(key)}" if path else self._key_label(key)
                lines.extend(self._render(child, child_path, depth + 1))
            return lines

        if isinstance(value, list):
            if not value:
                return [f"{path}: []" if path else "[]"]
            # 纯标量数组压成一行，避免 tags 之类的字段被拆成几十行
            if all(not isinstance(x, (dict, list)) for x in value):
                items = ", ".join(self._scalar(x) for x in value)
                return [f"{path}: {items}" if path else items]
            lines = []
            for i, item in enumerate(value):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                lines.extend(self._render(item, child_path, depth + 1))
            return lines

        return [f"{path}: {self._scalar(value)}" if path else self._scalar(value)]

    @staticmethod
    def _key_label(key: Any) -> str:
        """字段名拆词：camelCase / snake_case 额外给出空格形式，兼顾字面精确匹配与分词召回"""
        raw = str(key)
        normalized = re.sub(r"[_\-]+", " ", raw)
        normalized = _CAMEL_RE.sub(r"\1 \2", normalized).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        # 已经等于分词形式（如纯中文、全小写单词）就不重复输出
        if normalized and normalized != raw.lower():
            return f"{raw} ({normalized})"
        return raw

    @staticmethod
    def _scalar(value: Any) -> str:
        """标量转字符串：布尔用 true/false，超长字符串截断，换行压成空格"""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).replace("\n", " ").replace("\r", " ").strip()
        if len(text) > MAX_VALUE_LEN:
            text = text[:MAX_VALUE_LEN] + "…(已截断)"
        return text

    def _emit(self, section: str, lines: List[str], chunks: List[dict]) -> None:
        """把一条记录的行渲染成 1 个或多个 chunk；超过 MAX_CHUNK_SIZE 时按行窗口切"""
        content = "\n".join(line.strip() for line in lines if line.strip())
        if not content:
            return

        if len(content) <= MAX_CHUNK_SIZE:
            pieces = [content]
        else:
            pieces = []
            buf = ""
            for line in content.split("\n"):
                if buf and len(buf) + 1 + len(line) > MAX_CHUNK_SIZE:
                    pieces.append(buf)
                    buf = line
                else:
                    buf = f"{buf}\n{line}" if buf else line
                # 单行本身超长（如超长字符串兜底），按字符切
                while len(buf) > MAX_CHUNK_SIZE:
                    pieces.append(buf[:MAX_CHUNK_SIZE])
                    buf = buf[MAX_CHUNK_SIZE:]
            if buf:
                pieces.append(buf)

        for piece in pieces:
            chunks.append({
                "content": piece,
                "index": len(chunks),
                "length": len(piece),
                "type": "text",
                "section": section,
            })

    def supported_extensions(self) -> List[str]:
        return [".json"]
