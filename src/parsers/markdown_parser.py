import io
from pathlib import Path
from typing import List

from src.parsers.base import BaseParser, ParseResult
from src.parsers.cleaner import TextCleaner
from src.parsers.table_parser import TableProcessor


class MarkdownParser(BaseParser):
    """Markdown 解析器：直接读取 .md，复用现有清洗与分块逻辑，无需模型"""

    def __init__(self, image_dir: str = "data/images"):
        # markdown 里图片已经是路径引用，不需要重新提取；image_dir 仅为接口统一
        self.image_dir = image_dir
        self.cleaner = TextCleaner()
        self.table_processor = TableProcessor()

    def parse(self, file_path: str) -> ParseResult:
        text = Path(file_path).read_text(encoding="utf-8")
        return self._build(text, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        text = file_bytes.decode("utf-8")
        return self._build(text, filename, filename)

    def _build(self, text: str, filename: str, file_path: str) -> ParseResult:
        # 只做保守清洗：按行去掉页码/分隔线、去零宽字符；
        # 不用 normalize_whitespace——它会把换行压成空格，毁掉表格和标题结构
        cleaned = self.cleaner.clean(text)
        cleaned = self.cleaner.remove_extra_chars(cleaned)

        tables = self._extract_tables(cleaned)
        return ParseResult(
            filename=filename,
            total_pages=0,           # markdown 没有页的概念
            total_chunks=len(cleaned),
            table_count=len(tables),
            chunks=[],
            raw_text=cleaned,
            tables=tables,
            metadata={"parser": "markdown", "file_path": file_path},
        )

    def _extract_tables(self, text: str) -> List[dict]:
        """从 markdown 里抽出所有表格块，复用 TableProcessor 做规范化（去重复表头/多余分隔行）"""
        tables = []
        lines = text.split("\n")
        i = 0
        n = 0
        while i < len(lines):
            if lines[i].strip().startswith("|"):
                block = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                markdown = self.table_processor._data_lines_markdown("\n".join(block))
                n += 1
                tables.append({"table_number": n, "page": None, "markdown": markdown})
            else:
                i += 1
        return tables

    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]
