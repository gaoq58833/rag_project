from pathlib import Path
from typing import List

from src.parsers.base import BaseParser, ParseResult
from src.parsers.cleaner import TextCleaner


class TXTParser(BaseParser):
    """纯文本（.txt）解析器

    TXT 文档结构化弱（无标题语法、无表格/图片标记），因此：
    - 分块采用段落策略（按空行分段 → 超限按句子 → 再超限按字符），不做标题入栈；
    - 编码不固定，按 utf-8 → utf-8-sig → gb18030 顺序回退解码。
    """

    def __init__(self, image_dir: str = "data/images"):
        # TXT 没有图片，image_dir 仅为接口统一
        self.image_dir = image_dir
        self.cleaner = TextCleaner()

    def parse(self, file_path: str) -> ParseResult:
        raw = Path(file_path).read_bytes()
        text = self._decode(raw)
        return self._build(text, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        text = self._decode(file_bytes)
        return self._build(text, filename, filename)

    @staticmethod
    def _decode(raw: bytes) -> str:
        """TXT 编码不固定，优先 utf-8，失败回退 utf-8-sig / gb18030（中文文本常见）"""
        for enc in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _build(self, text: str, filename: str, file_path: str) -> ParseResult:
        # 保守清洗：去页码/分隔线、零宽字符；不做 normalize_whitespace（保留段落空行）
        cleaned = self.cleaner.clean(text)
        cleaned = self.cleaner.remove_extra_chars(cleaned)

        return ParseResult(
            filename=filename,
            total_pages=0,            # 纯文本无页概念
            total_chunks=len(cleaned),
            table_count=0,            # TXT 无表格结构
            chunks=[],
            raw_text=cleaned,
            tables=[],
            metadata={"parser": "txt", "file_path": file_path},
        )

    def supported_extensions(self) -> List[str]:
        return [".txt", ".text"]
