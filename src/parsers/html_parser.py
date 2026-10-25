import io
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from src.parsers.base import BaseParser, ParseResult
from src.parsers.cleaner import TextCleaner
from src.parsers.table_parser import TableProcessor


class _CodeBlockConverter(MarkdownConverter):
    """markdownify 转换器：<pre><code> 输出带语言标识的围栏代码块"""

    def convert_pre(self, el, text, parent_tags):
        code = el.find("code")
        lang = ""
        if code is not None:
            for cls in code.get("class", []) or []:
                if cls.startswith("language-"):
                    lang = cls[len("language-"):]
                    break
        inner = el.get_text()
        return f"\n```{lang}\n{inner}\n```\n"


class HTMLParser(BaseParser):
    """HTML 解析器：用 markdownify 把 HTML 转成 markdown，复用现有清洗与分块逻辑"""

    def __init__(self, image_dir: str = "data/images"):
        self.image_dir = image_dir
        self.cleaner = TextCleaner()
        self.table_processor = TableProcessor()

    def parse(self, file_path: str) -> ParseResult:
        raw = Path(file_path).read_bytes()
        text = self._decode(raw)
        return self._build(text, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        text = self._decode(file_bytes)
        return self._build(text, filename, filename)

    @staticmethod
    def _decode(raw: bytes) -> str:
        """HTML 编码不固定，优先 utf-8，失败回退 gb18030（中文网页常见）"""
        for enc in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _normalize_code_blocks(html: str) -> str:
        """把各种代码容器（<pre>、<div class="code-block"> 等）归一化为 <pre><code>，
        便于 markdownify 统一输出围栏代码块"""
        soup = BeautifulSoup(html, "lxml")
        # 匹配 <pre> 以及 class 含 "code" 的元素（code-block、highlight、source 等）
        for el in soup.find_all(True):
            classes = el.get("class", []) or []
            is_code = el.name == "pre" or any("code" in c.lower() for c in classes)
            if not is_code:
                continue
            # 提取语言：class 里的 language-xxx
            lang = ""
            for cls in classes:
                m = re.match(r"language-(\w+)", cls)
                if m:
                    lang = m.group(1)
                    break
            inner = el.get_text()
            new_tag = soup.new_tag("pre")
            code_tag = soup.new_tag("code")
            if lang:
                code_tag["class"] = f"language-{lang}"
            code_tag.string = inner
            new_tag.append(code_tag)
            el.replace_with(new_tag)
        return str(soup)

    def _build(self, html: str, filename: str, file_path: str) -> ParseResult:
        # 1. 归一化代码块
        html = self._normalize_code_blocks(html)
        # 2. markdownify 转换：h1-h6 → ATX 标题，table → markdown 表格，img → 图片引用，pre → 围栏代码
        markdown = _CodeBlockConverter(
            heading_style="ATX",
            bullets="-",
            strip=[],
        ).convert(html)
        # 3. 保守清洗：按行去页码/分隔线 + 去零宽字符；不用 normalize_whitespace，避免毁掉表格换行
        cleaned = self.cleaner.clean(markdown)
        cleaned = self.cleaner.remove_extra_chars(cleaned)

        tables = self._extract_tables(cleaned)
        return ParseResult(
            filename=filename,
            total_pages=0,
            total_chunks=len(cleaned),
            table_count=len(tables),
            chunks=[],
            raw_text=cleaned,
            tables=tables,
            metadata={"parser": "markdownify", "file_path": file_path},
        )

    def _extract_tables(self, text: str) -> List[dict]:
        """从 markdown 里抽出所有表格块，复用 TableProcessor 规范化"""
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
        return [".html", ".htm"]
