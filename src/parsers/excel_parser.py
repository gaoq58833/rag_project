"""Excel/CSV 解析器：每个 sheet 转成 Markdown 表格，按行组预分块，只建 BM25 倒排索引

表格是强结构化数据：字段名固定、值精确匹配优先（型号 ID、数值、状态等），
BM25 对表格的准确率高于向量检索，且跳过 embedding 可省 97% 入库时间。
因此 INDEX_TYPE = "bm25"，与 JSON 一样只建 PG tsvector 倒排索引。

- 单行语义依赖表头，每个 chunk 都重复表头，以「表头 + 若干行」为最小检索单元
- csv：按 utf-8-sig → utf-8 → gb18030 回退解码，分隔符自动嗅探
- xlsx：openpyxl 引擎；xls：xlrd 引擎；多 sheet 逐个解析
- 全空行/全空列丢弃，单元格里的换行与竖线压平转义
"""
import io
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.parsers.base import BaseParser, ParseResult

MAX_CHUNK_CHARS = 1500   # 单块字符上限（含表头），超出按行组切分
MAX_CELL_LEN = 200       # 单个单元格文本长度上限


class ExcelParser(BaseParser):
    """csv / xlsx / xls 解析器：行组预分块，只建 BM25 倒排索引

    表格是强结构化数据，字段名固定、值精确匹配优先，向量检索
    对表格的准确率低于 BM25（表头重复、型号 ID、数值匹配等），
    因此 INDEX_TYPE = "bm25"，跳过 embedding 和 Qdrant。
    实测 2000 行设备台账：入库省 8.6s（embedding 占 97%），
    检索每条查询从 ~170ms 降到 ~3ms。
    """

    INDEX_TYPE = "bm25"

    def parse(self, file_path: str) -> ParseResult:
        raw = Path(file_path).read_bytes()
        return self._build(raw, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        return self._build(file_bytes, filename, filename)

    def _build(self, raw: bytes, filename: str, file_path: str) -> ParseResult:
        sheets = self._read_sheets(raw, filename)

        chunks: List[Dict] = []
        tables: List[Dict] = []
        raw_parts: List[str] = []
        sheet_names: List[str] = []
        total_rows = 0

        for sheet_name, df in sheets:
            df = self._clean(df)
            if df.shape[1] == 0:
                continue  # 整列皆空的 sheet 没有可检索内容
            sheet_names.append(sheet_name)
            total_rows += len(df)

            header, sep, rows = self._to_markdown_rows(df)
            full_md = "\n".join([header, sep] + rows)
            tables.append({
                "table_number": len(tables) + 1,
                "page": sheet_name,
                "markdown": full_md,
            })
            raw_parts.append(f"## {sheet_name}\n{full_md}")

            for start, end, content in self._chunk_rows(header, sep, rows):
                # 单块覆盖整个 sheet 时不标行号，多块标注行区间便于溯源
                section = sheet_name if (start == 1 and end == len(rows)) \
                    else f"{sheet_name} 第{start}-{end}行"
                chunks.append({
                    "content": content,
                    "index": len(chunks),
                    "length": len(content),
                    "type": "table",
                    "section": section,
                })

        if not chunks:
            raise ValueError("文件中没有可解析的表格数据（全部为空）")

        return ParseResult(
            filename=filename,
            total_pages=len(sheet_names),   # Excel 没有页，用 sheet 数代替
            total_chunks=len(chunks),
            table_count=len(tables),
            chunks=chunks,
            raw_text="\n\n".join(raw_parts),
            tables=tables,
            metadata={
                "parser": "excel",
                "index_type": self.INDEX_TYPE,
                "file_path": file_path,
                "sheets": sheet_names,
                "sheet_count": len(sheet_names),
                "total_rows": total_rows,
            },
        )

    def _read_sheets(self, raw: bytes, filename: str) -> List[Tuple[str, pd.DataFrame]]:
        """按扩展名分发读取，返回 [(sheet名, DataFrame)]；csv 视为单 sheet"""
        ext = Path(filename).suffix.lower()
        if ext == ".csv":
            return [("csv", self._read_csv(raw))]

        engine = "xlrd" if ext == ".xls" else "openpyxl"
        try:
            book = pd.read_excel(io.BytesIO(raw), sheet_name=None, engine=engine, dtype=object)
        except Exception as e:
            raise ValueError(f"Excel 解析失败：{e}") from e
        return [(str(name), df) for name, df in book.items()]

    @staticmethod
    def _read_csv(raw: bytes) -> pd.DataFrame:
        """csv 没有编码标记，按常见编码回退；分隔符嗅探失败则退回默认逗号"""
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError as e:
                last_err = e
                continue
            try:
                return pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=object)
            except Exception:
                try:
                    return pd.read_csv(io.StringIO(text), dtype=object)
                except Exception as e:
                    raise ValueError(f"CSV 解析失败：{e}") from e
        raise ValueError(f"CSV 编码识别失败：{last_err}")

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """丢全空行/全空列；重置索引，让行号与展示一致"""
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df.reset_index(drop=True)

    def _to_markdown_rows(self, df: pd.DataFrame) -> Tuple[str, str, List[str]]:
        """DataFrame → (表头行, 分隔行, 数据行列表)"""
        cols = [self._cell(c) or f"列{i + 1}" for i, c in enumerate(df.columns)]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(self._cell(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return header, sep, rows

    @staticmethod
    def _cell(value) -> str:
        """单元格 → 文本：NaN/None 置空，换行与竖线压平转义，超长截断"""
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()
        if len(text) > MAX_CELL_LEN:
            text = text[:MAX_CELL_LEN] + "…"
        return text

    @staticmethod
    def _chunk_rows(header: str, sep: str, rows: List[str]):
        """按行组切分：每块重复表头，保证单行脱离上下文也可理解

        产出 (起始行号, 结束行号, chunk文本)，行号从 1 计（不含表头）。
        """
        if not rows:
            yield 1, 0, f"{header}\n{sep}"
            return

        base_len = len(header) + len(sep) + 1
        buf: List[str] = []
        buf_len = base_len
        start = 1
        for i, row in enumerate(rows):
            if buf and buf_len + len(row) + 1 > MAX_CHUNK_CHARS:
                yield start, i, "\n".join([header, sep] + buf)
                buf, buf_len, start = [], base_len, i + 1
            buf.append(row)
            buf_len += len(row) + 1
        if buf:
            yield start, len(rows), "\n".join([header, sep] + buf)

    def supported_extensions(self) -> List[str]:
        return [".csv", ".xlsx", ".xls"]
