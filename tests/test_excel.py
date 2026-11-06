"""测试 Excel/CSV 解析：sheet → Markdown 表格 + 行组预分块

用法：
  python tests/test_excel.py

验证点：
- 工厂按 .csv/.xlsx/.xls 路由到 ExcelParser，索引类型为 vector
- csv 多编码回退（utf-8-sig / gb18030），分隔符嗅探
- xlsx 多 sheet 逐个解析，section 标注 sheet 名与行区间
- xls 走 xlrd 引擎
- 行组切分：每块重复表头，单块不超 MAX_CHUNK_CHARS + 单行
- 单元格清洗：NaN 置空、换行/竖线压平转义
- 全空文件抛 ValueError
"""
import io
import os
import sys
import tempfile

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.excel_parser import MAX_CHUNK_CHARS, ExcelParser
from src.parsers.factory import get_parser, index_type_for

CSV_TEXT = "姓名,部门,工资\n张三,技术部,15000\n李四,产品部,18000\n"


def make_xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({
            "姓名": ["张三", "李四"],
            "部门": ["技术部", "产品部"],
            "工资": [15000, 18000],
        }).to_excel(writer, sheet_name="员工", index=False)
        pd.DataFrame({
            "产品": ["键盘", "鼠标"],
            "库存": [100, None],
        }).to_excel(writer, sheet_name="产品", index=False)
    return buf.getvalue()


def make_xls_bytes() -> bytes:
    import xlwt
    wb = xlwt.Workbook()
    ws = wb.add_sheet("数据")
    ws.write(0, 0, "编号")
    ws.write(0, 1, "名称")
    ws.write(1, 0, 1)
    ws.write(1, 1, "服务器")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_factory_routing():
    for ext in (".csv", ".xlsx", ".xls"):
        parser = get_parser(f"data{ext}")
        assert isinstance(parser, ExcelParser), (ext, type(parser))
        assert index_type_for(f"data{ext}") == "vector"
    assert ExcelParser().supported_extensions() == [".csv", ".xlsx", ".xls"]
    assert ExcelParser.INDEX_TYPE == "vector"


def test_csv_parse():
    result = ExcelParser().parse_from_bytes(CSV_TEXT.encode("utf-8-sig"), "staff.csv")
    assert result.table_count == 1
    assert result.total_pages == 1
    assert len(result.chunks) == 1

    chunk = result.chunks[0]
    assert chunk["type"] == "table"
    assert chunk["section"] == "csv"
    lines = chunk["content"].split("\n")
    assert lines[0] == "| 姓名 | 部门 | 工资 |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 张三 | 技术部 | 15000 |"

    assert result.tables[0]["page"] == "csv"
    assert "## csv" in result.raw_text
    assert result.metadata["parser"] == "excel"
    assert result.metadata["sheets"] == ["csv"]
    assert result.metadata["total_rows"] == 2


def test_csv_gbk():
    result = ExcelParser().parse_from_bytes(CSV_TEXT.encode("gb18030"), "staff.csv")
    assert "张三" in result.chunks[0]["content"]
    assert "技术部" in result.chunks[0]["content"]


def test_csv_semicolon():
    text = "a;b\n1;2\n"
    result = ExcelParser().parse_from_bytes(text.encode("utf-8"), "semi.csv")
    assert result.chunks[0]["content"].split("\n")[0] == "| a | b |"


def test_xlsx_multisheet():
    result = ExcelParser().parse_from_bytes(make_xlsx_bytes(), "book.xlsx")
    assert result.total_pages == 2
    assert result.table_count == 2
    assert result.metadata["sheets"] == ["员工", "产品"]
    assert result.metadata["total_rows"] == 4

    sections = [c["section"] for c in result.chunks]
    assert sections == ["员工", "产品"]
    # NaN 单元格置空
    product = result.chunks[1]["content"]
    assert "| 鼠标 |  |" in product
    # raw_text 按 sheet 分节
    assert "## 员工" in result.raw_text
    assert "## 产品" in result.raw_text


def test_xls_parse():
    result = ExcelParser().parse_from_bytes(make_xls_bytes(), "legacy.xls")
    assert result.metadata["sheets"] == ["数据"]
    chunk = result.chunks[0]
    assert chunk["section"] == "数据"
    assert "| 编号 | 名称 |" in chunk["content"]
    assert "| 1 | 服务器 |" in chunk["content"]


def test_row_group_chunking():
    # 每行约 60 字符，100 行必然分多块
    df = pd.DataFrame({
        "编号": list(range(1, 101)),
        "描述": ["这是一段用于凑长度的产品描述文本" * 2] * 100,
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    result = ExcelParser().parse_from_bytes(buf.getvalue(), "big.xlsx")

    assert len(result.chunks) > 1
    header = "| 编号 | 描述 |"
    for i, chunk in enumerate(result.chunks):
        lines = chunk["content"].split("\n")
        # 每块都重复表头
        assert lines[0] == header
        assert lines[1] == "| --- | --- |"
        assert chunk["section"].startswith("Sheet1 第")
        assert chunk["section"].endswith("行")
        # 单块长度不超上限 + 一行
        assert chunk["length"] <= MAX_CHUNK_CHARS + 100, (i, chunk["length"])
    # 行区间连续覆盖 1-100
    assert "第1-" in result.chunks[0]["section"]
    assert result.chunks[-1]["section"].endswith("-100行")


def test_cell_cleaning():
    df = pd.DataFrame({
        "备注": ["第一行\n第二行", "含|竖线", None],
        "值": [1, 2, 3],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    result = ExcelParser().parse_from_bytes(buf.getvalue(), "dirty.xlsx")

    content = result.chunks[0]["content"]
    assert "第一行 第二行" in content       # 换行压平
    assert "含\\|竖线" in content           # 竖线转义
    assert "|  | 3 |" in content            # None 置空


def test_empty_raises():
    # 完全空的 csv
    try:
        ExcelParser().parse_from_bytes(b"", "empty.csv")
    except ValueError as e:
        assert "CSV" in str(e)
    else:
        raise AssertionError("空 csv 未抛 ValueError")

    # 全空 sheet 的 xlsx
    buf = io.BytesIO()
    pd.DataFrame().to_excel(buf, index=False, engine="openpyxl")
    try:
        ExcelParser().parse_from_bytes(buf.getvalue(), "empty.xlsx")
    except ValueError as e:
        assert "没有可解析的表格数据" in str(e)
    else:
        raise AssertionError("空 xlsx 未抛 ValueError")


def test_parse_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "staff.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(CSV_TEXT)
        result = ExcelParser().parse(path)
        assert result.filename == "staff.csv"
        assert len(result.chunks) == 1
        assert result.metadata["file_path"] == path


def main():
    tests = [
        test_factory_routing,
        test_csv_parse,
        test_csv_gbk,
        test_csv_semicolon,
        test_xlsx_multisheet,
        test_xls_parse,
        test_row_group_chunking,
        test_cell_cleaning,
        test_empty_raises,
        test_parse_file,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    result = ExcelParser().parse_from_bytes(make_xlsx_bytes(), "book.xlsx")
    print(f"\n块数：{len(result.chunks)}，sheet：{result.metadata['sheets']}")
    for c in result.chunks:
        print(f"  [{c['index']}] section={c['section']!r} type={c['type']} length={c['length']}")
        print("      " + c["content"].replace("\n", "\n      "))


if __name__ == "__main__":
    main()
