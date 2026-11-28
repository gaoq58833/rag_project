"""测试大表跨块回填：分块打标 table_index + 检索期整表合并"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.chunker import ChunkStrategy, TextChunker
from src.rag.table_expand import merge_table_chunks


def _table(rows: int, tag: str) -> str:
    """生成一张足够长（>1000 字）的 Markdown 表，force 切块"""
    lines = ["| 编号 | 名称 | 说明 |", "| --- | --- | --- |"]
    for i in range(rows):
        lines.append(f"| {tag}-{i} | 设备{tag}{i} | 这是第 {i} 行的一段说明文字，用于把表格撑长 |")
    return "\n".join(lines)


def test_merge_table_chunks_dedupes_header_and_overlap():
    """表头只留一份、行级重叠去重、首块导语保留、前缀去掉"""
    chunk1 = (
        "【表格 · 第1-3行/共6行】\n"
        "下表是设备台账：\n"
        "| 编号 | 名称 |\n| --- | --- |\n| 1 | 甲 |\n| 2 | 乙 |\n| 3 | 丙 |"
    )
    chunk2 = (
        "【表格 · 第3-6行/共6行】\n"
        "| 编号 | 名称 |\n| --- | --- |\n| 3 | 丙 |\n| 4 | 丁 |\n| 5 | 戊 |\n| 6 | 己 |"
    )
    merged = merge_table_chunks([chunk1, chunk2])

    assert "下表是设备台账：" in merged, "首块导语应保留"
    assert "【表格" not in merged, "分块前缀应去掉"
    assert merged.count("| 编号 | 名称 |") == 1, "表头只应保留一份"
    assert merged.count("| 3 | 丙 |") == 1, "重叠行应去重"
    # 行顺序完整
    order = [merged.index(f"| {i} |") for i in range(1, 7)]
    assert order == sorted(order), "合并后行顺序应保持"


def test_merge_table_chunks_truncates_long_table():
    """超过上限时截断并注明"""
    body = "\n".join(f"| {i} | 第{i}行数据填充内容 |" for i in range(300))
    chunk = f"【表格 · 第1-300行/共300行】\n| 编号 | 内容 |\n| --- | --- |\n{body}"
    merged = merge_table_chunks([chunk], max_chars=500)

    assert len(merged) < 600
    assert merged.endswith("（表格过长，已截断，仅保留前部分行）")


def test_chunker_tags_same_table_with_same_index():
    """同一张表被切开后子块共享 table_index，第二张表递增"""
    text = "# 台账\n\n" + _table(40, "A") + "\n\n中间说明段落。\n\n" + _table(40, "B")
    chunker = TextChunker(
        strategy=ChunkStrategy.PARAGRAPH,
        max_chunk_size=1000,
        min_chunk_size=100,
        chunk_overlap=200,
    )
    chunks = chunker.split(text)
    tables = [c for c in chunks if c["type"] == "table"]

    assert len(tables) > 2, f"两张长表都应被切开，实际表格块 {len(tables)}"
    first = [c for c in tables if "设备A" in c["content"]]
    second = [c for c in tables if "设备B" in c["content"]]
    assert first and second
    assert len({c["table_index"] for c in first}) == 1, "表 A 的子块应共享 table_index"
    assert len({c["table_index"] for c in second}) == 1, "表 B 的子块应共享 table_index"
    assert first[0]["table_index"] != second[0]["table_index"], "两张表的序号应不同"


def test_roundtrip_chunker_and_merge():
    """分块器切开的整表经回填后能还原全部数据行"""
    text = _table(40, "C")
    chunker = TextChunker(
        strategy=ChunkStrategy.PARAGRAPH,
        max_chunk_size=1000,
        min_chunk_size=100,
        chunk_overlap=200,
    )
    parts = [c["content"] for c in chunker.split(text) if c["type"] == "table"]
    assert len(parts) > 1

    merged = merge_table_chunks(parts)
    for i in range(40):
        assert f"| C-{i} |" in merged, f"第 {i} 行丢失"


if __name__ == "__main__":
    test_merge_table_chunks_dedupes_header_and_overlap()
    test_merge_table_chunks_truncates_long_table()
    test_chunker_tags_same_table_with_same_index()
    test_roundtrip_chunker_and_merge()
    print("[OK] 表格回填全部用例通过")
