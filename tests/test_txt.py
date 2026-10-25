"""测试 TXT 解析 + 分块：TXTParser → TextChunker（段落策略）

用法：
  python tests/test_txt.py                        默认测 test_file/sample.txt
  python tests/test_txt.py <txt> <输出目录>         测指定文档

产出 chunks.json 与 stats.json，供量化评分使用。
"""
import sys
import os
import re
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TXTParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = os.path.join(ROOT, "test_file", "sample.txt")
OUT = os.path.join(ROOT, "data", "txt_test")
if len(sys.argv) > 1:
    TXT = os.path.abspath(sys.argv[1])
if len(sys.argv) > 2:
    OUT = os.path.join(ROOT, sys.argv[2])


def squeeze(text):
    """去掉所有空白，用于内容覆盖率统计"""
    return re.sub(r'[\s]+', '', text)


def main():
    os.makedirs(OUT, exist_ok=True)

    result = TXTParser().parse(TXT)
    text = result.raw_text
    out_txt = os.path.join(OUT, os.path.splitext(os.path.basename(TXT))[0] + ".txt")
    open(out_txt, "w", encoding="utf-8").write(text)
    print(f"清洗后的文本已保存：{out_txt}")

    # ---- 解析准确率 ----
    raw = open(TXT, encoding="utf-8", errors="ignore").read()
    src = Counter(squeeze(raw))
    cleaned = Counter(squeeze(text))
    missing = sum((src - cleaned).values())
    extra = sum((cleaned - src).values())
    total = sum(src.values())
    print(f"解析覆盖率：原文 {total} 字，清洗后 {total - missing} 字"
          f"（{(total - missing) / total * 100:.2f}%），多出 {extra} 字")

    # ---- 分块（段落策略，不做标题入栈）----
    from src.parsers.chunker import TextChunker, ChunkStrategy
    chunker = TextChunker(
        strategy=ChunkStrategy.PARAGRAPH,
        max_chunk_size=1000,
        min_chunk_size=100,
        chunk_overlap=200,
    )
    chunks = chunker.split(text)

    sizes = [c["length"] for c in chunks]
    texts = [c for c in chunks if c["type"] == "text"]
    over = [s for s in sizes if s > chunker.max_chunk_size]
    under = [s for s in sizes if s < chunker.min_chunk_size]

    covered = Counter()
    for c in chunks:
        covered.update(squeeze(c["content"]))
    missing_chars = sum((src - covered).values())
    dup_chars = sum((covered - src).values())

    print(f"切出块数：{len(chunks)}（正文 {len(texts)} / 表格 0）")
    print(f"块长度：min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")
    print(f"超过 max 的块：{len(over)}  低于 min 的块：{len(under)}")
    print(f"字符覆盖：丢失 {missing_chars} 字，重复 {dup_chars} 字")

    titled = [c for c in texts if c.get("section")]
    print(f"正文块带章节路径：{len(titled)}/{len(texts)}（TXT 无标题结构，预期为 0）")

    json.dump(chunks, open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({
        "source": f"{os.path.basename(TXT)}（TXTParser 解析）",
        "chunks_total": len(chunks),
        "text_count": len(texts),
        "table_count": 0,
        "min_len": min(sizes) if sizes else 0,
        "max_len": max(sizes) if sizes else 0,
        "avg_len": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "over_limit": len(over),
        "under_min": len(under),
        "table_no_header": 0,
        "text_with_section": len(titled),
        "missing_chars": missing_chars,
        "dup_chars": dup_chars,
        "src_chars": total,
        "parse_coverage": round((total - missing) / total * 100, 2) if total else 0,
        "max_chunk_size": chunker.max_chunk_size,
        "min_chunk_size": chunker.min_chunk_size,
    }, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
