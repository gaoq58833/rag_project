"""测试 Markdown 解析 + 分块：MarkdownParser → TextChunker

用法：
  python tests/test_markdown.py                        默认测 test_file 下的 markdown
  python tests/test_markdown.py <md> <输出目录>         测指定文档

产出 chunks.json 与 stats.json，供量化评分使用。
"""
import sys
import os
import re
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.markdown_parser import MarkdownParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "test_file", "基于MindSpore的RNN情感分类实验.md")
OUT = os.path.join(ROOT, "data", "md_test")
if len(sys.argv) > 1:
    MD = os.path.abspath(sys.argv[1])
if len(sys.argv) > 2:
    OUT = os.path.join(ROOT, sys.argv[2])


def squeeze(text):
    """去掉 markdown 标记和所有空白，用于内容覆盖率统计"""
    return re.sub(r'[\s]+', '', re.sub(r'[`*#>\-|!\[\]():/_.]', '', text))


def main():
    os.makedirs(OUT, exist_ok=True)

    result = MarkdownParser().parse(MD)
    text = result.raw_text
    md_path = os.path.join(OUT, os.path.splitext(os.path.basename(MD))[0] + ".md")
    open(md_path, "w", encoding="utf-8").write(text)
    print(f"清洗后的 markdown 已保存：{md_path}")

    # ---- 解析准确率：清洗后内容 vs 原始文件内容 ----
    raw = open(MD, encoding="utf-8").read()
    src = Counter(squeeze(raw))
    md = Counter(squeeze(text))
    missing = sum((src - md).values())
    extra = sum((md - src).values())
    total = sum(src.values())
    print(f"解析覆盖率：原文 {total} 字，清洗后 {total - missing} 字"
          f"（{(total - missing) / total * 100:.2f}%），多出 {extra} 字")
    print(f"（被清洗掉的主要是页码/分隔线等噪音行）")

    # ---- 分块 ----
    from src.parsers.chunker import TextChunker, ChunkStrategy
    chunker = TextChunker(
        strategy=ChunkStrategy.PARAGRAPH,
        max_chunk_size=1000,
        min_chunk_size=100,
        chunk_overlap=200,
    )
    chunks = chunker.split(text)

    sizes = [c["length"] for c in chunks]
    tables = [c for c in chunks if c["type"] == "table"]
    texts = [c for c in chunks if c["type"] == "text"]
    over = [s for s in sizes if s > chunker.max_chunk_size]
    under = [s for s in sizes if s < chunker.min_chunk_size]
    no_header = [c for c in tables if "\n|" not in c["content"]]
    with_img = [c for c in chunks if c.get("images")]

    covered = Counter()
    for c in chunks:
        covered.update(squeeze(c["content"]))
    missing_chars = sum((src - covered).values())
    dup_chars = sum((covered - src).values())

    print(f"切出块数：{len(chunks)}（正文 {len(texts)} / 表格 {len(tables)}）")
    print(f"块长度：min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")
    print(f"超过 max 的块：{len(over)}  低于 min 的块：{len(under)}")
    print(f"表格块：{len(tables)}  缺表头：{len(no_header)}  带图片的块：{len(with_img)}")
    print(f"字符覆盖：丢失 {missing_chars} 字，重复 {dup_chars} 字")

    titled = [c for c in texts if c.get("section")]
    print(f"正文块带章节路径：{len(titled)}/{len(texts)}")

    json.dump(chunks, open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({
        "source": f"{os.path.basename(MD)}（MarkdownParser 解析）",
        "chunks_total": len(chunks),
        "text_count": len(texts),
        "table_count": len(tables),
        "min_len": min(sizes) if sizes else 0,
        "max_len": max(sizes) if sizes else 0,
        "avg_len": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "over_limit": len(over),
        "under_min": len(under),
        "table_no_header": len(no_header),
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
