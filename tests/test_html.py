"""测试 HTML 解析 + 分块：HTMLParser（markdownify）→ TextChunker

用法：
  python tests/test_html.py                        默认测 test_file/rag_test_doc.html
  python tests/test_html.py <html> <输出目录>       测指定文档

产出 chunks.json 与 stats.json，供量化评分使用。
"""
import sys
import os
import re
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.html_parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "test_file", "rag_test_doc.html")
OUT = os.path.join(ROOT, "data", "html_test")
if len(sys.argv) > 1:
    HTML = os.path.abspath(sys.argv[1])
if len(sys.argv) > 2:
    OUT = os.path.join(ROOT, sys.argv[2])


def squeeze(text):
    return re.sub(r'[\s]+', '', re.sub(r'[`*#>\-|!\[\]():/_.]', '', text))


def main():
    os.makedirs(OUT, exist_ok=True)

    result = HTMLParser().parse(HTML)
    text = result.raw_text
    md_path = os.path.join(OUT, os.path.splitext(os.path.basename(HTML))[0] + ".md")
    open(md_path, "w", encoding="utf-8").write(text)
    print(f"转换后的 markdown 已保存：{md_path}")

    # 解析准确率：用原始 HTML 提取的纯文本做基线（去掉所有标签）
    from bs4 import BeautifulSoup
    raw_html = open(HTML, encoding="utf-8").read()
    soup = BeautifulSoup(raw_html, "lxml")
    baseline = soup.get_text("\n", strip=True)
    src = Counter(squeeze(baseline))
    md = Counter(squeeze(text))
    missing = sum((src - md).values())
    extra = sum((md - src).values())
    total = sum(src.values())
    print(f"解析覆盖率：原文 {total} 字，转换后 {total - missing} 字"
          f"（{(total - missing) / total * 100:.2f}%），多出 {extra} 字（表格竖线/图片路径等格式字符）")

    from src.parsers.chunker import TextChunker, ChunkStrategy
    chunker = TextChunker(strategy=ChunkStrategy.PARAGRAPH,
                          max_chunk_size=1000, min_chunk_size=100, chunk_overlap=200)
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
        "source": f"{os.path.basename(HTML)}（HTMLParser/markdownify 解析）",
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
