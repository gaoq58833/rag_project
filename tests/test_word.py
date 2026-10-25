"""测试 Word 解析 + 分块：python-docx → markdown → TextChunker

用法：
  python tests/test_word.py                    默认测实验报告
  python tests/test_word.py <docx> <输出目录>   测指定文档

产出 markdown、chunks.json 与 stats.json，供量化评分使用。
"""
import sys
import os
import re
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from src.parsers.word_parser import WordParser, _iter_block_items
from docx.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(ROOT, "test_file", "实验报告_对称密码实验-高铨-2024150227.docx")
OUT = os.path.join(ROOT, "data", "word_test")
if len(sys.argv) > 1:
  DOCX = os.path.abspath(sys.argv[1])
if len(sys.argv) > 2:
  OUT = os.path.join(ROOT, sys.argv[2])


def squeeze(text):
  """去掉 markdown 标记和所有空白，用于内容覆盖率统计"""
  return re.sub(r'[\s]+', '', re.sub(r'[`*#>\-|!\[\]():/_.]', '', text))


def baseline_text(path):
  """基线：python-docx 直接读出的全部段落文字 + 表格单元格文字（100% 准确的原始内容）"""
  doc = Document(path)
  parts = []
  from src.parsers.word_parser import _iter_block_items
  for block in _iter_block_items(doc):
    if isinstance(block, Table):
      for row in block.rows:
        for cell in row.cells:
          parts.append(cell.text)
    else:
      parts.append(block.text)
  return "".join(parts)


def main():
  os.makedirs(OUT, exist_ok=True)

  result = WordParser(image_dir=os.path.join("data", "images")).parse(DOCX)
  text = result.raw_text
  # 保存解析出的 markdown，便于直接查看解析结果
  md_path = os.path.join(OUT, os.path.splitext(os.path.basename(DOCX))[0] + ".md")
  open(md_path, "w", encoding="utf-8").write(text)
  print(f"markdown 已保存：{md_path}")

  # ---- 解析准确率：markdown 里的内容字符 vs python-docx 直接读出的原始字符 ----
  src = Counter(squeeze(baseline_text(DOCX)))
  md = Counter(squeeze(text))
  missing = sum((src - md).values())
  extra = sum((md - src).values())
  total = sum(src.values())
  print(f"解析覆盖率：原文 {total} 字，markdown 覆盖 {total - missing} 字"
        f"（{(total - missing) / total * 100:.2f}%），多出 {extra} 字（表格竖线/图片路径等格式字符）")

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
    "source": f"{os.path.basename(DOCX)}（python-docx 解析）",
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
