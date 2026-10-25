"""测试按段落分块"""
import sys
import os
import re
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.pdf_parser import PDFParser
from src.parsers.chunker import TextChunker, ChunkStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "test_file", "基于MindSpore的RNN情感分类实验.pdf")
OUT = os.path.join(ROOT, "data", "chunk_test")


def strip_images(text):
  """去掉内嵌图片 base64，否则会被当成正文切碎"""
  return re.sub(r'!\[.*?\]\(data:image/[^)]*\)', '', text, flags=re.S)


def squeeze(text):
  """去掉 markdown 标记和所有空白，用于内容覆盖率统计"""
  return re.sub(r'[\s]+', '', re.sub(r'[`*#>\-]', '', text))


def main():
  os.makedirs(OUT, exist_ok=True)

  result = PDFParser(do_ocr=False).parse(PDF)
  text = strip_images(result.raw_text)

  chunker = TextChunker(
    strategy=ChunkStrategy.PARAGRAPH,
    max_chunk_size=1000,
    min_chunk_size=100,
    chunk_overlap=200,
   )
  chunks = chunker.split(text)

  # 统计
  sizes = [c["length"] for c in chunks]
  tables = [c for c in chunks if c["type"] == "table"]
  texts = [c for c in chunks if c["type"] == "text"]
  over = [s for s in sizes if s > chunker.max_chunk_size]
  under = [s for s in sizes if s < chunker.min_chunk_size]

  # 每个表格块是否都带了表头（子块重复表头策略的验收点）
  no_header = [c for c in tables if "\n|" not in c["content"]]

  # 覆盖率：原文段落有多少字进了块里（按字符多重集，重叠会导致 >100%）
  src = Counter(squeeze(text))
  covered = Counter()
  for c in chunks:
    covered.update(squeeze(c["content"]))
  missing_chars = sum((src - covered).values())
  dup_chars = sum((covered - src).values())

  print(f"切出块数：{len(chunks)}（正文 {len(texts)} / 表格 {len(tables)}）")
  print(f"块长度：min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0} "
        f"avg={sum(sizes)/len(sizes) if sizes else 0:.0f}")
  print(f"超过 max_chunk_size 的块：{len(over)}  低于 min_chunk_size 的块：{len(under)}")
  print(f"表格块总数：{len(tables)}  其中缺表头的：{len(no_header)}")
  print(f"字符覆盖：丢失 {missing_chars} 字（{missing_chars/len(squeeze(text))*100:.2f}%），"
        f"重复 {dup_chars} 字（重叠 {chunker.chunk_overlap}）")

  # 章节归属：正文块是否都带上了所属标题路径
  titled = [c for c in texts if c.get("section")]
  print(f"正文块带章节路径：{len(titled)}/{len(texts)}")
  for c in titled[:3]:
    print(f"  块{c['index']:>2} 【{c['section']}】 {c['content'][:40].replace(chr(10), ' ')}")

  print("\n表格块一览：")
  for c in tables:
    first = c["content"].split("\n")[0]
    print(f"  块{c['index']:>2} {c['length']:>4}字  {first}")

  with open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

  with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump({
      "source": "PDFParser 实时解析 PDF",
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
      "src_chars": len(squeeze(text)),
      "max_chunk_size": chunker.max_chunk_size,
      "min_chunk_size": chunker.min_chunk_size,
    }, f, ensure_ascii=False, indent=2)

  with open(os.path.join(OUT, "chunks_preview.md"), "w", encoding="utf-8") as f:
    for c in chunks[:10]:
      f.write(f"### 块 {c['index']}（{c['length']} 字，{c['type']}）\n\n")
      f.write(c["content"][:600] + "\n\n---\n\n")

  print("\n前 3 块预览：")
  for c in chunks[:3]:
    print(f"--- 块{c['index']} ({c['length']}字) ---")
    print(c["content"][:120].replace("\n", " ⏎ "))


  # 大表格验证：真实文档里的表都小于限额，这里造一张 60 行表走行窗口分支
  big = "\n".join(["| 序号 | 指标 | 说明 |", "| --- | --- | --- |"] +
                  [f"| {i} | 指标{i} | 这是第 {i} 行的说明文字 |" for i in range(1, 61)])
  subs = chunker._split_table(big)
  assert len(subs) > 1, "大表格没有被切开"
  assert all(s.startswith("【表格") for s in subs), "子块缺少行号前缀"
  assert all(s.split("\n")[1].startswith("| 序号") for s in subs), "子块没有重复表头"
  over_big = [len(s) for s in subs if len(s) > chunker.max_chunk_size]
  print(f"\n大表格（60 行）切成 {len(subs)} 块，超限 {len(over_big)} 块，"
        f"每块都重复表头：是")
  print(f"  首块开头：{subs[0].split(chr(10))[0]}")
  print(f"  末块开头：{subs[-1].split(chr(10))[0]}")


if __name__ == "__main__":
  main()
