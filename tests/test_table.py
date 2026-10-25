"""测试 TableProcessor 表格解析器（数据来自 tests/a.md 的真实表格）"""
import sys
import os
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.table_parser import TableProcessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(PROJECT_ROOT, "tests", "a.md")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "table_test")


def extract_tables(markdown: str):
  """从 markdown 中提取表格块：表头行 + 分隔行 + 后续以 | 开头的行

  Returns:
    [(起始行号, markdown 文本), ...]
  """
  is_sep = lambda s: bool(re.fullmatch(r'\|[\s:|-]+\|', s))
  lines = markdown.split('\n')
  tables, i = [], 0
  while i < len(lines) - 1:
    head, sep = lines[i].strip(), lines[i + 1].strip()
    if head.startswith('|') and is_sep(sep):
      block, j = [head, sep], i + 2
      while j < len(lines) and lines[j].strip().startswith('|'):
        # 某行后面紧跟分隔行，说明它是下一张表（跨页拆分）的表头
        if j + 1 < len(lines) and is_sep(lines[j + 1].strip()):
          break
        block.append(lines[j].strip())
        j += 1
      tables.append((i + 1, '\n'.join(block)))
      i = j
    else:
      i += 1
  return tables


def main():
  print("=== 测试 TableProcessor（真实数据源：tests/a.md）===")
  os.makedirs(OUT_DIR, exist_ok=True)

  with open(MD_PATH, encoding="utf-8") as f:
    markdown = f.read()

  tables = extract_tables(markdown)
  assert tables, "未能从 a.md 中提取到表格"
  print(f"从 a.md 提取到 {len(tables)} 个表格块\n")

  processor = TableProcessor()

  # 1. 逐个解析：转 DataFrame 并导出 CSV，同时做完整性检查
  frames = []
  for n, (line_no, block) in enumerate(tables, 1):
    df = processor.markdown_to_dataframe(block)
    frames.append(df)

    cols = processor._table_cols(block)
    assert list(df.columns) and len(df.columns) == cols, f"表{n} 列数与表头不符"
    # 表头行与分隔行都不应出现在数据里
    for row in df.astype(str).values.tolist():
      assert not all(re.fullmatch(r':?-+:?', c.replace(' ', '')) for c in row if c), \
        f"表{n} 含分隔线脏行"
    assert not any(row == list(df.columns) for row in df.astype(str).values.tolist()), \
      f"表{n} 含重复表头脏行"

    df.to_csv(os.path.join(OUT_DIR, f"table_{n:02d}.csv"), index=False, encoding="utf-8-sig")
    print(f"表{n}（第 {line_no} 行）: {cols} 列 × {len(df)} 行 -> table_{n:02d}.csv")

  # 2. 表格转文本描述，汇总输出
  with open(os.path.join(OUT_DIR, "tables_text.md"), "w", encoding="utf-8") as f:
    for n, (line_no, block) in enumerate(tables, 1):
      f.write(f"## 表格 {n}（第 {line_no} 行）\n\n```\n")
      f.write(processor.table_to_text(block))
      f.write("\n```\n\n")
  print(f"\n[OK] table_to_text 结果 -> tables_text.md")

  # 3. 空表边界
  assert processor.markdown_to_dataframe("").empty
  assert processor.markdown_to_dataframe("| 只有一个表头 |").empty
  assert processor.table_to_text("") == ""
  print("[OK] 空表/单行表返回空结果")

  # 4. 跨页合并：用 PDFParser 的真实键名（table_number/page/markdown）
  pdf_tables = [
    {"table_number": 1, "page": 5, "markdown": "| 项目 | 取值 |\n|---|---|\n| 训练样本 | 25000 条 |"},
    {"table_number": 2, "page": 6, "markdown": "| 项目 | 取值 |\n|---|---|\n| 测试样本 | 4000 条 |"},
    {"table_number": 3, "page": 7, "markdown": "| 参数 | 作用 | 后果 |\n|---|---|---|\n| lr | 学习率 | 过大不收敛 |"},
   ]
  merged = processor.merge_tables(pdf_tables)

  assert len(merged) == 2, f"跨页同列数的前两张应合并，实际得到 {len(merged)} 张"
  assert merged[0]["rows"] == 2, f"合并后应有 2 行数据，实际 {merged[0]['rows']}"
  assert merged[0]["markdown"].count("| 项目 | 取值 |") == 1, "合并后不应保留第二套表头"
  assert "训练样本" in merged[0]["markdown"] and "测试样本" in merged[0]["markdown"]
  # 入参不应被就地修改
  assert "rows" not in pdf_tables[0], "merge_tables 不应修改入参"

  # 页码缺失时仅按列数判断，不报错
  no_page = processor.merge_tables([{"markdown": "| a | b |\n|---|---|\n| 1 | 2 |"}])
  assert len(no_page) == 1

  with open(os.path.join(OUT_DIR, "merged.md"), "w", encoding="utf-8") as f:
    for i, t in enumerate(merged, 1):
      f.write(f"## 合并后表格 {i}"
              f"（第 {t.get('page')} 页，{t['cols']} 列 {t['rows']} 行）\n\n{t['markdown']}\n\n")
  with open(os.path.join(OUT_DIR, "merged.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
  print("[OK] merge_tables 跨页合并 -> merged.md / merged.json")

  total_rows = sum(len(df) for df in frames)
  print(f"\n[OK] 全部通过：{len(tables)} 个表格块，共 {total_rows} 行数据，结果在 {OUT_DIR}")


if __name__ == "__main__":
  main()
