"""量化评估：比对 PDFParser 解析出的表格与原始 markdown 中的表格

用法：uv run python 量化/eval_tables.py
结果写入 量化/指标记录.md（表格形式）
"""
import sys
import os
import re
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.pdf_parser import PDFParser
from src.parsers.table_parser import TableProcessor
from docling.datamodel.base_models import InputFormat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, "test_file", "基于MindSpore的RNN情感分类实验.pdf")
GT_PATH = os.path.join(PROJECT_ROOT, "test_file", "基于MindSpore的RNN情感分类实验.md")
OUT_DIR = os.path.join(PROJECT_ROOT, "量化")
OUT_MD = os.path.join(OUT_DIR, "指标记录.md")

IS_SEP = re.compile(r'\|[\s:|-]+\|')


def parse_row(line):
  """'| a | b |' -> ['a', 'b']"""
  return [c.strip() for c in line.strip().strip('|').split('|')]


def extract_tables(markdown):
  """提取表格，返回 [[[cell, ...], ...], ...]（表 -> 行 -> 单元格）"""
  lines = markdown.split('\n')
  tables, i = [], 0
  while i < len(lines) - 1:
    head, sep = lines[i].strip(), lines[i + 1].strip()
    if head.startswith('|') and IS_SEP.fullmatch(sep):
      rows = []
      j = i
      while j < len(lines) and lines[j].strip().startswith('|'):
        # 某行后紧跟分隔行 → 下一张表的表头，停止当前表
        if j > i and j + 1 < len(lines) and IS_SEP.fullmatch(lines[j + 1].strip()):
          break
        if not IS_SEP.fullmatch(lines[j].strip()):
          rows.append(parse_row(lines[j]))
        j += 1
      tables.append(rows)
      i = j
    else:
      i += 1
  return tables


def norm(cell):
  """归一化：去掉 markdown 强调/代码标记与所有空白"""
  return re.sub(r'\s+', '', re.sub(r'[*`]', '', cell))


def compare(gt_tables, pr_tables):
  """按表头对齐逐格比对（不能按序号对齐：OCR 失败丢表会让后面全部错位）"""
  def key(rows):
    return "|".join(norm(c) for c in rows[0]) if rows and rows[0] else ""

  pool = list(pr_tables)
  details = []
  for idx, gt in enumerate(gt_tables, 1):
    k = key(gt)
    match = next((t for t in pool if key(t) == k), None)
    if match is not None:
      pool.remove(match)
    pr = match or []
    gt_cells = len(gt) * len(gt[0]) if gt else 0
    structural = (bool(pr) and len(gt) == len(pr) and len(gt[0]) == len(pr[0]))

    exact = normed = 0
    diffs = []
    for r in range(min(len(gt), len(pr))):
      for c in range(min(len(gt[r]), len(pr[r]))):
        if gt[r][c] == pr[r][c]:
          exact += 1
        if norm(gt[r][c]) == norm(pr[r][c]):
          normed += 1
        elif len(diffs) < 3:
          diffs.append((r, c, gt[r][c], pr[r][c]))

    details.append({
      "index": idx,
      "header": gt[0][0] if gt and gt[0] else "-",
      "gt_shape": f"{len(gt[0]) if gt else 0}×{len(gt)}",
      "pr_shape": f"{len(pr[0]) if pr else 0}×{len(pr)}" if pr else "缺失",
      "structural": structural,
      "gt_cells": gt_cells,
      "exact": exact,
      "normed": normed,
      "diffs": diffs,
      "missing_rows": max(len(gt) - len(pr), 0) if pr else len(gt),
     })
  return details, pool


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  processor = TableProcessor()
  use_ocr = "--no-ocr" not in sys.argv

  with open(GT_PATH, encoding="utf-8") as f:
    gt_raw = f.read()
  gt_tables = extract_tables(gt_raw)

  parser = PDFParser(do_ocr=use_ocr)
  applied = parser.converter.format_to_options[InputFormat.PDF].pipeline_options
  print(f"配置：do_ocr={applied.do_ocr} ocr_scale={applied.ocr_options.scale} "
        f"images_scale={applied.images_scale}")

  result = parser.parse(PDF_PATH)
  pr_tables = [extract_tables(t["markdown"])[0] for t in processor.merge_tables(result.tables)
               if extract_tables(t["markdown"])]

  details, leftover = compare(gt_tables, pr_tables)

  gt_total = sum(d["gt_cells"] for d in details)
  exact_total = sum(d["exact"] for d in details)
  normed_total = sum(d["normed"] for d in details)
  structural_ok = sum(1 for d in details if d["structural"])
  missing = sum(1 for d in details if d["pr_shape"] == "缺失")

  exact_rate = exact_total / gt_total * 100 if gt_total else 0
  normed_rate = normed_total / gt_total * 100 if gt_total else 0
  struct_rate = structural_ok / len(gt_tables) * 100 if gt_tables else 0
  count_rate = (len(gt_tables) - missing) / len(gt_tables) * 100 if gt_tables else 0

  print(f"原始表格数={len(gt_tables)}  解析表格数={len(pr_tables)}  缺失={missing}  多余={len(leftover)}")
  print(f"结构一致 {structural_ok}/{len(gt_tables)}  单元格 {exact_total}/{gt_total}  归一化 {normed_total}/{gt_total}")
  for d in details:
    e = d["exact"] / d["gt_cells"] * 100 if d["gt_cells"] else 0
    n = d["normed"] / d["gt_cells"] * 100 if d["gt_cells"] else 0
    print(f"  表{d['index']} {d['header'][:12]:<14} 结构{'一致' if d['structural'] else '不一致'} "
          f"GT={d['gt_shape']} PR={d['pr_shape']} 精确={e:.1f}% 归一化={n:.1f}%")

  d = eval_document(gt_raw, result.raw_text)
  print(f"\n整篇文档：精确={d['precision']*100:.2f}% 召回={d['recall']*100:.2f}% "
        f"F1={d['f1']*100:.2f}%  字符数 {d['pr_chars']} vs {d['gt_chars']} "
        f"({d['len_delta']:+.1f}%)  行一致={d['line_rate']*100:.2f}% "
        f"图片={d['pr_images']} vs {d['gt_images']}")

  write_report(details, {
    "count_rate": count_rate, "struct_rate": struct_rate,
    "exact_rate": exact_rate, "normed_rate": normed_rate,
    "exact_total": exact_total, "normed_total": normed_total, "gt_total": gt_total,
    "gt_count": len(gt_tables), "pr_count": len(pr_tables),
    "missing": missing, "extra": len(leftover), "use_ocr": use_ocr,
    "ocr_scale": applied.ocr_options.scale, "images_scale": applied.images_scale,
    "doc": eval_document(gt_raw, result.raw_text),
   })

  json.dump({"details": details, "extra_tables": leftover},
            open(os.path.join(OUT_DIR, "table_details.json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=2)
  with open(os.path.join(OUT_DIR, "解析输出.md"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(t["markdown"] for t in processor.merge_tables(result.tables)))
  print(f"\n指标已写入 {OUT_MD}")


def build_report(details, m):
  now = datetime.now().strftime("%Y-%m-%d %H:%M")
  lines = []
  lines.append("# 量化指标记录\n")
  lines.append("> 所有量化指标以表格形式记录于此。每次评估追加一行，不覆盖历史。\n")
  lines.append(f"评估时间：{now}\n")
  lines.append("## 1. PDF 解析器 —— 表格抽取准确率\n")
  lines.append("样本：`基于MindSpore的RNN情感分类实验.pdf`")
  lines.append("基线：`test_file/基于MindSpore的RNN情感分类实验.md`（同名原始 markdown）\n")
  lines.append(f"运行配置：`do_ocr={m['use_ocr']}`、`ocr_scale={m['ocr_scale']}`、"
               f"`images_scale={m['images_scale']}`\n")
  lines.append("| 指标 | 数值 | 口径说明 |")
  lines.append("| --- | --- | --- |")
  lines.append(f"| 表格数量准确率 | {m['count_rate']:.1f}% | 解析 {m['pr_count']} 张，原始 {m['gt_count']} 张，缺失 {m['missing']} 张 |")
  lines.append(f"| 表格结构一致率 | {m['struct_rate']:.1f}% | 行数与列数同时一致的表数占比 |")
  lines.append(f"| 单元格精确匹配率 | {m['exact_rate']:.2f}% | 逐格比对，字符完全相同才算对（{m['exact_total']}/{m['gt_total']}） |")
  lines.append(f"| **整体表格准确率** | **{m['normed_rate']:.2f}%** | 逐格比对，忽略空白与 markdown 强调符后比对（{m['normed_total']}/{m['gt_total']}） |\n")

  lines.append("### 分表明细\n")
  lines.append("| # | 表头首列 | 原始 列×行 | 解析 列×行 | 结构 | 精确匹配 | 归一化匹配 |")
  lines.append("| --- | --- | --- | --- | --- | --- | --- |")
  for d in details:
    e = d["exact"] / d["gt_cells"] * 100 if d["gt_cells"] else 0
    n = d["normed"] / d["gt_cells"] * 100 if d["gt_cells"] else 0
    lines.append(f"| {d['index']} | {d['header']} | {d['gt_shape']} | {d['pr_shape']} | "
                 f"{'一致' if d['structural'] else '不一致'} | {e:.1f}% | {n:.1f}% |")

  problems = []
  for d in details:
    if d["pr_shape"] == "缺失":
      problems.append(f"- 表{d['index']}「{d['header']}」整张缺失")
    elif d["missing_rows"]:
      problems.append(f"- 表{d['index']}「{d['header']}」少 {d['missing_rows']} 行"
                      f"（原始 {d['gt_shape']}，解析 {d['pr_shape']}）")
    elif d["diffs"] and d["normed"] < d["gt_cells"]:
      cells = "；".join(f"R{r}C{c}: 原文「{g[:28]}」→ 解析「{p[:28]}」" for r, c, g, p in d["diffs"])
      problems.append(f"- 表{d['index']}「{d['header']}」单元格不一致：{cells}")

  if problems:
    lines.append("\n### 差异说明\n")
    lines.extend(problems)

  lines.append(build_doc_section(m))
  return "\n".join(lines) + "\n"


def norm_doc(text):
  """全文归一化：去掉图片 base64、markdown 标记、所有空白与引号差异"""
  text = re.sub(r'!\[.*?\]\(data:image/[^)]*\)', ' ', text, flags=re.S)
  text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
  text = re.sub(r'[`*~#>|\-]', '', text)
  text = re.sub(r'^\d+[.、]\s*', '', text)  # 去掉行首序号，如 "4.一、..." 与 "一、..."
  text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
  text = re.sub(r'[\s\u3000]+', '', text)
  return text


def char_f1(a, b):
  """按字符集合算准确率/召回率/F1（中文无词边界，不做分词）"""
  from collections import Counter
  ca, cb = Counter(a), Counter(b)
  hit = sum((ca & cb).values())
  p = hit / len(b) if b else 0
  r = hit / len(a) if a else 0
  f1 = 2 * p * r / (p + r) if p + r else 0
  return p, r, f1


def build_doc_section(m):
  d = m["doc"]
  lines = ["\n## 2. PDF 解析器 —— 整篇文档文本准确率\n"]
  lines.append("样本：同上的 PDF 全文（含正文、标题、列表、表格、图片）")
  lines.append("基线：`test_file/基于MindSpore的RNN情感分类实验.md`")
  lines.append("口径：两侧去掉图片 base64、markdown 标记与所有空白后，按**字符多重集**比对\n")
  lines.append("| 指标 | 数值 | 口径说明 |")
  lines.append("| --- | --- | --- |")
  lines.append(f"| 字符准确率 | {d['precision']*100:.2f}% | 解析结果中有多少字符是对的（{d['hit']}/{d['pr_chars']}） |")
  lines.append(f"| 字符召回率 | {d['recall']*100:.2f}% | 原文有多少字符被解析出来（{d['hit']}/{d['gt_chars']}） |")
  lines.append(f"| **整篇文档字符 F1** | **{d['f1']*100:.2f}%** | 准确率与召回率的调和平均，作为整篇文档准确率 |")
  lines.append(f"| 字符数偏差 | {d['len_delta']:+.1f}% | 解析 {d['pr_chars']} 字 vs 原文 {d['gt_chars']} 字 |")
  lines.append(f"| 行级完全一致率 | {d['line_rate']*100:.2f}% | 逐行比对，忽略空白后完全相同的行占比（{d['line_hit']}/{d['line_total']}） |")
  lines.append(f"| 图片数量 | {d['pr_images']} vs {d['gt_images']} | 解析出的图片数 vs 原文图片数 |\n")
  if d["missing_lines"]:
    lines.append("### 代表性缺失行（原文有、解析无）\n")
    for s in d["missing_lines"]:
      lines.append(f"- `{s}`")
    lines.append("")
  if d["extra_lines"]:
    lines.append("### 代表性多出行（解析有、原文无）\n")
    for s in d["extra_lines"]:
      lines.append(f"- `{s}`")
    lines.append("")
  try:
    from eval_chunks import build_section
    chunk_md = build_section()
    if chunk_md:
      lines.append(chunk_md)
  except Exception:
    pass
  return "\n".join(lines)


def eval_document(gt_raw, pr_raw):
  """整篇文档级别评估"""
  gt_lines = [norm_doc(l) for l in gt_raw.split("\n")]
  pr_lines = [norm_doc(l) for l in pr_raw.split("\n")]
  gt_lines = [l for l in gt_lines if l]
  pr_lines = [l for l in pr_lines if l]

  gt_text, pr_text = "".join(gt_lines), "".join(pr_lines)
  p, r, f1 = char_f1(gt_text, pr_text)

  from collections import Counter
  cg, cp = Counter(gt_lines), Counter(pr_lines)
  line_hit = sum((cg & cp).values())
  missing = [l for l in list(dict.fromkeys(gt_lines)) if l not in cp][:5]
  extra = [l for l in list(dict.fromkeys(pr_lines)) if l not in cg][:5]

  return {
    "precision": p, "recall": r, "f1": f1,
    "hit": sum((Counter(gt_text) & Counter(pr_text)).values()),
    "gt_chars": len(gt_text), "pr_chars": len(pr_text),
    "len_delta": (len(pr_text) - len(gt_text)) / len(gt_text) * 100 if gt_text else 0,
    "line_hit": line_hit, "line_total": len(gt_lines),
    "line_rate": line_hit / len(gt_lines) if gt_lines else 0,
    "gt_images": gt_raw.count("!["),
    "pr_images": pr_raw.count("data:image"),
    "missing_lines": [s[:70] for s in missing],
    "extra_lines": [s[:70] for s in extra],
   }


def write_report(details, m):
  open(OUT_MD, "w", encoding="utf-8").write(build_report(details, m))


if __name__ == "__main__":
  main()
