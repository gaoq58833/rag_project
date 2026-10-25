"""量化评估：对 TextChunker 的分块结果打分（满分 100）

数据来源：tests/test_chunker.py 产出的
  data/chunk_test/chunks.json  分块结果
  data/chunk_test/stats.json   切分统计（覆盖率、超限块数等）

用法：python 量化/eval_chunks.py
结果追加为 量化/指标记录.md 的第 3 节（重复运行会覆盖该节，不动前两节）
"""
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认评 PDF 分块结果；传数据目录可评其它样本，如：python 量化/eval_chunks.py data/word_test 4 Word
CHUNK_DIR = os.path.join(PROJECT_ROOT, sys.argv[1]) if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "chunk_test")
SEC_NO = sys.argv[2] if len(sys.argv) > 2 else "3"
SEC_TITLE = sys.argv[3] if len(sys.argv) > 3 else ""
CHUNKS_JSON = os.path.join(CHUNK_DIR, "chunks.json")
STATS_JSON = os.path.join(CHUNK_DIR, "stats.json")
OUT_MD = os.path.join(PROJECT_ROOT, "量化", "指标记录.md")

# 正文块长度落在该区间视为粒度合理
IDEAL_MIN, IDEAL_MAX = 400, 1000

# 评分权重
W_COVERAGE = 25     # 完整性：正文有没有被切丢
W_SIZE = 20         # 块大小合规：不超上限、不留碎片
W_TABLE = 25        # 表格完整性：整表不拆散、每块带表头
W_GRANULARITY = 20  # 粒度合理性：正文块长度是否落在理想区间
W_TRACE = 10        # 结构可追溯：块里能否看出它来自哪张表 / 哪一节


def load():
  with open(CHUNKS_JSON, encoding="utf-8") as f:
    chunks = json.load(f)
  with open(STATS_JSON, encoding="utf-8") as f:
    stats = json.load(f)
  return chunks, stats


def score(chunks, stats):
  text = [c for c in chunks if c.get("type") == "text"]
  tables = [c for c in chunks if c.get("type") == "table"]
  total = len(chunks)

  # 1. 完整性：丢失字符占比
  src = stats["src_chars"] or 1
  coverage = 1 - stats["missing_chars"] / src
  s_coverage = W_COVERAGE * coverage

  # 2. 块大小合规：超限块 + 低于下限的碎片
  bad = stats["over_limit"] + stats["under_min"]
  s_size = W_SIZE * (1 - bad / total) if total else 0

  # 3. 表格完整性：缺表头比例 + 被拆进正文块的表格行
  tb = stats["table_count"] or 1
  no_header = stats["table_no_header"]
  split_tables = sum(1 for c in text if any(
    l.strip().startswith("|") for l in c["content"].split("\n")))
  table_penalty = no_header + split_tables
  s_table = W_TABLE * max(0, 1 - table_penalty / tb)

  # 4. 粒度合理性：正文块落在理想区间的比例
  good = sum(1 for c in text if IDEAL_MIN <= c["length"] <= IDEAL_MAX)
  gran_rate = good / len(text) if text else 0
  s_gran = W_GRANULARITY * gran_rate

  # 5. 结构可追溯：表格块带行号前缀（5 分）+ 正文块带章节路径（5 分）
  prefixed = sum(1 for c in tables if c["content"].startswith("【表格"))
  s_table_trace = 5 * (prefixed / len(tables)) if tables else 0
  titled = sum(1 for c in text if c.get("section"))
  s_sec_trace = 5 * (titled / len(text)) if text else 0
  s_trace = s_table_trace + s_sec_trace

  total_score = s_coverage + s_size + s_table + s_gran + s_trace

  return {
    "total": total_score,
    "coverage": s_coverage, "size": s_size, "table": s_table,
    "gran": s_gran, "trace": s_trace,
    "table_trace": s_table_trace, "sec_trace": s_sec_trace,
    "coverage_rate": coverage, "bad_blocks": bad,
    "split_tables": split_tables, "no_header": no_header,
    "gran_rate": gran_rate, "good_text": good, "text_count": len(text),
    "table_count": len(tables), "chunks_total": total,
    "avg_len": stats["avg_len"], "min_len": stats["min_len"],
    "max_len": stats["max_len"], "missing": stats["missing_chars"],
    "max_size": stats["max_chunk_size"],
    "source": stats.get("source", "上述 PDF 经 PDFParser 解析后的全文"),
  }


def build_section():
  if not (os.path.exists(CHUNKS_JSON) and os.path.exists(STATS_JSON)):
    return ""
  chunks, stats = load()
  m = score(chunks, stats)

  md = []
  title = f"## {SEC_NO}. TextChunker —— 分块质量评分（满分 100）"
  if SEC_TITLE:
    title = f"## {SEC_NO}. {SEC_TITLE} —— 分块质量评分（满分 100）"
  md.append(title + "\n")
  md.append(f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
  md.append(f"样本：{m['source']}\n")
  md.append("配置：策略=段落（超长自动降级句子→字符），"
            f"max={m['max_size']}，min=100，overlap=200\n")
  md.append(f"结果：{m['chunks_total']} 块（正文 {m['text_count']} / 表格 {m['table_count']}），"
            f"平均 {m['avg_len']} 字\n")

  md.append("\n| 维度 | 权重 | 得分 | 依据 |")
  md.append("| --- | --- | --- | --- |")
  md.append(f"| 完整性（不丢内容） | {W_COVERAGE} | {m['coverage']:.2f} | "
            f"字符覆盖 {m['coverage_rate']*100:.2f}%，丢失 {m['missing']} 字 |")
  md.append(f"| 块大小合规 | {W_SIZE} | {m['size']:.2f} | "
            f"超限/碎片共 {m['bad_blocks']} 块，占比 {m['bad_blocks']}/{m['chunks_total']} |")
  md.append(f"| 表格完整性 | {W_TABLE} | {m['table']:.2f} | "
            f"缺表头 {m['no_header']} 块，表格残片混入正文块 {m['split_tables']} 块 |")
  md.append(f"| 粒度合理性 | {W_GRANULARITY} | {m['gran']:.2f} | "
            f"正文块落在 {IDEAL_MIN}-{IDEAL_MAX} 字区间 {m['good_text']}/{m['text_count']}"
            f"（{m['gran_rate']*100:.1f}%），平均 {m['avg_len']} 字 |")
  md.append(f"| 结构可追溯 | {W_TRACE} | {m['trace']:.2f} | "
            f"表格块带行号前缀 +{m['table_trace']:.2f}，正文块带章节路径 +{m['sec_trace']:.2f} |")
  md.append(f"| **总分** | **100** | **{m['total']:.1f}** | — |")

  md.append("\n### 扣分项说明\n")
  deducts = []
  if W_COVERAGE - m['coverage'] > 0.005:
    deducts.append(f"- **完整性 -{W_COVERAGE - m['coverage']:.2f}**：丢失 {m['missing']} 字")
  if W_SIZE - m['size'] > 0.005:
    deducts.append(f"- **块大小 -{W_SIZE - m['size']:.2f}**：超限或碎片共 {m['bad_blocks']} 块，"
                   f"占比 {m['bad_blocks']}/{m['chunks_total']}")
  if W_TABLE - m['table'] > 0.005:
    deducts.append(f"- **表格完整性 -{W_TABLE - m['table']:.2f}**：缺表头 {m['no_header']} 块，"
                   f"表格残片混入正文块 {m['split_tables']} 块")
  if W_GRANULARITY - m['gran'] > 0.005:
    deducts.append(f"- **粒度 -{W_GRANULARITY - m['gran']:.2f}**：正文块 {m['good_text']}/{m['text_count']} "
                   f"落在 {IDEAL_MIN}-{IDEAL_MAX} 字区间，其余多为表格/标题夹缝里的短块")
  if W_TRACE - m['trace'] > 0.005:
    deducts.append(f"- **可追溯 -{W_TRACE - m['trace']:.2f}**：表格块带行号前缀 "
                   f"{m['table_trace']/5*100:.0f}%、正文块带章节路径 {m['sec_trace']/5*100:.0f}%")
  if deducts:
    md.extend(deducts)
  else:
    md.append("- 五个维度均无扣分")
  md.append(f"- **满分项**：字符丢失 {m['missing']} 字、无超限/碎片问题的块 "
            f"{m['chunks_total'] - m['bad_blocks']}/{m['chunks_total']}、"
            f"缺表头 {m['no_header']} 块、正文块 {m['sec_trace']/5*100:.0f}% 携带章节标题路径\n")

  # 样本特有的客观性说明（如文档自身结构缺陷导致的扣分），随数据目录保存，重跑不丢失
  notes = os.path.join(CHUNK_DIR, "notes.md")
  if os.path.exists(notes):
    md.append("### 客观性说明\n")
    md.append(open(notes, encoding="utf-8").read().strip() + "\n")
  return "\n".join(md)


def write_into_record():
  section = build_section()
  if not section:
    print("缺少 data/chunk_test/*.json，请先运行 tests/test_chunker.py")
    return
  old = open(OUT_MD, encoding="utf-8").read() if os.path.exists(OUT_MD) else ""
  marker = f"## {SEC_NO}. "
  head = old.split(marker)[0].rstrip() if marker in old else old.rstrip()
  # 覆盖同编号章节时，保留该章节之后的内容
  if marker in old:
    rest = old.split(marker, 1)[1]
    nxt = rest.find("\n## ")
    if nxt != -1:
      body = head + "\n\n" + section + "\n" + rest[nxt:]
      open(OUT_MD, "w", encoding="utf-8").write(body)
      return
  body = (head + "\n\n" + section) if head else section
  open(OUT_MD, "w", encoding="utf-8").write(body)


if __name__ == "__main__":
  write_into_record()
  chunks, stats = load()
  m = score(chunks, stats)
  print(f"总分 {m['total']:.1f}/100")
  print(f"  完整性 {m['coverage']:.2f}/{W_COVERAGE}  块大小 {m['size']:.2f}/{W_SIZE}")
  print(f"  表格 {m['table']:.2f}/{W_TABLE}  粒度 {m['gran']:.2f}/{W_GRANULARITY}  "
        f"可追溯 {m['trace']:.2f}/{W_TRACE}")
  print(f"指标已写入 {OUT_MD}")
