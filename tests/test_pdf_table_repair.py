# 测试表格尾行补齐逻辑（不依赖 docling，只用 PDF 文本层）
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.pdf_table_repair import repair_missing_tail_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "test_file", "基于MindSpore的RNN情感分类实验.pdf")


def _bbox(l, b, r, t):
  return SimpleNamespace(l=l, b=b, r=r, t=t, coord_origin="BOTTOMLEFT")


def _table(added, cells_rows=7):
  """按真实 PDF 量出的行列边界造一张假表（第 7 行为「环节」表已解析出的最后一行）"""
  row_ys = [(655.0, 668.0), (631.5, 643.5), (605.5, 618.5),
            (580.5, 594.0), (556.0, 569.0), (532.5, 544.0), (506.0, 519.5)]
  col_xs = [(43.0, 140.0), (140.0, 215.0), (215.0, 430.0)]
  cells = []
  for r, (b, t) in enumerate(row_ys[:cells_rows]):
    for c, (l, rr) in enumerate(col_xs):
      cells.append(SimpleNamespace(
        start_row_offset_idx=r, end_row_offset_idx=r + 1,
        start_col_offset_idx=c, end_col_offset_idx=c + 1,
        bbox=_bbox(l, b, rr, t), text=f"r{r}c{c}",
      ))
  return SimpleNamespace(
    prov=[SimpleNamespace(page_no=7, bbox=_bbox(43.0, 505.0, 430.0, 668.0))],
    data=SimpleNamespace(num_rows=cells_rows, num_cols=3, table_cells=cells,
                         add_row=added.append),
  )


def _doc(table, texts):
  return SimpleNamespace(
    tables=[table], texts=texts,
    pages={7: SimpleNamespace(size=SimpleNamespace(height=841.92))},
    delete_items=lambda *, node_items: None,
  )


def test_repair_missing_tail_row():
  """docling 把「训练与评估」整行漏成正文，应补回表格并删除重复正文"""
  added = []
  orphans = [
    SimpleNamespace(label="text", text="训练与评估",
                    prov=[SimpleNamespace(page_no=7, bbox=_bbox(50.7, 484, 102.3, 493.8))]),
    SimpleNamespace(label="text", text="任务八、九 准确率、模型文件",
                    prov=[SimpleNamespace(page_no=7, bbox=_bbox(151.0, 484, 301.8, 493.8))]),
  ]
  fixed = repair_missing_tail_rows(_doc(_table(added), orphans), PDF)
  assert fixed == 1
  assert added[0] == ["训练与评估", "任务八、九", "准确率、模型文件"]


def test_skip_when_text_does_not_match():
  """窄条里的正文与表格行对不上时，不应误并"""
  added = []
  orphans = [SimpleNamespace(
    label="text", text="这段文字对不上",
    prov=[SimpleNamespace(page_no=7, bbox=_bbox(50.7, 484, 102.3, 493.8))])]
  assert repair_missing_tail_rows(_doc(_table(added), orphans), PDF) == 0
  assert not added


if __name__ == "__main__":
  test_repair_missing_tail_row()
  test_skip_when_text_does_not_match()
  print("OK")
