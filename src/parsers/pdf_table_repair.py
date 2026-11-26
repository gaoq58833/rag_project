"""补齐 docling 漏掉的表格尾行。

docling 的版面模型偶尔会把表格最后一行排除在表格 bbox 之外，TableFormer
因此看不到这一行，docling 会把它当成正文 TextItem 输出（见 ERRORS.md #10）。

这里用 PDF 文本层把该行按表格已有的列坐标取回来：
1. 由表格单元 bbox 推出列 x 区间、最后一行下边界与行距；
2. 在最后一行正下方取一条「一行高」的窄条，按列分别取文本；
3. 要求每列都有内容、都是单行，且拼起来与落在窄条里的正文 TextItem 完全一致；
4. 校验通过才追加为表格行，并删除重复的正文 TextItem。

只在「文档里确实还有这段文字」时才动手，宁可漏补也不误并。
"""
from statistics import median
from typing import Any, List, Optional, Sequence, Tuple

# 最多补几行（正常只会丢最后 1 行）
MAX_TAIL_ROWS = 2


def repair_missing_tail_rows(doc: Any, source: Any) -> int:
  """把表格下方漏掉的整行补进 TableItem，返回补了多少行。

  Args:
    doc: docling 的 DoclingDocument（只用到 tables/texts/pages/delete_items）
    source: PDF 路径或字节，交给 pypdfium2 打开

  Returns:
    补齐的行数；无法处理时返回 0
  """
  try:
    import pypdfium2 as pdfium
  except ImportError:
    return 0

  tables = list(getattr(doc, "tables", []) or [])
  if not tables:
    return 0
  try:
    pdf = pdfium.PdfDocument(source)
  except Exception:
    return 0

  texts = list(getattr(doc, "texts", []) or [])
  consumed: List[Any] = []
  fixed = 0
  try:
    for table in tables:
      try:
        added, orphans = _repair_one_table(pdf, doc, table, texts)
      except Exception:
        continue
      fixed += added
      consumed.extend(orphans)
    if consumed:
      # 同一段正文可能落在两张表的邻接区域，去重后再删
      doc.delete_items(node_items=list({id(item): item for item in consumed}.values()))
  finally:
    pdf.close()
  return fixed


def _repair_one_table(pdf: Any, doc: Any, table: Any, texts: Sequence[Any]) -> Tuple[int, List[Any]]:
  """处理单张表，返回 (补的行数, 待删除的正文 TextItem)"""
  prov = getattr(table, "prov", None) or []
  if not prov:
    return 0, []
  page_no = prov[0].page_no
  page_item = (getattr(doc, "pages", None) or {}).get(page_no)
  if page_item is None or getattr(page_item, "size", None) is None:
    return 0, []
  page_h = page_item.size.height

  data = getattr(table, "data", None)
  if data is None or data.num_rows < 2 or data.num_cols < 2:
    return 0, []
  cells = list(getattr(data, "table_cells", None) or [])

  table_box = _bottom_left(prov[0].bbox, page_h)
  col_ranges = _column_ranges(cells, data.num_cols, table_box, page_h)
  if col_ranges is None:
    return 0, []

  last_bottom, pitch = _last_row_geometry(cells, data.num_rows, table_box, page_h)
  if last_bottom is None or pitch is None:
    return 0, []

  # 表格最后一行正下方、约一行高的窄条
  strip_top = last_bottom - 1.5
  strip_bottom = last_bottom - 1.2 * pitch
  if strip_top - strip_bottom < 2:
    return 0, []

  table_l = min(col[0] for col in col_ranges)
  table_r = max(col[1] for col in col_ranges)

  # 窄条里必须真有被 docling 当正文输出的文字，否则不动
  orphans = _orphan_texts(texts, page_no, page_h, strip_bottom, strip_top,
                          table_l, table_r, pitch)
  if not orphans:
    return 0, []

  textpage = pdf[page_no - 1].get_textpage()
  col_texts = []
  for left, right in col_ranges:
    if right - left < 1:
      return 0, []
    text = _clean(textpage.get_text_bounded(
      left=left, bottom=strip_bottom, right=right, top=strip_top))
    if not text or "\n" in text or "\r" in text:
      return 0, []
    col_texts.append(text)

  joined = _norm("".join(col_texts))
  orphan_joined = _norm("".join(item.text for item in _sort_by_x(orphans, page_h)))
  if not joined or joined != orphan_joined:
    return 0, []

  # 已经解析出来的最后一行内容不能重复追加
  last_row = [c.text for c in cells
              if c.start_row_offset_idx == data.num_rows - 1 and c.text.strip()]
  if last_row and _norm("".join(last_row)) == joined:
    return 0, []

  data.add_row(col_texts)
  return 1, orphans


def _column_ranges(cells: List[Any], num_cols: int,
                   table_box: Tuple[float, float, float, float],
                   page_h: float) -> Optional[List[Tuple[float, float]]]:
  """由单元格 bbox 推每列的 (left, right)；推不出来时按表格宽度均分"""
  lefts: List[Optional[float]] = [None] * num_cols
  rights: List[Optional[float]] = [None] * num_cols
  for cell in cells:
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
      continue
    if cell.end_col_offset_idx - cell.start_col_offset_idx != 1:
      continue
    idx = cell.start_col_offset_idx
    if not 0 <= idx < num_cols:
      continue
    l, _, r, _ = _bottom_left(bbox, page_h)
    lefts[idx] = l if lefts[idx] is None else min(lefts[idx], l)
    rights[idx] = r if rights[idx] is None else max(rights[idx], r)

  if all(l is not None and r is not None for l, r in zip(lefts, rights)):
    ranges = [(float(l), float(r)) for l, r in zip(lefts, rights)]
    if all(r - l > 1 for l, r in ranges) and \
       all(ranges[i][0] < ranges[i + 1][0] for i in range(num_cols - 1)):
      # 相邻列偶尔有小幅重叠（单元格边框/文字外扩），按中点切开
      for i in range(num_cols - 1):
        if ranges[i][1] > ranges[i + 1][0]:
          mid = (ranges[i][1] + ranges[i + 1][0]) / 2
          ranges[i] = (ranges[i][0], mid)
          ranges[i + 1] = (mid, ranges[i + 1][1])
      return ranges

  t_l, _, t_r, _ = table_box
  width = (t_r - t_l) / num_cols
  if width <= 1:
    return None
  return [(t_l + i * width, t_l + (i + 1) * width) for i in range(num_cols)]


def _last_row_geometry(cells: List[Any], num_rows: int,
                       table_box: Tuple[float, float, float, float],
                       page_h: float) -> Tuple[Optional[float], Optional[float]]:
  """最后一行下边界（bottom-left y）与行距"""
  row_bottoms: dict = {}
  for cell in cells:
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
      continue
    _, b, _, _ = _bottom_left(bbox, page_h)
    for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
      row_bottoms.setdefault(row, []).append(b)

  bottoms = {row: median(vals) for row, vals in row_bottoms.items() if vals}
  if num_rows - 1 not in bottoms:
    return None, None
  last_bottom = bottoms[num_rows - 1]

  diffs = [bottoms[r] - bottoms[r + 1] for r in range(num_rows - 1)
           if r in bottoms and r + 1 in bottoms and bottoms[r] > bottoms[r + 1]]
  pitch = median(diffs) if diffs else None
  if pitch is None or pitch <= 3:
    t_l, t_b, t_r, t_t = table_box
    pitch = (t_t - t_b) / max(num_rows, 1)
  if pitch <= 3:
    return None, None
  return last_bottom, pitch


def _orphan_texts(texts: Sequence[Any], page_no: int, page_h: float,
                  strip_bottom: float, strip_top: float,
                  table_l: float, table_r: float, pitch: float) -> List[Any]:
  """落在表格正下方窄条、横向与表格有交集的正文 TextItem"""
  found = []
  for item in texts:
    label = str(getattr(item, "label", "")).lower()
    if "text" not in label and "paragraph" not in label:
      continue
    prov = getattr(item, "prov", None) or []
    if not prov or prov[0].page_no != page_no:
      continue
    l, b, r, t = _bottom_left(prov[0].bbox, page_h)
    if b < strip_bottom - 1 or t > strip_top + 1:
      continue
    if r <= table_l or l >= table_r:
      continue
    if t - b > 0.9 * pitch:  # 只认单行
      continue
    if not item.text.strip():
      continue
    found.append(item)
  return found


def _sort_by_x(items: Sequence[Any], page_h: float) -> List[Any]:
  def key(item):
    prov = getattr(item, "prov", None) or []
    return _bottom_left(prov[0].bbox, page_h)[0] if prov else 0.0
  return sorted(items, key=key)


def _bottom_left(bbox: Any, page_h: float) -> Tuple[float, float, float, float]:
  """统一成 (left, bottom, right, top) 的 bottom-left 坐标系"""
  if str(getattr(bbox, "coord_origin", "TOPLEFT")).upper().endswith("BOTTOMLEFT"):
    return bbox.l, bbox.b, bbox.r, bbox.t
  return bbox.l, page_h - bbox.b, bbox.r, page_h - bbox.t


def _clean(text: str) -> str:
  return " ".join((text or "").replace("\u00a0", " ").split())


def _norm(text: str) -> str:
  return "".join((text or "").split())
