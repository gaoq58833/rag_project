import re
from typing import List, Dict
import pandas as pd
class TableProcessor:
  """表格处理工具"""

  @staticmethod
  def _split_row(line: str) -> List[str]:
    """按竖线切分一行，去掉首尾的空单元格"""
    return [cell.strip() for cell in line.split('|')[1:-1]]

  @staticmethod
  def _is_separator(cells: List[str]) -> bool:
    """判断是否为表头下方那行分隔线，如 |---|:---:|"""
    if not cells:
      return False
    return all(re.fullmatch(r':?-+:?', cell.replace(' ', '')) for cell in cells)

  @staticmethod
  def _lines(table_markdown: str) -> List[str]:
    """切分为非空行"""
    return [line.strip() for line in (table_markdown or '').strip().split('\n') if line.strip()]

  def _table_cols(self, table_markdown: str) -> int:
    """表格列数"""
    lines = self._lines(table_markdown)
    return len(self._split_row(lines[0])) if lines else 0

  def _table_rows(self, table_markdown: str) -> int:
    """数据行数（跳过所有分隔线与重复表头）"""
    lines = self._lines(table_markdown)
    if len(lines) < 2:
      return 0
    headers = self._split_row(lines[0])
    return sum(
      1 for line in lines[1:]
      if (cells := self._split_row(line)) and not self._is_separator(cells) and cells != headers
     )

  def markdown_to_dataframe(self, table_markdown: str) -> pd.DataFrame:
    """
     将Markdown表格转换为DataFrame

     Args:
       table_markdown: Markdown格式的表格

     Returns:
       pandas DataFrame
     """
    lines = self._lines(table_markdown)

    if len(lines) < 2:
      return pd.DataFrame()

    # 解析表头
    headers = self._split_row(lines[0])

    # 解析数据行：跳过分隔线，以及跨页表格重复输出的表头
    data = []
    for line in lines[1:]:
      cells = self._split_row(line)
      if not cells or self._is_separator(cells) or cells == headers:
        continue
      if len(cells) == len(headers):
        data.append(cells)

    return pd.DataFrame(data, columns=headers)

  def table_to_text(self, table_markdown: str) -> str:
    """
     将表格转换为文本描述

     Args:
       table_markdown: Markdown格式的表格

     Returns:
       表格的文本描述
     """
    df = self.markdown_to_dataframe(table_markdown)

    if df.empty:
      return ""

    # 生成文本描述
    text_parts = []
    text_parts.append("表格内容：")

    # 添加表头
    text_parts.append(" | ".join(df.columns))

    # 添加数据行
    for _, row in df.iterrows():
      text_parts.append(" | ".join(str(v) for v in row))

    return "\n".join(text_parts)

  def _data_lines_markdown(self, table_markdown: str) -> str:
    """规范化表格：保留表头与分隔行，丢弃重复表头和多余分隔行"""
    lines = self._lines(table_markdown)
    if len(lines) < 2:
      return table_markdown.strip()
    headers = self._split_row(lines[0])
    kept, sep_kept = [lines[0]], False
    for line in lines[1:]:
      cells = self._split_row(line)
      if not cells:
        continue
      if self._is_separator(cells):
        if not sep_kept:
          kept.append(line)
          sep_kept = True
        continue
      if cells == headers:
        continue
      kept.append(line)
    return '\n'.join(kept)

  def merge_tables(self, tables: List[dict]) -> List[dict]:
    """
     合并跨页的拆分表格

     Args:
       tables: 表格列表，每项需含 markdown 字段；
               cols / rows / page 缺失时会自动计算

     Returns:
       合并后的表格列表（新对象，不修改入参）
     """
    if len(tables) <= 1:
      return [dict(t) for t in tables]

    def describe(table: dict) -> dict:
      """补齐 cols / rows 字段"""
      item = dict(table)
      markdown = item.get('markdown', '')
      item['cols'] = item.get('cols') or self._table_cols(markdown)
      item['rows'] = item.get('rows') or self._table_rows(markdown)
      item['markdown'] = self._data_lines_markdown(markdown)
      return item

    merged = [describe(tables[0])]

    for table in tables[1:]:
      current = describe(table)
      last = merged[-1]

      # 跨页拆分时 docling 会把表头重复输出一遍，表头完全一致才可能是同一张表；
      # 页码都存在时还要求页码连续，避免两张无关表格因表头相同被误合并
      same_header = (
        self._split_row(current['markdown'].split('\n')[0]) ==
        self._split_row(last['markdown'].split('\n')[0])
       )
      last_page, cur_page = last.get('page'), current.get('page')
      page_ok = (
        True if last_page is None or cur_page is None
        else cur_page == last_page + 1
       )

      if same_header and page_ok and current['cols'] == last['cols']:
        # 只拼接数据行：去掉第二块的表头，以及紧跟其后的分隔行
        rows = current['markdown'].split('\n')[1:]
        if rows and self._is_separator(self._split_row(rows[0])):
          rows = rows[1:]
        last['markdown'] += '\n' + '\n'.join(rows)
        last['rows'] += current['rows']
      else:
        merged.append(current)

    return merged
