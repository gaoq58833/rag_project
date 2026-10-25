from pathlib import Path
from typing import List
import os
from src.parsers.base import BaseParser, ParseResult
from src.parsers.table_parser import TableProcessor
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

class PDFParser(BaseParser):
  """基于Docling的PDF解析器"""
  def __init__(self, images_scale: float = 1.0, ocr_scale: float = 1.5,
               do_ocr: bool = True, image_dir: str = "data/images"):
    self.table_processor = TableProcessor()
    self.image_dir = image_dir
    pipeline_options = PdfPipelineOptions()
    # 默认不生成图片位图，需显式开启，否则图片只有占位符
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = images_scale
    # OCR 是内存大头，文字版 PDF 可关掉；开启时把默认放大倍数 3.0 调小以压缩内存
    pipeline_options.do_ocr = do_ocr
    pipeline_options.ocr_options.scale = ocr_scale
    # 降低批大小，压缩峰值内存
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    self.converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
       }
     )
  def parse(self, file_path: str) -> ParseResult:
    # 使用Docling转换为Document对象
    doc = self.converter.convert(file_path)
    # 提取内容（图片保存到本地，markdown 里只留引用路径）
    raw_text = self._export_markdown(doc, Path(file_path).stem)
    # 提取表格
    tables = self._extract_tables(doc)
    return ParseResult(
      filename=Path(file_path).name,
      total_pages=len(doc.pages),
      total_chunks=len(raw_text),
      table_count=len(tables),
      chunks=[],
      raw_text=raw_text,
      tables=tables,
      metadata={"parser":"docling","file_path":file_path}
     )
  def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
    # 使用Docling转换为Document对象
    doc = self.converter.convert_from_bytes(file_bytes)
    # 提取内容（图片保存到本地，markdown 里只留引用路径）
    raw_text = self._export_markdown(doc, Path(filename).stem)
    # 提取表格
    tables = self._extract_tables(doc)
    return ParseResult(
      filename=filename,
      total_pages=len(doc.pages),
      total_chunks=len(raw_text),
      table_count=len(tables),
      chunks=[],
      raw_text=raw_text,
      tables=tables,
      metadata={"parser":"docling","file_path":filename}
     )
  def _export_markdown(self, doc, stem: str) -> str:
    """导出 markdown：图片存到本地目录，正文里只留 ![图片](路径) 引用，不内嵌 base64"""
    md = doc.document.export_to_markdown()  # 默认占位符模式，图片位置是 <!-- image -->
    pictures = list(getattr(doc.document, "pictures", []) or [])
    if not pictures:
      return md
    parts = md.split("<!-- image -->")
    if len(parts) - 1 != len(pictures):
      # 占位符和图片数量对不上时退回 base64 内嵌，避免图片错位
      return doc.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    os.makedirs(self.image_dir, exist_ok=True)
    out = [parts[0]]
    for i, pic in enumerate(pictures, 1):
      img = pic.get_image(doc.document)
      path = os.path.join(self.image_dir, f"{stem}_{i}.png")
      img.save(path)
      ref = path.replace("\\", "/")
      out.append(f"![图片{i}]({ref})")
      out.append(parts[i])
    return "".join(out)
  def _extract_tables(self, doc) -> List:
    tables = []

    # 直接从 document 对象获取所有表格
    if not hasattr(doc.document, "tables") or not doc.document.tables:
      return tables

    for i, table in enumerate(doc.document.tables, 1):
      try:
        # 获取表格所在的页码（通过 prov 属性）
        prov = getattr(table, "prov", [])
        page_no = prov[0].page_no if prov else None

        table_data = {
          "table_number": i,
          "page": page_no,
          "markdown": table.export_to_markdown(doc=doc.document),
         }
        tables.append(table_data)
      except Exception as e:
        print(f"警告：无法处理表格 {i}: {e}")
        continue

    # 合并被 docling 按页拆开的跨页表格
    return self.table_processor.merge_tables(tables)

  def supported_extensions(self) -> List[str]:
    return [".pdf", ".PDF"]

