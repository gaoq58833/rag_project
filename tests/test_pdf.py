# 测试PDF解析器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.pdf_parser import PDFParser
from src.parsers.cleaner import TextCleaner

def test_pdf_parser():
  parser = PDFParser()
  result = parser.parse("./test_file/基于MindSpore的RNN情感分类实验.pdf")

  # 整份 PDF 已由 docling 转为 markdown，这里只做逐行去噪（保留 markdown 结构）
  cleaner = TextCleaner()
  markdown = cleaner.clean(result.raw_text)

  print(f"页数: {result.total_pages}, 表格数: {result.table_count}, markdown 字符数: {len(markdown)}")
  with open('./tests/a.md','w',encoding='utf-8') as f:
    f.write(markdown)

if __name__ == "__main__":
  test_pdf_parser()
