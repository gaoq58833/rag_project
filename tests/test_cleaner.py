"""测试 TextCleaner 文本清洗器"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.cleaner import TextCleaner

def test_cleaner():
  """测试文本清洗功能"""
  print("=== 测试 TextCleaner ===")
  cleaner = TextCleaner()

  # 测试文本
  test_text = """第1页

这是一段测试文本。

第2页

这是另一段内容。"""

  cleaned = cleaner.clean(test_text)
  print("清洗后文本:")
  print(cleaned)

  assert "第1页" not in cleaned
  assert "第2页" not in cleaned
  assert "测试文本" in cleaned

  print("[OK] 文本清洗测试通过")

if __name__ == "__main__":
  test_cleaner()
