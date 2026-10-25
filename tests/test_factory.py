"""测试解析器工厂：扩展名路由、单例复用、自定义参数、错误处理"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.factory import get_parser, parse, supported_extensions, register_parser
from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.word_parser import WordParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.html_parser import HTMLParser
from src.parsers.txt_parser import TXTParser


def test_routing():
    """扩展名 -> 解析器类 路由正确"""
    assert isinstance(get_parser("a.pdf"), PDFParser)
    assert isinstance(get_parser("a.docx"), WordParser)
    assert isinstance(get_parser("a.md"), MarkdownParser)
    assert isinstance(get_parser("a.markdown"), MarkdownParser)
    assert isinstance(get_parser("a.html"), HTMLParser)
    assert isinstance(get_parser("a.htm"), HTMLParser)
    assert isinstance(get_parser("a.txt"), TXTParser)
    assert isinstance(get_parser("a.TEXT"), TXTParser)  # 大小写不敏感
    assert isinstance(get_parser("/path/to/file.PDF"), PDFParser)
    print("[OK] 扩展名路由正确")


def test_singleton():
    """默认参数的解析器返回同一实例"""
    p1 = get_parser("a.txt")
    p2 = get_parser("b.txt")
    assert p1 is p2, "默认参数应返回单例"

    p3 = get_parser("x.docx")
    p4 = get_parser("y.docx")
    assert p3 is p4
    print("[OK] 单例复用生效")


def test_kwargs_not_cached():
    """传自定义参数时不缓存，每次新建"""
    p1 = get_parser("a.pdf", do_ocr=False)
    p2 = get_parser("a.pdf", do_ocr=False)
    assert p1 is not p2, "自定义参数不应走缓存"
    print("[OK] 自定义参数每次新建")


def test_unsupported():
    """不支持的扩展名抛 ValueError"""
    try:
        get_parser("a.xyz")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "不支持" in str(e)
    print("[OK] 不支持格式抛错")


def test_supported_extensions():
    exts = supported_extensions()
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".txt" in exts
    print(f"[OK] 支持 {len(exts)} 种扩展名：{exts}")


def test_register():
    """动态注册新格式"""
    register_parser(".foo", TXTParser)
    assert isinstance(get_parser("a.foo"), TXTParser)
    print("[OK] 动态注册生效")


def test_parse_convenience():
    """parse() 便捷方法可用"""
    result = parse(os.path.join("test_file", "sample.txt"))
    assert result.raw_text
    assert result.metadata["parser"] == "txt"
    print(f"[OK] parse() 便捷方法解析成功，{len(result.raw_text)} 字")


def main():
    test_routing()
    test_singleton()
    test_kwargs_not_cached()
    test_unsupported()
    test_supported_extensions()
    test_register()
    test_parse_convenience()
    print("\n所有工厂测试通过 ✅")


if __name__ == "__main__":
    main()
