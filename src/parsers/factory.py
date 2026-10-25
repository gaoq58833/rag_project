"""解析器工厂：根据文件扩展名自动选择解析器，并复用单例实例。

所有解析器的 parse() 均不修改实例状态，可安全复用：
- PDFParser 的 DocumentConverter 加载模型较重，复用可避免重复加载
- 其他解析器 __init__ 仅创建 TextCleaner/TableProcessor，开销极低，复用也无副作用
"""
from pathlib import Path
from typing import Dict, List, Type

from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.word_parser import WordParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.html_parser import HTMLParser
from src.parsers.txt_parser import TXTParser

# 扩展名（小写，含点）-> 解析器类
_PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    ".pdf": PDFParser,
    ".docx": WordParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
    ".txt": TXTParser,
    ".text": TXTParser,
}

# 单例缓存：默认参数的解析器只创建一次
_instances: Dict[Type[BaseParser], BaseParser] = {}


def get_parser(file_path: str, **kwargs) -> BaseParser:
    """根据文件路径的扩展名返回对应解析器实例。

    默认参数走单例缓存；传入自定义参数时每次新建（不缓存）。

    Args:
        file_path: 文件路径或文件名，用于识别扩展名
        **kwargs: 传给解析器构造函数的自定义参数

    Raises:
        ValueError: 不支持的扩展名
    """
    ext = Path(file_path).suffix.lower()
    parser_cls = _PARSER_REGISTRY.get(ext)
    if parser_cls is None:
        supported = ", ".join(sorted(_PARSER_REGISTRY))
        raise ValueError(f"不支持的文件格式：{ext!r}，支持的格式：{supported}")

    if kwargs:
        # 自定义参数不缓存，避免参数不同的实例互相覆盖
        return parser_cls(**kwargs)

    if parser_cls not in _instances:
        _instances[parser_cls] = parser_cls()
    return _instances[parser_cls]


def parse(file_path: str, **kwargs):
    """便捷方法：自动选解析器并解析文件，等价于 get_parser(file_path).parse(file_path)"""
    return get_parser(file_path, **kwargs).parse(file_path)


def parse_from_bytes(file_bytes: bytes, filename: str, **kwargs):
    """便捷方法：从字节数据解析，filename 用于识别扩展名"""
    return get_parser(filename, **kwargs).parse_from_bytes(file_bytes, filename)


def supported_extensions() -> List[str]:
    """返回所有支持的扩展名（小写，含点）"""
    return sorted(_PARSER_REGISTRY)


def register_parser(ext: str, parser_cls: Type[BaseParser]) -> None:
    """注册新的扩展名 -> 解析器映射（用于扩展支持的格式）"""
    _PARSER_REGISTRY[ext.lower()] = parser_cls
