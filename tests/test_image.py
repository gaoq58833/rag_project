"""测试图片解析：VLM 描述 → 文本 chunk → 向量索引

用法：
  python tests/test_image.py

VLM 调用被 mock，不会产生真实 API 费用。
"""
import io
import json
import os
import sys
import tempfile
from unittest.mock import patch

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.llm.client import llm_client
from src.parsers.image_parser import ImageParser

ANALYSIS = {
    "summary": "红色背景的报错弹窗",
    "objects": ["弹窗", "按钮"],
    "ocr_text": "Error 401",
    "tags": ["报错", "401", "鉴权"],
}


def _png_bytes(width: int = 1600, height: int = 900) -> bytes:
    img = Image.new("RGB", (width, height), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_vision(text: str):
    def fake(prompt, image_bytes, mime_type="image/jpeg", **kwargs):
        assert "JSON" in prompt
        assert image_bytes[:2] == b"\xff\xd8"  # 已转成 JPEG 再送模型
        return text
    return fake


def test_image_chunk():
    raw = _png_bytes()
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(llm_client, "chat_vision", _fake_vision(json.dumps(ANALYSIS, ensure_ascii=False))):
        result = ImageParser(image_dir=tmp).parse_from_bytes(raw, "shot.png")

        assert result.metadata["parser"] == "image"
        assert result.metadata["index_type"] == "vector"
        assert result.metadata["vision_model"] == settings.vision_model
        assert result.metadata["width"] == 1600 and result.metadata["height"] == 900

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk["section"] == "图片"
        assert chunk["type"] == "text"
        assert "图片描述：红色背景的报错弹窗" in chunk["content"]
        assert "图中文字：Error 401" in chunk["content"]
        assert "主要对象：弹窗, 按钮" in chunk["content"]
        assert "关键词：报错, 401, 鉴权" in chunk["content"]

        # 原图已落盘，chunk.images 指向该文件
        assert len(chunk["images"]) == 1
        assert os.path.isfile(chunk["images"][0])
        assert os.path.basename(chunk["images"][0]).startswith("image_")

        assert result.raw_text == chunk["content"]
        assert result.total_chunks == 1


def test_non_json_fallback():
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(llm_client, "chat_vision", _fake_vision("一张红色背景的图片")):
        result = ImageParser(image_dir=tmp).parse_from_bytes(_png_bytes(800, 600), "a.jpg")
        assert result.chunks[0]["content"] == "图片描述：一张红色背景的图片"


def test_invalid_and_oversized():
    with tempfile.TemporaryDirectory() as tmp:
        parser = ImageParser(image_dir=tmp)
        try:
            parser.parse_from_bytes(b"not an image", "bad.png")
        except ValueError as e:
            assert "无法识别图片" in str(e)
        else:
            raise AssertionError("非法图片未抛 ValueError")

        try:
            parser.parse_from_bytes(b"x" * (10 * 1024 * 1024 + 1), "big.png")
        except ValueError as e:
            assert "图片过大" in str(e)
        else:
            raise AssertionError("超大图片未抛 ValueError")


def test_parse_file():
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(llm_client, "chat_vision", _fake_vision(json.dumps(ANALYSIS, ensure_ascii=False))):
        path = os.path.join(tmp, "shot.webp")
        Image.new("RGB", (640, 480), (20, 120, 200)).save(path, format="WEBP")
        result = ImageParser(image_dir=tmp).parse(path)
        assert result.filename == "shot.webp"
        assert len(result.chunks) == 1


def main():
    tests = [test_image_chunk, test_non_json_fallback, test_invalid_and_oversized, test_parse_file]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(llm_client, "chat_vision", _fake_vision(json.dumps(ANALYSIS, ensure_ascii=False))):
        result = ImageParser(image_dir=tmp).parse_from_bytes(_png_bytes(), "shot.png")
    print(f"\n索引类型：{result.metadata['index_type']}，vision_model：{result.metadata['vision_model']}")
    print(result.chunks[0]["content"])
    print(f"原图：{result.chunks[0]['images'][0]}")


if __name__ == "__main__":
    main()
