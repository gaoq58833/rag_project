"""图片解析器：用视觉大模型提取描述/OCR，再走文本向量检索（文字找图）

流程：
  上传图片 → 压缩到长边 1568px → VLM 输出结构化描述 → 拼成可检索文本
  → 文本 embedding → Qdrant（INDEX_TYPE = "vector"）

原图保存到 data/images，chunk.images 记录路径，回答时可直接回传原图。
"""
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import settings
from src.llm.client import llm_client
from src.parsers.base import BaseParser, ParseResult

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 原图上限 10MB
MAX_SIDE = 1568                     # 发给 VLM 前的最长边，控制 token 与延迟
JPEG_QUALITY = 90

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_PROMPT = """请分析这张图片，严格只描述图中可见内容，不要推测或编造。
以 JSON 返回，格式如下：
{"summary": "一句话整体描述（中文）", "objects": ["主要对象或元素"], "ocr_text": "图中所有可辨认文字，原样照抄；没有则为空字符串", "tags": ["3-8 个检索关键词"]}
只输出 JSON，不要附加解释。"""


class ImageParser(BaseParser):
    """独立图片文件解析器：VLM 描述 + 文本向量化"""

    INDEX_TYPE = "vector"

    def __init__(self, image_dir: str = "data/images"):
        self.image_dir = image_dir

    def parse(self, file_path: str) -> ParseResult:
        raw = Path(file_path).read_bytes()
        return self._build(raw, Path(file_path).name, file_path)

    def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
        return self._build(file_bytes, filename, filename)

    def _build(self, raw: bytes, filename: str, file_path: str) -> ParseResult:
        ext = Path(filename).suffix.lower()
        mime = _MIME_BY_EXT.get(ext)
        if mime is None:
            raise ValueError(f"不支持的图片格式：{ext!r}")

        if len(raw) > MAX_IMAGE_BYTES:
            size_mb = len(raw) / 1024 / 1024
            raise ValueError(f"图片过大（{size_mb:.1f}MB），上限 {MAX_IMAGE_BYTES // 1024 // 1024}MB")

        prepared, width, height = self._prepare(raw)
        image_path = self._save(raw, ext)
        analysis = self._analyze(prepared)
        content = self._render_content(analysis)

        chunks = [{
            "content": content,
            "index": 0,
            "length": len(content),
            "type": "text",
            "section": "图片",
            "images": [image_path],
        }]

        return ParseResult(
            filename=filename,
            total_pages=0,
            total_chunks=1,
            table_count=0,
            chunks=chunks,
            raw_text=content,
            tables=[],
            metadata={
                "parser": "image",
                "index_type": self.INDEX_TYPE,
                "file_path": file_path,
                "image_path": image_path,
                "width": width,
                "height": height,
                "vision_model": settings.vision_model,
            },
        )

    @staticmethod
    def _prepare(raw: bytes) -> tuple:
        """校验图片并压缩为 JPEG（长边 <= MAX_SIDE），返回 (字节, 宽, 高)"""
        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                width, height = img.size
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=JPEG_QUALITY)
                return out.getvalue(), width, height
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"无法识别图片：{e}") from e

    def _save(self, raw: bytes, ext: str) -> str:
        """原图落盘，文件名用 UUID 避免冲突"""
        os.makedirs(self.image_dir, exist_ok=True)
        path = os.path.join(self.image_dir, f"image_{uuid.uuid4().hex[:12]}{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        return path.replace("\\", "/")

    @staticmethod
    def _analyze(image_bytes: bytes) -> Dict[str, Any]:
        """调用 VLM；模型没按 JSON 返回时，退化为把全文当 summary"""
        text = llm_client.chat_vision(_PROMPT, image_bytes, mime_type="image/jpeg")
        data = _extract_json(text)
        if data is None:
            return {"summary": text.strip()}
        return data

    @staticmethod
    def _render_content(analysis: Dict[str, Any]) -> str:
        """拼成适合 embedding 与字面检索的正文"""
        lines: List[str] = []
        summary = _as_text(analysis.get("summary"))
        ocr = _as_text(analysis.get("ocr_text"))
        objects = _as_list_text(analysis.get("objects"))
        tags = _as_list_text(analysis.get("tags"))

        if summary:
            lines.append(f"图片描述：{summary}")
        if ocr:
            lines.append(f"图中文字：{ocr}")
        if objects:
            lines.append(f"主要对象：{objects}")
        if tags:
            lines.append(f"关键词：{tags}")
        return "\n".join(lines) or "图片描述：无"

    def supported_extensions(self) -> List[str]:
        return [".jpg", ".jpeg", ".png", ".webp"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抠出第一个 JSON 对象，兼容 ```json 包裹"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _as_list_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()
