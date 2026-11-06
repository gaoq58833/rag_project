"""测试以图搜图：多模态 embedding 客户端 + 图片向量检索

用法：
  python tests/test_image_search.py

httpx / embedding / Qdrant / PG 全部 mock，不产生真实 API 调用与费用。
"""
import io
import os
import sys
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.llm.client import llm_client
from src.rag.pipeline import retrieve_images
from src.vectorstore import image_vector_store

FAKE_VEC = [0.1] * settings.mm_embedding_dim


def _png_bytes(width: int = 800, height: int = 600) -> bytes:
    img = Image.new("RGB", (width, height), (50, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_httpx_client(response_json: dict):
    """构造 src.llm.client.httpx.Client 的替身，返回指定 JSON"""
    resp = MagicMock()
    resp.json.return_value = response_json
    resp.raise_for_status.return_value = None
    http = MagicMock()
    http.post.return_value = resp
    client_cls = MagicMock()
    client_cls.return_value.__enter__.return_value = http
    client_cls.return_value.__exit__.return_value = False
    return client_cls, http


def test_embed_image():
    """图片 embedding：走 DashScope 原生接口，图片 base64 内联"""
    client_cls, http = _mock_httpx_client({"output": {"embeddings": [{"embedding": FAKE_VEC}]}})
    with patch("src.llm.client.httpx.Client", client_cls):
        vec = llm_client.embed_image(b"\xff\xd8fake-jpeg", mime_type="image/jpeg")

    assert vec == FAKE_VEC
    args, kwargs = http.post.call_args
    url = args[0]
    assert url.endswith("/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding")
    assert url.startswith(settings.mm_embedding_base_url.rstrip("/"))
    assert kwargs["headers"]["Authorization"] == f"Bearer {settings.mm_embedding_api_key}"
    payload = kwargs["json"]
    assert payload["model"] == settings.mm_embedding_model
    contents = payload["input"]["contents"]
    assert len(contents) == 1 and contents[0]["image"].startswith("data:image/jpeg;base64,")


def test_embed_mm_text():
    """文本 embedding：与图片同一接口同一向量空间"""
    client_cls, http = _mock_httpx_client({"output": {"embeddings": [{"embedding": FAKE_VEC}]}})
    with patch("src.llm.client.httpx.Client", client_cls):
        vec = llm_client.embed_mm_text("一只猫")

    assert vec == FAKE_VEC
    contents = http.post.call_args.kwargs["json"]["input"]["contents"]
    assert contents == [{"text": "一只猫"}]


def test_image_collection_config():
    """图片向量集合独立于文本集合，维度用 MM_EMBEDDING_DIM"""
    assert image_vector_store.collection == settings.qdrant_image_collection
    assert image_vector_store.collection != settings.qdrant_collection
    assert image_vector_store.vector_size == settings.mm_embedding_dim


def test_retrieve_images():
    """以图搜图：预处理 → embed → 按 user_id 过滤检索 → 回 PG 组装结果"""
    candidates = [
        {"payload": {"qdrant_point_id": "pid-1", "user_id": "u1", "filename": "cat.png"}, "score": 0.91},
        {"payload": {"qdrant_point_id": "pid-2", "user_id": "u1", "filename": "dog.png"}, "score": 0.72},
        {"payload": {"qdrant_point_id": "pid-3", "user_id": "u1", "filename": "gone.png"}, "score": 0.50},
    ]
    pg_rows = [
        {"qdrant_point_id": "pid-1", "content": "图片描述：一只橘猫", "images": ["data/images/a.png"]},
        {"qdrant_point_id": "pid-2", "content": "图片描述：一只柴犬", "images": ["data/images/b.png"]},
        # pid-3 在 PG 中不存在（已删除残留），应被跳过
    ]

    with patch.object(llm_client, "embed_image", return_value=FAKE_VEC) as mock_embed, \
            patch.object(image_vector_store, "search", return_value=candidates) as mock_search, \
            patch("src.rag.pipeline.ChunkRepository.get_by_qdrant_ids", return_value=pg_rows) as mock_pg:
        results = retrieve_images(_png_bytes(), user_id="u1", top_k=5)

    # embed_image 收到的是预处理后的 JPEG 字节
    (prepared,), _ = mock_embed.call_args
    assert prepared[:2] == b"\xff\xd8"

    # 检索按 user_id 过滤，top_k 透传
    _, search_kwargs = mock_search.call_args
    assert search_kwargs["top_k"] == 5
    conditions = search_kwargs["filter_"].must
    assert any(c.key == "user_id" and c.match.value == "u1" for c in conditions)

    # 回查 PG 用的正是检索到的 point_id
    (pg_ids,), _ = mock_pg.call_args
    assert pg_ids == ["pid-1", "pid-2", "pid-3"]

    # 结果按检索顺序、带相似度分数与 PG 数据，残留点被跳过
    assert [r["filename"] for r in results] == ["cat.png", "dog.png"]
    assert results[0]["score"] == 0.91
    assert results[0]["content"] == "图片描述：一只橘猫"
    assert results[0]["images"] == ["data/images/a.png"]


def test_retrieve_images_empty():
    """集合中没有命中时直接返回空列表，不查 PG"""
    with patch.object(llm_client, "embed_image", return_value=FAKE_VEC), \
            patch.object(image_vector_store, "search", return_value=[]), \
            patch("src.rag.pipeline.ChunkRepository.get_by_qdrant_ids") as mock_pg:
        results = retrieve_images(_png_bytes(), user_id="u1")
    assert results == []
    mock_pg.assert_not_called()


def test_retrieve_images_invalid():
    """非法图片字节抛 ValueError"""
    try:
        retrieve_images(b"not an image", user_id="u1")
    except ValueError as e:
        assert "无法识别图片" in str(e)
    else:
        raise AssertionError("非法图片未抛 ValueError")


def main():
    tests = [
        test_embed_image,
        test_embed_mm_text,
        test_image_collection_config,
        test_retrieve_images,
        test_retrieve_images_empty,
        test_retrieve_images_invalid,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
