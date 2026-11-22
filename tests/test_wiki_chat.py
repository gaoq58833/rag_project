"""P4：Wiki 融入问答上下文测试（torch-free，mock chunk/Wiki 检索）。

覆盖：上下文拼接格式、来源合并与 type 标记、WIKI_IN_CHAT 开关、
Wiki 检索失败降级（不阻断问答 + error 显式化）、wiki-only 仍产出资料。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wiki import chat_integration  # noqa: E402
import src.wiki.service as wiki_service  # noqa: E402
from src.config import settings  # noqa: E402

SYSTEM_PROMPT = "你是知识库问答助手。"

CHUNK_SOURCES = [
    {"content": "RNN 逐个读取输入词。", "section": "3.1", "type": "text", "score": 0.9},
    {"content": "隐藏状态在时间步间传递。", "section": "3.2", "type": "text", "score": 0.8},
]
WIKI_PAGES = [
    {"slug": "rnn", "title": "循环神经网络", "summary": "处理序列",
     "content": "RNN 通过隐藏状态处理序列。", "score": 0.95},
]


def _make_retrieve(sources):
    calls = []
    def fake_retrieve(query, user_id=None, use_rerank=True, timings=None):
        calls.append({"query": query, "user_id": user_id})
        if timings is not None:
            timings["retrieve_total_s"] = 0.01
        return list(sources)
    return fake_retrieve, calls


def _with_wiki(enabled, fn):
    old = settings.wiki_in_chat
    settings.wiki_in_chat = enabled
    try:
        return fn()
    finally:
        settings.wiki_in_chat = old


def case(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return False


# ---------- 测试用例 ----------

def test_assemble_context_format():
    context_msg, all_sources = chat_integration.assemble_context(
        CHUNK_SOURCES, WIKI_PAGES, "RNN 是什么？"
    )
    # Wiki 块在前
    assert "## 知识库结构化页面（Wiki）" in context_msg
    assert "[W1] 《循环神经网络》" in context_msg
    assert "摘要：处理序列" in context_msg
    # 文档片段块在后
    assert "## 文档片段" in context_msg
    assert "[1] [3.1] RNN 逐个读取输入词。" in context_msg
    # 问题分离
    assert context_msg.endswith("用户问题：RNN 是什么？")
    # 来源：wiki 项在前，type 标记
    assert all_sources[0]["type"] == "wiki"
    assert all_sources[0]["slug"] == "rnn"
    assert all_sources[0]["title"] == "循环神经网络"
    assert all_sources[1]["type"] == "text"  # chunk 原样保留
    assert len(all_sources) == 3


def test_assemble_empty_both():
    context_msg, all_sources = chat_integration.assemble_context([], [], "你好")
    assert context_msg == "你好"
    assert all_sources == []


def test_build_fusion_enabled():
    retrieve_chunks, chunk_calls = _make_retrieve(CHUNK_SOURCES)
    wiki_calls = []

    def fake_wiki(query, user_id, top_k=3):
        wiki_calls.append({"query": query, "user_id": user_id, "top_k": top_k})
        return list(WIKI_PAGES)

    orig = wiki_service.retrieve_wiki
    wiki_service.retrieve_wiki = fake_wiki
    try:
        timings = {}
        messages, all_sources, wiki_info = _with_wiki(True, lambda: chat_integration.build_chat_context(
            "RNN？", "u1", [{"role": "user", "content": "hi"}], True, timings,
            retrieve_chunks, SYSTEM_PROMPT,
        ))
    finally:
        wiki_service.retrieve_wiki = orig

    assert chunk_calls and chunk_calls[0]["user_id"] == "u1"
    assert wiki_calls and wiki_calls[0]["top_k"] == chat_integration.WIKI_TOP_K
    assert wiki_info["used"] is True and wiki_info["count"] == 1 and wiki_info["error"] is None
    assert "wiki_s" in timings
    # messages: system + 1 history + user
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "知识库结构化页面" in messages[-1]["content"]
    assert all_sources[0]["type"] == "wiki"


def test_build_disabled_no_wiki_call():
    retrieve_chunks, _ = _make_retrieve(CHUNK_SOURCES)
    called = {"n": 0}

    def fake_wiki(*a, **k):
        called["n"] += 1
        return WIKI_PAGES

    orig = wiki_service.retrieve_wiki
    wiki_service.retrieve_wiki = fake_wiki
    try:
        timings = {}
        messages, all_sources, wiki_info = _with_wiki(False, lambda: chat_integration.build_chat_context(
            "q", "u1", [], True, timings, retrieve_chunks, SYSTEM_PROMPT,
        ))
    finally:
        wiki_service.retrieve_wiki = orig

    assert called["n"] == 0, "WIKI_IN_CHAT=false 时不应调用 retrieve_wiki"
    assert wiki_info["used"] is False
    assert "wiki_s" not in timings
    assert all(s.get("type") != "wiki" for s in all_sources)
    assert "知识库结构化页面" not in messages[-1]["content"]


def test_build_wiki_error_degrades():
    retrieve_chunks, _ = _make_retrieve(CHUNK_SOURCES)

    def boom(*a, **k):
        raise RuntimeError("qdrant down")

    orig = wiki_service.retrieve_wiki
    wiki_service.retrieve_wiki = boom
    try:
        timings = {}
        messages, all_sources, wiki_info = _with_wiki(True, lambda: chat_integration.build_chat_context(
            "q", "u1", [], True, timings, retrieve_chunks, SYSTEM_PROMPT,
        ))
    finally:
        wiki_service.retrieve_wiki = orig

    # 不崩：chunk 来源仍在，error 显式暴露
    assert wiki_info["used"] is False
    assert wiki_info["error"] and "RuntimeError" in wiki_info["error"]
    assert len(all_sources) == 2  # 只有 chunk
    assert "文档片段" in messages[-1]["content"]
    assert "知识库结构化页面" not in messages[-1]["content"]
    assert "wiki_s" in timings  # 即使失败也记录耗时


def test_build_wiki_only_bypasses_empty_fallback():
    """chunk 为空但 Wiki 有页面 → 仍应产出资料（all_sources 非空），pipeline 据此不触发兜底。"""
    retrieve_chunks, _ = _make_retrieve([])  # 无 chunk

    def fake_wiki(*a, **k):
        return list(WIKI_PAGES)

    orig = wiki_service.retrieve_wiki
    wiki_service.retrieve_wiki = fake_wiki
    try:
        _, all_sources, wiki_info = _with_wiki(True, lambda: chat_integration.build_chat_context(
            "q", "u1", [], True, {}, retrieve_chunks, SYSTEM_PROMPT,
        ))
    finally:
        wiki_service.retrieve_wiki = orig

    assert wiki_info["used"] is True
    assert len(all_sources) == 1 and all_sources[0]["type"] == "wiki"


if __name__ == "__main__":
    print("== P4 Wiki 融入问答上下文测试 ==")
    results = [
        case("assemble_context 拼接格式 + 来源合并/type 标记", test_assemble_context_format),
        case("assemble_context 双空时 context=query、sources 空", test_assemble_empty_both),
        case("build 融合：chunk+Wiki 都检索、messages 含 Wiki 块、wiki_info 正确", test_build_fusion_enabled),
        case("WIKI_IN_CHAT=false 不调用 retrieve_wiki、无 wiki 块", test_build_disabled_no_wiki_call),
        case("Wiki 检索失败：降级仅用 chunk + error 显式化 + 不崩", test_build_wiki_error_degrades),
        case("chunk 空但 Wiki 有页 → 仍产出资料（不触发兜底）", test_build_wiki_only_bypasses_empty_fallback),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)
