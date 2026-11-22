"""chat_json 契约测试（不依赖网络/数据库，mock 底层 _chat_completion）。

验证：
1. 正常 JSON object → 返回 dict
2. 带 ```json 围栏 → 自动剥离后解析
3. 解析失败 → 重试后仍失败则显式抛 ValueError（不静默返回空）
4. 顶层非 object（数组/字符串）→ 抛 ValueError
5. response_format 不被网关支持 → 首次异常后退化为 prompt 约束重试
6. 不修改调用方传入的 messages（无副作用）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import LLMClient


def make_client():
    return LLMClient()


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


def test_normal_object():
    c = make_client()
    c._chat_completion = lambda *a, **k: '{"title": "RNN", "summary": "循环神经网络"}'
    data = c.chat_json([{"role": "user", "content": "生成页面"}], schema_hint='{"title":str,"summary":str}')
    assert isinstance(data, dict), f"期望 dict，得到 {type(data)}"
    assert data["title"] == "RNN"
    assert data["summary"] == "循环神经网络"


def test_strip_code_fence():
    c = make_client()
    c._chat_completion = lambda *a, **k: '```json\n{"a": 1}\n```'
    data = c.chat_json([{"role": "user", "content": "x"}])
    assert data == {"a": 1}, data


def test_parse_failure_raises():
    c = make_client()
    c._chat_completion = lambda *a, **k: "这不是 JSON，是一段自然语言回答"
    try:
        c.chat_json([{"role": "user", "content": "x"}], retries=1)
    except ValueError:
        return
    raise AssertionError("解析失败应抛 ValueError")


def test_non_object_raises():
    c = make_client()
    c._chat_completion = lambda *a, **k: '["a", "b"]'
    try:
        c.chat_json([{"role": "user", "content": "x"}], retries=0)
    except ValueError:
        return
    raise AssertionError("顶层数组应抛 ValueError")


def test_response_format_unsupported_fallback():
    """首次调用因 response_format 抛异常 → 退化为无 response_format 重试并成功"""
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            # 模拟网关不支持 response_format 参数
            raise Exception("Unsupported parameter: response_format")
        return '{"ok": true}'

    c = make_client()
    c._chat_completion = fake
    data = c.chat_json([{"role": "user", "content": "x"}], retries=1)
    assert data == {"ok": True}, data
    assert calls["n"] == 2, f"应调用 2 次（首次异常 + 退化重试），实际 {calls['n']}"


def test_no_mutation_of_input():
    c = make_client()
    captured = {}

    def fake(messages, *a, **k):
        captured["msgs"] = messages
        return '{"x": 1}'

    c._chat_completion = fake
    original = [{"role": "user", "content": "保持原样"}]
    c.chat_json(original)
    # 入参不应被修改
    assert original == [{"role": "user", "content": "保持原样"}], original
    # 传给底层的应带 system 指令（追加而非覆盖）
    assert captured["msgs"][0]["role"] == "system"
    assert "JSON" in captured["msgs"][0]["content"]
    assert any(m["role"] == "user" and m["content"] == "保持原样" for m in captured["msgs"])


def test_system_merged_when_present():
    """已有 system 消息时，指令应追加进去，不产生两条 system"""
    c = make_client()
    captured = {}

    def fake(messages, *a, **k):
        captured["msgs"] = messages
        return '{"x": 1}'

    c._chat_completion = fake
    c.chat_json([{"role": "system", "content": "原系统提示"}, {"role": "user", "content": "hi"}])
    sys_msgs = [m for m in captured["msgs"] if m["role"] == "system"]
    assert len(sys_msgs) == 1, f"应只有 1 条 system，实际 {len(sys_msgs)}"
    assert sys_msgs[0]["content"].startswith("原系统提示")
    assert "JSON" in sys_msgs[0]["content"]


if __name__ == "__main__":
    print("== chat_json 契约测试 ==")
    results = [
        case("正常 JSON object 解析", test_normal_object),
        case("剥离 ```json 围栏", test_strip_code_fence),
        case("解析失败显式抛 ValueError", test_parse_failure_raises),
        case("顶层非 object 抛 ValueError", test_non_object_raises),
        case("response_format 不支持时退化重试", test_response_format_unsupported_fallback),
        case("不修改入参 messages", test_no_mutation_of_input),
        case("已有 system 时合并而非新增", test_system_merged_when_present),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)
