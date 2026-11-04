"""测试 JSON 解析：记录级预分块 + BM25 倒排索引类型

用法：
  python tests/test_json.py

验证点：
- 一条记录一个 chunk，section 记录 JSON 路径
- 字段名保留原文并额外拆词（camelCase / snake_case），保证 BM25 字面召回
- 非法 JSON 抛 ValueError
- 工厂按 .json 路由到 JSONParser，索引类型为 bm25
"""
import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.fts import tokenize
from src.parsers.factory import get_parser, index_type_for
from src.parsers.json_parser import JSONParser

SAMPLE = {
    "users": [
        {
            "id": 1,
            "name": "张三",
            "errorCode": "E1024",
            "profile": {"city": "深圳"},
        },
        {
            "id": 2,
            "name": "李四",
            "tags": ["vip", "new"],
        },
    ],
    "config": {"port": 8000, "dbHost": "localhost", "enabled": True},
}


def test_factory_and_index_type():
    parser = get_parser("sample.json")
    assert isinstance(parser, JSONParser), type(parser)
    assert parser.supported_extensions() == [".json"]
    assert parser.INDEX_TYPE == "bm25"
    assert index_type_for("sample.json") == "bm25"
    assert index_type_for("sample.pdf") == "vector"
    assert index_type_for("unknown.xyz") == "vector"


def test_records_and_tokens():
    result = JSONParser().parse_from_bytes(
        json.dumps(SAMPLE, ensure_ascii=False).encode("utf-8"), "sample.json"
    )

    # users[0] / users[1] / config 各一条记录
    assert len(result.chunks) == 3, [c["section"] for c in result.chunks]
    assert [c["section"] for c in result.chunks] == ["users[0]", "users[1]", "config"]

    first = result.chunks[0]["content"]
    assert "id: 1" in first
    assert "profile.city: 深圳" in first
    # 字段名保留原文 + 拆词形式
    assert "errorCode (error code): E1024" in first

    # BM25 分词应同时命中原文和拆词形式
    tokens = tokenize(first).split()
    assert "errorcode" in tokens
    assert "error" in tokens and "code" in tokens

    assert "tags: vip, new" in result.chunks[1]["content"]
    config = result.chunks[2]["content"]
    assert "dbHost (db host): localhost" in config
    assert "enabled: true" in config

    # raw_text 按 section 分节，便于 /parse 直接展示
    assert "## users[0]" in result.raw_text
    assert result.metadata["parser"] == "json"
    assert result.metadata["index_type"] == "bm25"
    assert result.metadata["top_level_keys"] == ["users", "config"]
    assert result.total_chunks == 3


def test_root_shapes():
    parser = JSONParser()

    # 纯标量数组：整体一条
    result = parser.parse_from_bytes(b"[1, 2, 3]", "numbers.json")
    assert len(result.chunks) == 1
    assert result.chunks[0]["section"] == "root"
    assert result.chunks[0]["content"] == "1, 2, 3"

    # 对象数组：每个元素一条
    result = parser.parse_from_bytes(b'[{"a": 1}, {"a": 2}]', "list.json")
    assert [c["section"] for c in result.chunks] == ["root[0]", "root[1]"]


def test_long_record_split():
    record = {f"field_{i}": "x" * 40 for i in range(40)}
    data = [record]
    result = JSONParser().parse_from_bytes(
        json.dumps(data, ensure_ascii=False).encode("utf-8"), "long.json"
    )
    assert len(result.chunks) > 1
    assert all(c["length"] <= 1000 for c in result.chunks)
    assert all(c["section"] == "root[0]" for c in result.chunks)


def test_invalid_json():
    try:
        JSONParser().parse_from_bytes(b'{"a": 1,}', "bad.json")
    except ValueError as e:
        assert "JSON 解析失败" in str(e)
    else:
        raise AssertionError("非法 JSON 未抛 ValueError")


def test_parse_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE, f, ensure_ascii=False)
        result = JSONParser().parse(path)
        assert len(result.chunks) == 3
        assert result.filename == "sample.json"


def main():
    tests = [
        test_factory_and_index_type,
        test_records_and_tokens,
        test_root_shapes,
        test_long_record_split,
        test_invalid_json,
        test_parse_file,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    result = JSONParser().parse_from_bytes(
        json.dumps(SAMPLE, ensure_ascii=False).encode("utf-8"), "sample.json"
    )
    print(f"\n块数：{len(result.chunks)}，索引类型：{result.metadata['index_type']}")
    for c in result.chunks:
        print(f"  [{c['index']}] section={c['section']!r} length={c['length']}")
        print(f"      {c['content'].replace(chr(10), ' | ')}")
    print(f"\ntokenize 示例：{tokenize(result.chunks[0]['content'])}")


if __name__ == "__main__":
    main()
