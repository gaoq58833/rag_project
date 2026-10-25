"""快速验证 API 接口：启动服务后运行此脚本"""
import json
import os
import sys

import requests

BASE = "http://localhost:8000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print(f"[OK] /health -> {r.json()}")


def test_formats():
    r = requests.get(f"{BASE}/formats")
    assert r.status_code == 200
    data = r.json()
    assert ".pdf" in data["supported_extensions"]
    assert ".txt" in data["supported_extensions"]
    print(f"[OK] /formats -> {len(data['supported_extensions'])} 种格式")


def test_parse_txt():
    path = os.path.join(ROOT, "test_file", "sample.txt")
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/parse", files={"file": ("sample.txt", f, "text/plain")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["filename"] == "sample.txt"
    assert d["table_count"] == 0
    assert len(d["raw_text"]) > 0
    print(f"[OK] /parse (txt) -> {len(d['raw_text'])} 字, table_count={d['table_count']}")


def test_chunk_txt():
    path = os.path.join(ROOT, "test_file", "sample.txt")
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/chunk", files={"file": ("sample.txt", f, "text/plain")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["chunk_count"] > 0
    for c in d["chunks"]:
        assert "content" in c
        assert "type" in c
    print(f"[OK] /chunk (txt) -> {d['chunk_count']} 块")
    for c in d["chunks"]:
        print(f"      chunk[{c['index']}] type={c['type']} len={c['length']}")


def test_chunk_with_params():
    """自定义分块参数"""
    path = os.path.join(ROOT, "test_file", "sample.txt")
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/chunk?max_chunk_size=200&min_chunk_size=50&chunk_overlap=50",
            files={"file": ("sample.txt", f, "text/plain")},
        )
    assert r.status_code == 200, r.text
    d = r.json()
    print(f"[OK] /chunk (max=200) -> {d['chunk_count']} 块")


def test_unsupported_format():
    """不支持的格式返回 400"""
    r = requests.post(
        f"{BASE}/parse",
        files={"file": ("a.xyz", b"hello", "application/octet-stream")},
    )
    assert r.status_code == 400
    print(f"[OK] 不支持格式 -> 400: {r.json()['detail'][:40]}...")


def test_parse_html():
    path = os.path.join(ROOT, "test_file", "rag_test_doc.html")
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/parse", files={"file": ("rag_test_doc.html", f, "text/html")})
    assert r.status_code == 200, r.text
    d = r.json()
    print(f"[OK] /parse (html) -> {len(d['raw_text'])} 字, table_count={d['table_count']}")


def main():
    test_health()
    test_formats()
    test_parse_txt()
    test_chunk_txt()
    test_chunk_with_params()
    test_unsupported_format()
    test_parse_html()
    print("\n所有 API 测试通过 ✅")


if __name__ == "__main__":
    main()
