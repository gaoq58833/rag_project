"""检索层评测脚本：对比「纯向量」与「向量+BM25 混合」两种检索模式

用法（rag-api 容器内运行）：
    docker cp 量化/eval_retrieval.py rag-api:/tmp/
    docker cp 量化/eval_dataset.json rag-api:/tmp/
    docker exec rag-api /app/.venv/bin/python /tmp/eval_retrieval.py [用户名] [数据集JSON路径]

指标（K 与 .env 的 RERANK_TOP_N 一致，当前为 3）：
    Hit Rate@K：前 K 条结果中包含正确块的问题占比
    MRR@K     ：第一个正确块排名倒数的平均值（未命中计 0）

命中判定：返回块的 filename 与 expected_filename 一致，且 content 包含 expected_substring
输出：markdown 对比表（stdout 打印 + 写入 /tmp/eval_result.md）
"""
import json
import sys
from datetime import datetime, timedelta, timezone

_CST = timezone(timedelta(hours=8))  # 容器内为 UTC，报告时间统一转东八区

sys.path.insert(0, "/app")

from src.config import settings
from src.db.repository import UserRepository
from src.rag.pipeline import retrieve


def find_rank(results, expected_filename: str, expected_substring: str):
    """正确块在结果中的排名（从 1 开始），未命中返回 None"""
    for rank, r in enumerate(results, start=1):
        if r.get("filename", "") == expected_filename and expected_substring in r.get("content", ""):
            return rank
    return None


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "alice"
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/eval_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) else data

    user = UserRepository.get_by_username(username)
    if not user:
        print(f"用户 {username} 不存在", file=sys.stderr)
        sys.exit(1)
    user_id = str(user["id"])

    k = settings.rerank_top_n
    rows = []
    stats = {"vector": {"hit": 0, "rr": 0.0}, "hybrid": {"hit": 0, "rr": 0.0}}

    for i, item in enumerate(questions, start=1):
        q = item["question"]
        exp_file = item["expected_filename"]
        exp_sub = item["expected_substring"]
        print(f"[{i}/{len(questions)}] {q}", file=sys.stderr)

        ranks = {}
        for mode, use_bm25 in (("vector", False), ("hybrid", True)):
            results = retrieve(q, user_id=user_id, use_bm25=use_bm25)
            rank = find_rank(results, exp_file, exp_sub)
            ranks[mode] = rank
            if rank is not None:
                stats[mode]["hit"] += 1
                stats[mode]["rr"] += 1.0 / rank

        rows.append((i, q, exp_file, ranks["vector"], ranks["hybrid"]))

    n = len(questions)
    hit_vec = stats["vector"]["hit"] / n if n else 0.0
    mrr_vec = stats["vector"]["rr"] / n if n else 0.0
    hit_hyb = stats["hybrid"]["hit"] / n if n else 0.0
    mrr_hyb = stats["hybrid"]["rr"] / n if n else 0.0

    def fmt(rank):
        return str(rank) if rank is not None else "未命中"

    lines = [
        f"评估时间：{datetime.now(_CST).strftime('%Y-%m-%d %H:%M')}",
        "",
        f"评测用户：{username}，样本数：{n}",
        "",
        f"运行配置：RETRIEVAL_TOP_K={settings.retrieval_top_k}、BM25_TOP_K={settings.bm25_top_k}、RERANK_TOP_N={k}（重排后返回条数，即本评测的 K）",
        "",
        "命中判定口径：返回块的 filename 与期望文档一致，且 content 包含期望子串",
        "",
        f"| 检索模式 | Hit Rate@{k} | MRR@{k} |",
        "| --- | --- | --- |",
        f"| 纯向量 | {hit_vec:.2%} | {mrr_vec:.4f} |",
        f"| 向量+BM25 混合 | {hit_hyb:.2%} | {mrr_hyb:.4f} |",
        "",
        "### 逐题明细",
        "",
        "| # | 问题 | 期望文档 | 纯向量排名 | 混合排名 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, q, exp_file, rv, rh in rows:
        lines.append(f"| {i} | {q} | {exp_file} | {fmt(rv)} | {fmt(rh)} |")
    report = "\n".join(lines)

    print(report)
    with open("/tmp/eval_result.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
