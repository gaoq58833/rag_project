"""LLM Wiki 量化评测脚本（4 项核心指标）。

指标：
  1. link_validity      — 互链正确率（纯脚本）：outgoing_links 是否都指向真实存在的页面
  2. chunk_coverage     — 内容覆盖率（纯脚本）：eligible chunk 被 Wiki 页面引用的比例
  3. wiki_faithfulness  — 溯源保真率（LLM-as-Judge）：段落内容是否可由引用 chunk 推导
  4. cluster_purity     — 聚类纯度（LLM-as-Judge）：页面引用的多个 chunk 是否同主题

用法：
  python scripts/eval_wiki.py                      # 自动选 pages 最多的用户，全量评测
  python scripts/eval_wiki.py --user-id <uuid>     # 指定用户
  python scripts/eval_wiki.py --skip-llm           # 只跑纯脚本指标（#1 #4）
  python scripts/eval_wiki.py --out report.json    # 指定报告输出路径
"""
import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

# 让脚本可直接 `python scripts/eval_wiki.py` 运行（无需设 PYTHONPATH）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.db.database import db
from src.db.repository import ChunkRepository, WikiPageRepository


# ---------- 工具 ----------

def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…(截断)"


def _pick_user(user_id: str | None) -> Dict[str, Any]:
    """选 pages 最多的用户；若指定则校验存在。"""
    if user_id:
        row = db.query_one("SELECT id, username FROM users WHERE id = %s", (user_id,))
        if not row:
            raise SystemExit(f"用户不存在: {user_id}")
        return row
    rows = db.query_all(
        """SELECT u.id, u.username, COUNT(w.id) AS pages
           FROM users u LEFT JOIN wiki_pages w ON w.user_id = u.id
           GROUP BY u.id, u.username ORDER BY pages DESC LIMIT 1"""
    )
    if not rows or rows[0]["pages"] == 0:
        raise SystemExit("没有任何用户生成过 Wiki 页面，无法评测。请先生成 Wiki。")
    return rows[0]


def _load_chunk_map(user_id: str) -> Dict[str, str]:
    """该用户全部 chunk 的 id -> content（用于溯源回查，全量载入）。"""
    rows = db.query_all(
        "SELECT qdrant_point_id::text AS cid, content FROM chunks WHERE document_id IN "
        "(SELECT id FROM documents WHERE user_id = %s)",
        (user_id,),
    )
    return {r["cid"]: (r.get("content") or "") for r in rows}


# ---------- #2 链接正确率（纯脚本） ----------

def eval_link_validity(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    slug_set = {p["slug"] for p in pages}
    total = 0
    valid = 0
    invalid: List[Dict[str, str]] = []
    for p in pages:
        for link in (p.get("outgoing_links") or []):
            total += 1
            if link in slug_set:
                valid += 1
            else:
                invalid.append({"page": p["title"], "broken_slug": link})
    value = (valid / total) if total else 1.0
    return {
        "value": round(value, 4),
        "total": total,
        "valid": valid,
        "invalid": invalid,
    }


# ---------- #3 内容覆盖率（纯脚本） ----------

def eval_chunk_coverage(user_id: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    elig = ChunkRepository.get_wiki_eligible_chunks(user_id)
    elig_ids = {str(c["qdrant_point_id"]) for c in elig}
    used: set = set()
    for p in pages:
        for cid in (p.get("source_chunk_ids") or []):
            used.add(str(cid))
    covered = used & elig_ids
    uncovered = sorted(elig_ids - used)
    dangling = sorted(used - elig_ids)  # 引用了非 eligible 或不存在的 chunk
    value = (len(covered) / len(elig_ids)) if elig_ids else 1.0
    return {
        "value": round(value, 4),
        "total": len(elig_ids),
        "covered": len(covered),
        "uncovered": uncovered,
        "dangling_refs": dangling,
    }


# ---------- #1 溯源保真率（LLM-as-Judge） ----------

_FAITH_SYSTEM = """你是一个严格的事实核查员。给定一段 Wiki 文本和它声称引用的原始资料片段，判断这段文本的每一个事实断言是否都能从资料中推导出来。
规则：
- 若文本引入了资料中没有的新事实、新数据、新结论或新定义，即为不通过；
- 同义改写、归纳概括、合理合并资料中已有的信息，算通过；
- 只依据给定资料判断，禁止使用你自己的知识；
- 只输出 JSON。"""

_FAITH_SCHEMA = '{"supported": true 或 false, "reason": "一句话说明；若不通过，指出哪句无依据"}'


def _judge_faithfulness(paragraph_text: str, sources_text: str, llm_client) -> Dict[str, Any] | None:
    msgs = [
        {"role": "system", "content": _FAITH_SYSTEM},
        {"role": "user", "content": f"Wiki 段落：\n{paragraph_text}\n\n原始资料（该段落声称引用的来源）：\n{sources_text}\n\n判断：这段文本的内容是否完全可由上述资料推导？"},
    ]
    try:
        data = llm_client.chat_json(msgs, schema_hint=_FAITH_SCHEMA, temperature=0, max_tokens=400)
        return {"supported": bool(data.get("supported")), "reason": data.get("reason", "")}
    except Exception as e:
        return {"error": str(e)[:200]}


def eval_faithfulness(pages, chunk_map: Dict[str, str], llm_client) -> Dict[str, Any]:
    total = 0
    supported = 0
    unsupported: List[Dict[str, str]] = []
    judge_errors = 0
    for p in pages:
        for para in (p.get("paragraphs") or []):
            text = (para.get("text") or "").strip()
            if not text:
                continue
            refs = [str(x) for x in (para.get("chunk_ids") or [])]
            srcs = "\n---\n".join(_truncate(chunk_map.get(cid, "(来源缺失)"), 600) for cid in refs) or "(无)"
            total += 1
            res = _judge_faithfulness(_truncate(text, 800), _truncate(srcs, 1800), llm_client)
            if res is None or "error" in res:
                judge_errors += 1
                continue
            if res["supported"]:
                supported += 1
            else:
                unsupported.append({"page": p["title"], "paragraph": _truncate(text, 80), "reason": res.get("reason", "")})
    scored = total - judge_errors
    value = (supported / scored) if scored else 0.0
    return {
        "value": round(value, 4),
        "total": total,
        "scored": scored,
        "supported": supported,
        "unsupported": unsupported,
        "judge_errors": judge_errors,
    }


# ---------- #4 聚类纯度（LLM-as-Judge） ----------

_PURITY_SYSTEM = """你是一个知识库质量审核员。给定一个 Wiki 页面的标题和它引用的所有原始资料片段，判断这些资料是否属于同一个主题，页面主题是否单一、没有把不相关的内容混杂在一起。
规则：
- 若所有资料都围绕同一主题展开，算通过（pure=true）；
- 若资料明显涉及两个及以上互不相关的主题，算不通过；
- 只依据给定资料判断，禁止使用你自己的知识；
- 只输出 JSON。"""

_PURITY_SCHEMA = '{"pure": true 或 false, "reason": "一句话说明"}'


def _judge_purity(title: str, sources_text: str, llm_client) -> Dict[str, Any] | None:
    msgs = [
        {"role": "system", "content": _PURITY_SYSTEM},
        {"role": "user", "content": f"页面标题：{title}\n\n引用的原始资料片段：\n{sources_text}\n\n判断：这些资料是否属于同一主题，页面是否主题单一无混杂？"},
    ]
    try:
        data = llm_client.chat_json(msgs, schema_hint=_PURITY_SCHEMA, temperature=0, max_tokens=400)
        return {"pure": bool(data.get("pure")), "reason": data.get("reason", "")}
    except Exception as e:
        return {"error": str(e)[:200]}


def eval_cluster_purity(pages, chunk_map: Dict[str, str], llm_client) -> Dict[str, Any]:
    total = 0
    pure = 0
    impure: List[Dict[str, str]] = []
    judge_errors = 0
    for p in pages:
        src_ids = [str(x) for x in (p.get("source_chunk_ids") or [])]
        srcs = "\n---\n".join(_truncate(chunk_map.get(cid, "(来源缺失)"), 500) for cid in src_ids) or "(无)"
        total += 1
        res = _judge_purity(p["title"], _truncate(srcs, 2200), llm_client)
        if res is None or "error" in res:
            judge_errors += 1
            continue
        if res["pure"]:
            pure += 1
        else:
            impure.append({"page": p["title"], "reason": res.get("reason", "")})
    scored = total - judge_errors
    value = (pure / scored) if scored else 0.0
    return {
        "value": round(value, 4),
        "total": total,
        "scored": scored,
        "pure": pure,
        "impure": impure,
        "judge_errors": judge_errors,
    }


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="LLM Wiki 量化评测（4 项核心指标）")
    ap.add_argument("--user-id", help="指定用户 UUID（默认自动选 pages 最多的用户）")
    ap.add_argument("--skip-llm", action="store_true", help="只跑纯脚本指标，不调 LLM")
    ap.add_argument("--out", default="wiki_eval_report.json", help="报告输出路径")
    args = ap.parse_args()

    user = _pick_user(args.user_id)
    user_id = user["id"]
    print(f"评测用户：{user['username']}（{user_id}）")

    pages = WikiPageRepository.get_all_by_user(user_id)
    print(f"Wiki 页面数：{len(pages)}")
    if not pages:
        raise SystemExit("该用户没有 Wiki 页面，无法评测。")

    report: Dict[str, Any] = {
        "user": user["username"],
        "user_id": user_id,
        "page_count": len(pages),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {},
    }

    # 纯脚本指标
    report["metrics"]["link_validity"] = eval_link_validity(pages)
    report["metrics"]["chunk_coverage"] = eval_chunk_coverage(user_id, pages)

    # LLM 判分指标
    if not args.skip_llm:
        from src.llm.client import llm_client
        chunk_map = _load_chunk_map(user_id)
        print(f"载入 chunk 原文：{len(chunk_map)} 条，开始 LLM 判分（#1 溯源保真 / #4 聚类纯度）…")
        t0 = time.time()
        report["metrics"]["wiki_faithfulness"] = eval_faithfulness(pages, chunk_map, llm_client)
        report["metrics"]["cluster_purity"] = eval_cluster_purity(pages, chunk_map, llm_client)
        report["llm_judge_seconds"] = round(time.time() - t0, 1)
    else:
        print("--skip-llm：跳过 #1 #4")

    # 写报告
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端摘要
    print("\n" + "=" * 60)
    print("LLM Wiki 评测结果")
    print("=" * 60)
    m = report["metrics"]
    print(f"#2 链接正确率 link_validity      : {m['link_validity']['value']:.4f}  "
          f"({m['link_validity']['valid']}/{m['link_validity']['total']})")
    print(f"#3 内容覆盖率 chunk_coverage     : {m['chunk_coverage']['value']:.4f}  "
          f"({m['chunk_coverage']['covered']}/{m['chunk_coverage']['total']})")
    if "wiki_faithfulness" in m:
        f = m["wiki_faithfulness"]
        print(f"#1 溯源保真率 wiki_faithfulness  : {f['value']:.4f}  "
              f"({f['supported']}/{f['scored']}，判分错误 {f['judge_errors']})")
    if "cluster_purity" in m:
        c = m["cluster_purity"]
        print(f"#4 聚类纯度   cluster_purity     : {c['value']:.4f}  "
              f"({c['pure']}/{c['scored']}，判分错误 {c['judge_errors']})")
    print("=" * 60)
    print(f"报告已写入：{out_path.resolve()}")

    db.close()


if __name__ == "__main__":
    main()
