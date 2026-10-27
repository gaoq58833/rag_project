"""生成层评测：使用 RAGAS 框架评测 RAG 问答质量（容器内运行）

评测指标（RAGAS 官方实现，以 LLM 为裁判）：
- faithfulness      ：回答是否基于检索上下文（不编造、不引入上下文外信息）
- answer_relevancy  ：回答是否切题、对问题有帮助
- context_precision ：检索到的上下文中有多少是相关的
- context_recall    ：检索上下文是否覆盖了参考答案所需信息

用法：
    docker cp 量化/eval_generation_ragas.py rag-api:/tmp/
    docker exec rag-api /app/.venv/bin/python /tmp/eval_generation_ragas.py [用户名] [评测集] [题数]

说明：
- 评测集复用检索层的 eval_dataset，ground_truth 取 expected_substring
- RAGAS 以项目的 LLM 网关为裁判，每题每个指标约 1 次 LLM 调用，100 题较慢，
  建议先用「题数」参数跑 20~30 题验证，再决定是否全量
"""
import json
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/app")

# ---- 兼容 shim：ragas 导入链硬依赖 VertexAI，但本项目用 OpenAI 兼容网关，无需 VertexAI ----
import types as _types
_vcm = _types.ModuleType("langchain_community.chat_models.vertexai")
_vcm.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vcm
import langchain_community.llms as _llms_mod
if not hasattr(_llms_mod, "VertexAI"):
    _llms_mod.VertexAI = type("VertexAI", (), {})
# -------------------------------------------------------------------------------------

from src.config import settings
from src.db.repository import UserRepository
from src.rag.pipeline import _chat_generate

_CST = timezone(timedelta(hours=8))


def collect_rag_outputs(username, questions):
    """跑 RAG 管线，收集 question/answer/contexts/ground_truth"""
    user = UserRepository.get_by_username(username)
    user_id = str(user["id"])

    rows = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        # 空 history 避免多轮上下文干扰生成评测
        result = _chat_generate(q, user_id=user_id, history=[], use_rerank=True)
        rows.append({
            "question": q,
            "answer": result["answer"],
            "contexts": [s["content"] for s in result["sources"]],
            "ground_truth": item["expected_substring"],
        })
        print(f"[RAG {i}/{len(questions)}] {q[:30]}...", file=sys.stderr)
    return rows


def run_ragas(rows):
    """用 RAGAS 评测，返回 {metric: score}"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness
    from langchain_openai import ChatOpenAI

    ds = Dataset.from_list(rows)

    # RAGAS 裁判模型复用项目的 OpenAI 兼容网关
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )

    # 说明：answer_relevancy 依赖 embeddings 接口，DashScope 的文本嵌入端点与
    # langchain OpenAIEmbeddings 的请求格式不兼容（400 InvalidParameter），
    # 故暂不纳入；faithfulness / context_precision / context_recall 已能覆盖
    # 「不编造」「检索相关率」「检索覆盖率」三个核心维度。
    score = evaluate(
        dataset=ds,
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
    )
    return score


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "alice"
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/eval_dataset_100.json"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = 全部

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) else data
    if limit > 0:
        questions = questions[:limit]

    print(f"用户 {username}，评测 {len(questions)} 题", file=sys.stderr)

    # 1. 跑 RAG 管线
    rows = collect_rag_outputs(username, questions)

    # 2. RAGAS 评测
    print("开始 RAGAS 评测（LLM 裁判，较慢）...", file=sys.stderr)
    score = run_ragas(rows)

    df = score.to_pandas()
    # ragas 会把输入列重命名：question→user_input, contexts→retrieved_contexts,
    # answer→response, ground_truth→reference
    metrics = ["faithfulness", "context_precision", "context_recall"]
    means = {m: float(df[m].mean()) for m in metrics if m in df.columns}

    lines = [
        f"评估时间：{datetime.now(_CST).strftime('%Y-%m-%d %H:%M')}",
        "",
        f"评测用户：{username}，样本数：{len(questions)}",
        "",
        "评测框架：RAGAS（LLM-as-judge）",
        "",
        "| 指标 | 分数 | 含义 |",
        "| --- | --- | --- |",
        f"| faithfulness | {means.get('faithfulness', 0):.4f} | 回答是否基于检索上下文（不编造） |",
        f"| context_precision | {means.get('context_precision', 0):.4f} | 检索上下文相关率 |",
        f"| context_recall | {means.get('context_recall', 0):.4f} | 检索上下文覆盖率 |",
        "",
        "### 逐题明细",
        "",
        "| # | 问题 | faithfulness | context_precision | context_recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, row in df.iterrows():
        q = str(row.get("user_input", ""))[:40]
        lines.append(
            f"| {i + 1} | {q} | "
            f"{row.get('faithfulness', '')} | "
            f"{row.get('context_precision', '')} | "
            f"{row.get('context_recall', '')} |"
        )
    report = "\n".join(lines)
    print(report)

    with open("/tmp/eval_generation_result.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 同时保存每题原始数据（含 answer/contexts）以便复核
    with open("/tmp/eval_generation_raw.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
