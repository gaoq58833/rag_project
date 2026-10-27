"""扩充检索层评测集：基于真实分块内容调用 LLM 生成评测题（容器内运行）

用法：
    docker cp 量化/generate_eval_dataset.py rag-api:/tmp/
    docker exec rag-api /app/.venv/bin/python /tmp/generate_eval_dataset.py [用户名] [目标题数] [输出路径]

生成逻辑：
1. 从 PG 读取指定用户的所有分块（content）
2. 对每个分块调用 LLM 生成若干道题，每道题包含：
   - question：问题文本
   - expected_filename：分块所属文档
   - expected_substring：能从该分块 content 中截取到的答案子串
3. 校验：expected_substring 必须真实出现在该分块 content 中（避免 LLM 幻觉出不存在的内容）
4. 去重（question 文本相同只保留一条），收集满目标题数后输出 JSON

输出格式与 eval_dataset.json 一致，可直接喂给 eval_retrieval.py。
"""
import json
import sys

sys.path.insert(0, "/app")

from src.db import db
from src.llm import llm_client

TARGET_DEFAULT = 100
USERNAME_DEFAULT = "alice"
OUTPUT_DEFAULT = "/tmp/eval_dataset_100.json"

PROMPT = """你是 RAG 检索评测集出题专家。下面给你一段知识库分块文本，请基于它的真实内容出几道问答题。

要求：
1. 问题必须能仅根据这段文本回答，不能引入文本外的知识。
2. 每道题给出一个「答案子串」expected_substring：从原文中逐字复制一段能回答该问题的连续文字（5~40字），不要改写。
3. 题目要多样化，覆盖两类：
   (a) 精确查询类：问编号、参数、数字、专名（如"ε是什么""编号40的项目名"），这类靠关键词匹配。
   (b) 语义查询类：问概念、原理、步骤、作用，可用口语化或同义改写表达（如"怎么让模型分类出错"），这类靠语义理解。
4. 不要出答案子串完全相同的重复题。
5. 只输出 JSON 数组，不要任何解释文字，格式：
   [{"question": "...", "expected_substring": "..."}, ...]

分块文本：
\"\"\"
{content}
\"\"\"
"""


def fetch_chunks(username: str):
    """读取该用户所有分块的完整 content"""
    sql = """
        SELECT c.id, c.content, d.filename
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        JOIN users u ON d.user_id = u.id
        WHERE u.username = %s
        ORDER BY d.filename, c.chunk_index
    """
    return db.query_all(sql, (username,))


def generate_for_chunk(content: str, count: int = 4) -> list:
    """对单个分块生成 count 道题，返回 list[dict]（已过滤无效项）"""
    # 分块太长会撑爆 prompt，截断到合理范围（保留首尾，答案子串校验时会过滤掉不在原文里的）
    text = content
    if len(text) > 1800:
        text = text[:1500] + "\n...（中间截断）...\n" + text[-300:]

    prompt = PROMPT.replace("{content}", text)
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = llm_client.chat(messages, temperature=0.3, max_tokens=1200)
    except Exception as e:
        print(f"  LLM 调用失败: {e}", file=sys.stderr)
        return []

    # 解析 JSON（容错：去掉代码围栏、截取数组部分）
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []

    valid = []
    for it in items:
        q = it.get("question", "").strip()
        sub = it.get("expected_substring", "").strip()
        if not q or not sub:
            continue
        if len(sub) > 60:  # 答案子串过长，检索命中判定会过松
            continue
        if sub not in content:  # 关键校验：子串必须真实存在于原文
            continue
        valid.append({"question": q, "expected_substring": sub})
    return valid[:count]


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else USERNAME_DEFAULT
    target = int(sys.argv[2]) if len(sys.argv) > 2 else TARGET_DEFAULT
    output = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DEFAULT

    chunks = fetch_chunks(username)
    print(f"用户 {username} 共 {len(chunks)} 个分块，目标生成 {target} 题", file=sys.stderr)

    seen_questions = set()
    results = []

    def save():
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"questions": results}, f, ensure_ascii=False, indent=2)

    # 每轮给每个分块生成 4 道，多轮迭代直到凑够 target
    max_rounds = 8
    for r in range(max_rounds):
        if len(results) >= target:
            break
        print(f"第 {r + 1}/{max_rounds} 轮生成...", file=sys.stderr)
        for idx, ch in enumerate(chunks):
            if len(results) >= target:
                break
            content = ch["content"] or ""
            if len(content) < 20:
                continue
            items = generate_for_chunk(content, count=4)
            for it in items:
                if it["question"] in seen_questions:
                    continue
                seen_questions.add(it["question"])
                results.append({
                    "question": it["question"],
                    "expected_filename": ch["filename"],
                    "expected_substring": it["expected_substring"],
                })
                if len(results) >= target:
                    break
            save()  # 增量保存，避免中途丢失
            if (idx + 1) % 5 == 0:
                print(f"  已生成 {len(results)}/{target}", file=sys.stderr)

    print(f"最终生成 {len(results)} 题", file=sys.stderr)


if __name__ == "__main__":
    main()
