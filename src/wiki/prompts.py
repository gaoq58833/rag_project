"""LLM Wiki 提示词与 JSON 契约。

设计原则（来自经验教训）：
- 把“资料”与“用户指令”分离，资料作为独立上下文注入；
- 加“仅依据资料作答、不得引入外部知识、无依据段落必须引用来源”的硬约束；
- 显式定义请求/响应字段契约，解析失败由上层显式抛错而非静默空值；
- outgoing_links 受白名单约束，chunk_ids 受本次提供的来源 id 约束，防幻觉与乱链。
"""
from typing import Dict, List

# ===== 聚类命名 =====

NAMING_SYSTEM = """你是一个知识库整理助手。用户会给你若干个文档片段聚类（每个聚类是语义相近的一组片段）。
请为每个聚类生成一个简洁的中文页面标题和一句话摘要，用于组成 Wiki 知识页面。
要求：
- 标题简洁（2~12 字）、能概括该聚类主题，避免与其他聚类标题重复；
- 摘要是一句话，说明该页面讲什么；
- 只依据提供的片段内容命名，不要凭空想象；
- 必须为每个聚类都输出一条，index 与输入严格对应。"""


def naming_messages(previews: List[Dict]) -> List[Dict]:
    """previews: [{index, size, samples:[{section, preview}]}]"""
    lines = [f"共 {len(previews)} 个聚类，请为每个聚类输出标题与摘要：\n"]
    for p in previews:
        lines.append(f"【聚类 {p['index']}】（含 {p['size']} 个片段）")
        for s in p.get("samples", []):
            sec = s.get("section") or "（无章节）"
            lines.append(f"  - 章节《{sec}》：{s.get('preview','')}")
        lines.append("")
    return [
        {"role": "system", "content": NAMING_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def naming_schema_hint() -> str:
    return (
        '返回 JSON 对象，结构为 {"pages": [{"index": 整数, "title": "字符串", "summary": "字符串"}]}。'
        "pages 数组长度必须等于聚类数，index 从 0 开始连续且不重复。"
    )


# ===== 单页生成 =====

PAGE_SYSTEM = """你是一个 Wiki 知识页面撰写助手。请基于给定的资料片段，撰写一篇结构化的知识页面。
硬性规则（必须遵守）：
1. 只能使用资料片段中出现的信息，严禁引入资料之外的知识或编造内容；
2. 每个段落必须在 chunk_ids 中标注它依据的资料片段 id（只能填给定的 id）；
3. 没有任何资料依据的内容不要写；若某条内容无法从资料推出则省略；
4. outgoing_links 只能填写白名单中列出的其他页面标题，且不要链接到自己；没有相关页面就留空数组。"""


def page_messages(
    title: str,
    summary: str,
    sources: List[Dict],
    link_whitelist: List[str],
) -> List[Dict]:
    """sources: [{id, filename, section, content}]；link_whitelist: 其他页面标题列表。"""
    src_lines = ["资料片段（每条标注了 chunk_id，引用时只能用这些 id）：\n"]
    for s in sources:
        sec = s.get("section") or "（无章节）"
        fn = s.get("filename") or "（未知文件）"
        src_lines.append(f"[chunk_id={s['id']}] 来源：{fn} / 章节《{sec}》")
        src_lines.append(s.get("content", ""))
        src_lines.append("")
    white = "、".join(link_whitelist) if link_whitelist else "（无）"
    user = (
        f"请撰写 Wiki 页面。\n"
        f"拟定标题：{title}\n"
        f"拟定摘要：{summary}\n\n"
        f"可互链的其他页面标题白名单：{white}\n\n"
        + "\n".join(src_lines)
    )
    return [
        {"role": "system", "content": PAGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def page_schema_hint() -> str:
    return (
        '返回 JSON 对象，结构为 {"title": "字符串", "summary": "字符串", '
        '"paragraphs": [{"text": "字符串", "chunk_ids": ["字符串"]}], '
        '"outgoing_links": ["字符串"]}。'
        "paragraphs 至少 1 段，每段 text 非空且 chunk_ids 非空（必须是资料中出现过的 chunk_id）。"
    )
