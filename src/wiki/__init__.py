"""LLM Wiki：从已入库文档自动生成可浏览、可溯源、带互链的知识页面。

- prompts.py：LLM 提示词与 JSON 契约（命名 / 页面生成）
- generator.py：生成内核（聚类 → 命名 → 生成 → 溯源校验 → 互链白名单）
- service.py：编排（异步任务、写库、建索引）—— P3
"""
from src.wiki.generator import WikiPageDraft, generate_wiki

__all__ = ["WikiPageDraft", "generate_wiki"]
