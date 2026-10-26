"""LLM 客户端：embedding / chat / rerank，统一 OpenAI 兼容格式"""
from typing import List, Optional

import httpx
from openai import OpenAI

from src.config import settings


class LLMClient:
    """封装三个模型调用，全部使用 OpenAI 兼容接口"""

    def __init__(self):
        self._embedding_client: Optional[OpenAI] = None
        self._chat_client: Optional[OpenAI] = None

    # ---------- Embedding ----------
    @property
    def embedding_client(self) -> OpenAI:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
        return self._embedding_client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding，返回向量列表"""
        resp = self.embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """单个 query embedding"""
        return self.embed([text])[0]

    # ---------- Chat (LLM) ----------
    @property
    def chat_client(self) -> OpenAI:
        if self._chat_client is None:
            self._chat_client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return self._chat_client

    def chat(self, messages: List[dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """对话补全，返回模型输出文本"""
        resp = self.chat_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[dict], temperature: float = 0.2, max_tokens: int = 2048):
        """流式对话补全，逐 token yield 文本片段"""
        stream = self.chat_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        """重排，返回排序后的文档索引列表（分数从高到低）

        使用 DashScope 原生 rerank API（/services/rerank/text-rerank/text-rerank）。
        """
        top_n = top_n or settings.rerank_top_n
        url = settings.reranker_base_url.rstrip("/") + "/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {settings.reranker_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.reranker_model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }
        with httpx.Client(timeout=30) as http:
            resp = http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
        return [r["index"] for r in results]


# 全局单例
llm_client = LLMClient()
