"""LLM 客户端：embedding / chat / rerank，统一 OpenAI 兼容格式"""
from typing import List, Optional

from openai import OpenAI

from src.config import settings


class LLMClient:
    """封装三个模型调用，全部使用 OpenAI 兼容接口"""

    def __init__(self):
        self._embedding_client: Optional[OpenAI] = None
        self._chat_client: Optional[OpenAI] = None
        self._rerank_client: Optional[OpenAI] = None

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

    def chat(self, messages: List[dict], temperature: float = 0.7) -> str:
        """对话补全，返回模型输出文本"""
        resp = self.chat_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    # ---------- Rerank ----------
    @property
    def rerank_client(self) -> OpenAI:
        if self._rerank_client is None:
            self._rerank_client = OpenAI(
                api_key=settings.reranker_api_key,
                base_url=settings.reranker_base_url,
            )
        return self._rerank_client

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        """重排，返回排序后的文档索引列表（分数从高到低）

        部分供应商的 rerank 接口兼容 OpenAI 格式，通过 /rerank 端点调用。
        """
        top_n = top_n or settings.rerank_top_n
        resp = self.rerank_client.post(
            "/rerank",
            body={
                "model": settings.reranker_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        results = resp.parse()["results"]
        return [r["index"] for r in results]


# 全局单例
llm_client = LLMClient()
