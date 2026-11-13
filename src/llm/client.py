"""LLM 客户端：embedding / chat / vision / rerank，统一 OpenAI 兼容格式"""
import base64
import logging
from typing import List, Optional

import httpx
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """封装三个模型调用，全部使用 OpenAI 兼容接口"""

    def __init__(self):
        self._embedding_client: Optional[OpenAI] = None
        self._chat_client: Optional[OpenAI] = None
        self._vision_client: Optional[OpenAI] = None

    # ---------- Embedding ----------
    @property
    def embedding_client(self) -> OpenAI:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
        return self._embedding_client

    def embed(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """批量 embedding，返回向量列表。主模型失败自动降级备用模型重试一次"""
        try:
            return self._embed_with_model(settings.embedding_model, texts, batch_size)
        except Exception as e:
            if not settings.embedding_fallback_model or settings.embedding_fallback_model == settings.embedding_model:
                raise
            logger.warning(f"embedding 主模型 {settings.embedding_model} 失败（{e}），降级到 {settings.embedding_fallback_model}")
            return self._embed_with_model(settings.embedding_fallback_model, texts, batch_size)

    def _embed_with_model(self, model: str, texts: List[str], batch_size: int) -> List[List[float]]:
        """多数 embedding API 限制单批条数（如 DashScope 限 20），超过时自动分批调用再合并结果"""
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.embedding_client.embeddings.create(
                model=model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in resp.data)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单个 query embedding"""
        return self.embed([text])[0]

    # ---------- 多模态 Embedding（以图搜图，DashScope 原生 API） ----------
    def _embed_mm(self, contents: List[dict]) -> List[float]:
        """调 DashScope multimodal-embedding，返回单条向量

        contents 示例：[{"image": "data:image/jpeg;base64,..."}] 或 [{"text": "..."}]
        图片与文本向量落在同一向量空间，可跨模态检索。
        """
        url = settings.mm_embedding_base_url.rstrip("/") + "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        headers = {
            "Authorization": f"Bearer {settings.mm_embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.mm_embedding_model,
            "input": {"contents": contents},
        }
        with httpx.Client(timeout=60) as http:
            resp = http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            embeddings = resp.json()["output"]["embeddings"]
        return embeddings[0]["embedding"]

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> List[float]:
        """图片向量（以图搜图入库/查询用）"""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return self._embed_mm([{"image": f"data:{mime_type};base64,{b64}"}])

    def embed_mm_text(self, text: str) -> List[float]:
        """文本向量（与图片同空间，跨模态文搜图用）"""
        return self._embed_mm([{"text": text}])

    # ---------- Chat (LLM) ----------
    @property
    def chat_client(self) -> OpenAI:
        if self._chat_client is None:
            self._chat_client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return self._chat_client

    def chat(self, messages: List[dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """对话补全，返回模型输出文本。主模型失败自动降级备用模型重试一次"""
        temperature = settings.llm_temperature if temperature is None else temperature
        max_tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
        kwargs = dict(messages=messages, temperature=temperature, max_tokens=max_tokens)
        try:
            resp = self.chat_client.chat.completions.create(model=settings.llm_model, **kwargs)
        except Exception as e:
            if not settings.llm_fallback_model or settings.llm_fallback_model == settings.llm_model:
                raise
            logger.warning(f"LLM 主模型 {settings.llm_model} 失败（{e}），降级到 {settings.llm_fallback_model}")
            resp = self.chat_client.chat.completions.create(model=settings.llm_fallback_model, **kwargs)
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """流式对话补全，逐 token yield 文本片段。

        降级策略（只降级一次）：
        - 建立流失败：同一 messages 原样发给备用模型重新生成；
        - 流中途断连：已输出的半截答案作为 assistant 消息 + 续写指令追加进上下文，
          备用模型从断点接着写，前端无感知，不会重复输出；
        - 备用模型再失败：异常直接抛出。
        """
        temperature = settings.llm_temperature if temperature is None else temperature
        max_tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
        model = settings.llm_model
        partial = ""      # 已输出的内容，断连时用于续写
        resumed = False   # 是否已降级过
        while True:
            try:
                stream = self.chat_client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        partial += delta
                        yield delta
                return  # 正常结束
            except Exception as e:
                fb = settings.llm_fallback_model
                if resumed or not fb or fb == model:
                    raise
                resumed = True
                model = fb
                if partial:
                    logger.warning(f"LLM 主模型流式输出中断（{e}），已输出 {len(partial)} 字，降级到 {fb} 续写")
                    messages = messages + [
                        {"role": "assistant", "content": partial},
                        {"role": "user", "content": "你的上一段回答因网络中断被截断。请从截断处继续输出剩余内容，直接续写，不要重复已输出的部分，不要任何解释或道歉。"},
                    ]
                else:
                    logger.warning(f"LLM 主模型 {settings.llm_model} 建立流失败（{e}），降级到 {fb} 重新生成")

    # ---------- Vision（图片理解） ----------
    @property
    def vision_client(self) -> OpenAI:
        if self._vision_client is None:
            self._vision_client = OpenAI(
                api_key=settings.vision_api_key,
                base_url=settings.vision_base_url,
            )
        return self._vision_client

    def chat_vision(self, prompt: str, image_bytes: bytes, mime_type: str = "image/jpeg",
                    temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """多模态对话：prompt + 一张图片（base64 内联），返回模型输出文本。
        主模型失败自动降级备用模型重试一次"""
        temperature = settings.vision_temperature if temperature is None else temperature
        max_tokens = settings.vision_max_tokens if max_tokens is None else max_tokens
        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }]
        kwargs = dict(messages=messages, temperature=temperature, max_tokens=max_tokens)
        try:
            resp = self.vision_client.chat.completions.create(model=settings.vision_model, **kwargs)
        except Exception as e:
            if not settings.vision_fallback_model or settings.vision_fallback_model == settings.vision_model:
                raise
            logger.warning(f"vision 主模型 {settings.vision_model} 失败（{e}），降级到 {settings.vision_fallback_model}")
            resp = self.vision_client.chat.completions.create(model=settings.vision_fallback_model, **kwargs)
        return resp.choices[0].message.content or ""

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        """重排，返回排序后的文档索引列表（分数从高到低）。主模型失败自动降级备用模型重试一次"""
        top_n = top_n or settings.rerank_top_n
        try:
            return self._rerank_with_model(settings.reranker_model, query, documents, top_n)
        except Exception as e:
            if not settings.reranker_fallback_model or settings.reranker_fallback_model == settings.reranker_model:
                raise
            logger.warning(f"rerank 主模型 {settings.reranker_model} 失败（{e}），降级到 {settings.reranker_fallback_model}")
            return self._rerank_with_model(settings.reranker_fallback_model, query, documents, top_n)

    def _rerank_with_model(self, model: str, query: str, documents: List[str], top_n: int) -> List[int]:
        """使用 DashScope 原生 rerank API（/services/rerank/text-rerank/text-rerank）"""
        url = settings.reranker_base_url.rstrip("/") + "/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {settings.reranker_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
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
