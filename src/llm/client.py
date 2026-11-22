"""LLM 客户端：embedding / chat / vision / rerank，统一 OpenAI 兼容格式"""
import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

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
        return self._chat_completion(messages, temperature, max_tokens)

    def _chat_completion(
        self, messages: List[dict], temperature: float, max_tokens: int, **extra: Any
    ) -> str:
        """底层 chat 补全调用（主模型失败自动降级备用模型重试一次），返回文本内容。
        extra 可传 response_format 等额外参数。"""
        kwargs: Dict[str, Any] = dict(messages=messages, temperature=temperature, max_tokens=max_tokens)
        kwargs.update(extra)
        try:
            resp = self.chat_client.chat.completions.create(model=settings.llm_model, **kwargs)
        except Exception as e:
            fb = settings.llm_fallback_model
            if not fb or fb == settings.llm_model:
                raise
            logger.warning(f"LLM 主模型 {settings.llm_model} 失败（{e}），降级到 {fb}")
            resp = self.chat_client.chat.completions.create(model=fb, **kwargs)
        return resp.choices[0].message.content or ""

    @staticmethod
    def _prepend_system(messages: List[dict], instruction: str) -> List[dict]:
        """在不修改入参的前提下，把 JSON 输出指令并入 system 消息（已有则追加，没有则新建）。"""
        if not messages:
            return [{"role": "system", "content": instruction}]
        if messages[0].get("role") == "system":
            merged = [dict(messages[0])]
            merged[0]["content"] = (merged[0].get("content") or "") + "\n\n" + instruction
            return merged + [dict(m) for m in messages[1:]]
        return [{"role": "system", "content": instruction}] + [dict(m) for m in messages]

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """去掉模型可能误输出的 ```json / ``` 围栏，只保留里面的 JSON 文本。"""
        s = text.strip()
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
        if m:
            return m.group(1).strip()
        return s

    def chat_json(
        self,
        messages: List[dict],
        schema_hint: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """要求模型只输出一个合法 JSON 对象，返回解析后的 dict。

        - schema_hint：JSON 结构说明（字段、类型、约束），会作为 system 指令注入，帮助模型对齐契约。
        - 优先用 response_format={"type":"json_object"} 强约束；若网关不支持该参数，自动退化为纯 prompt 约束重试。
        - 失败显式抛 ValueError（不返回空 dict）：JSON 解析失败 / 顶层不是 object 时重试 retries 次，仍失败则抛错，
          让上层能决定降级或提示，避免静默拿到脏数据。
        """
        temperature = settings.llm_temperature if temperature is None else temperature
        max_tokens = settings.llm_max_tokens if max_tokens is None else max_tokens

        instruction = "你必须只输出一个合法的 JSON 对象（object），不要使用 markdown 代码块，不要输出任何解释文字、前后缀或注释。"
        if schema_hint:
            instruction += "\nJSON 结构要求：\n" + schema_hint
        msgs = self._prepend_system(messages, instruction)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            raw: Optional[str] = None
            try:
                raw = self._chat_completion(
                    msgs, temperature, max_tokens, response_format={"type": "json_object"}
                )
            except Exception as e:
                # 网关可能不支持 response_format 参数：首次失败时退化为纯 prompt 约束再试一次
                low = str(e).lower()
                if attempt == 0 and ("response_format" in low or "parameter" in low or "不支持" in str(e)):
                    logger.warning(f"chat_json response_format 不被支持（{e}），退化为 prompt 约束")
                    try:
                        raw = self._chat_completion(msgs, temperature, max_tokens)
                    except Exception as e2:
                        last_err = e2
                        continue
                else:
                    last_err = e
                    continue

            raw = self._strip_code_fence(raw or "")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning(f"chat_json 解析失败（第 {attempt + 1} 次）: {e}; 片段={raw[:160]!r}")
                continue
            if not isinstance(data, dict):
                last_err = ValueError(f"期望 JSON object，得到 {type(data).__name__}")
                logger.warning(f"chat_json 顶层非 object（第 {attempt + 1} 次）: {last_err}")
                continue
            return data

        raise ValueError(f"chat_json 重试 {retries + 1} 次后仍未得到合法 JSON: {last_err}")

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
