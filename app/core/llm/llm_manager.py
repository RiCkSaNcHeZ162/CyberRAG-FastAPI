"""
LLM Manager - Unified interface for Groq and Ollama LLM providers.

Supports:
- Groq API (free tier) with streaming
- Ollama (local) with streaming
- Automatic retry and fallback handling
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Optional

import httpx
from groq import AsyncGroq
from huggingface_hub import AsyncInferenceClient, model_info

from app.config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM interactions with support for Groq and Ollama."""

    _instance: Optional["LLMManager"] = None

    def __new__(cls) -> "LLMManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.provider = settings.LLM_PROVIDER
        self._groq_client: AsyncGroq | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._multimodal_client: AsyncInferenceClient | None = None

        if self.provider == "groq":
            if (
                not settings.GROQ_API_KEY
                or settings.GROQ_API_KEY == "your_groq_api_key_here"
            ):
                logger.warning(
                    "Groq API key not set. Please set GROQ_API_KEY in .env file. "
                    "Get a free key at https://console.groq.com"
                )
            else:
                self._groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        elif self.provider == "ollama":
            self._http_client = httpx.AsyncClient(
                base_url=settings.OLLAMA_BASE_URL,
                timeout=120.0,
            )
        elif self.provider == "multimodal":
            self._multimodal_client = AsyncInferenceClient(
                model=settings.MULTIMODAL_MODEL, token=settings.HF_TOKEN
            )

        logger.info(f"LLM Manager initialized with provider: {self.provider}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that answers questions based on the provided context.",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        images: list[dict] | None = None,
    ) -> str:
        """Generate a complete response from the LLM."""
        if self.provider == "groq":
            return await self._groq_generate(
                prompt, system_prompt, temperature, max_tokens, images
            )
        elif self.provider == "ollama":
            return await self._ollama_generate(
                prompt, system_prompt, temperature, max_tokens, images
            )
        elif self.provider == "multimodal":
            return await self._multimodal_generate(
                prompt, system_prompt, temperature, max_tokens, images
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that answers questions based on the provided context.",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from the LLM."""
        if self.provider == "groq":
            async for token in self._groq_stream(
                prompt, system_prompt, temperature, max_tokens
            ):
                yield token
        elif self.provider == "ollama":
            async for token in self._ollama_stream(
                prompt, system_prompt, temperature, max_tokens
            ):
                yield token
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ── Groq Implementation ──────────────────────────────────────

    async def _groq_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        images: list[dict] | None,
    ) -> str:
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized. Check your GROQ_API_KEY.")
        user_content = []
        if images:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img['format']};base64,{img['data']}"
                    },
                }
                for img in images
            ]
        user_content.append({"type": "text", "text": prompt})
        for attempt in range(3):
            try:
                response = await self._groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    temperature=temperature,
                    max_completion_tokens=3000,
                    extra_body={
                        "reasoning_effort": "none",
                        "reasoning_format": "hidden",
                    },
                )
                # logger.info("Got response waiting for 10 sec to not hit the limit")
                # await asyncio.sleep(10)
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Groq attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
        return ""

    async def _groq_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized. Check your GROQ_API_KEY.")

        stream = await self._groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ── Ollama Implementation ─────────────────────────────────────

    async def _ollama_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        images: list | None = None,
    ) -> str:
        if not self._http_client:
            raise RuntimeError("Ollama HTTP client not initialized.")
        if images:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": images},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        response = await self._http_client.post(
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def _ollama_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        if not self._http_client:
            raise RuntimeError("Ollama HTTP client not initialized.")

        async with self._http_client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            import json

            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    # ── multimodal Implementation ─────────────────────────────────────

    async def _multimodal_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        images: list | None = None,
    ) -> str:
        if not self._multimodal_client:
            raise RuntimeError("HF client not initialized.")
        info = model_info(
            "Qwen/Qwen2-VL-72B-Instruct", expand="inferenceProviderMapping"
        )
        info = json.dumps(info)
        print(json)
        user_content = []
        if images:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img['format']};base64,{img['data']}"
                    },
                }
                for img in images
            ]
        user_content.append({"type": "text", "text": prompt})
        for attempt in range(3):
            try:
                response = await self._multimodal_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    max_tokens=500,
                )

                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Multimodal attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
        return ""
        # stream = await self._multimodal_client.chat.completions.create(
        #     model="openai/gpt-oss-120b",
        #     messages=[{"role": "user", "content": "Say this is a test"}],
        #     stream=True,
        # )
        # async for chunk in stream:
        #     print(chunk.choices[0].delta.content or "", end="")

    # async def _multimodal_stream(
    #     self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    # ) -> AsyncGenerator[str, None]:
    #     if not self._http_client:
    #         raise RuntimeError("Ollama HTTP client not initialized.")

    #     async with self._http_client.stream(
    #         "POST",
    #         "/api/chat",
    #         json={
    #             "model": settings.OLLAMA_MODEL,
    #             "messages": [
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             "stream": True,
    #             "options": {
    #                 "temperature": temperature,
    #                 "num_predict": max_tokens,
    #             },
    #         },
    #     ) as response:
    #         import json

    #         async for line in response.aiter_lines():
    #             if line.strip():
    #                 data = json.loads(line)
    #                 content = data.get("message", {}).get("content", "")
    #                 if content:
    #                     yield content

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
