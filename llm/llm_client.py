import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any


class LLMClient:
    """
    Resilient Multi-Model LLM Client with sticky Gemini Flash rotation,
    automatic 404 endpoint blacklisting, and exponential backoff on 429/503 limits.
    """

    DEFAULT_GEMINI_POOL = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    ]

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 180,
        **kwargs: Any,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
        self.model = model or os.environ.get("LLM_MODEL", "gemini-3.6-flash")
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Build sticky rotation pool with requested model prioritized first
        self.model_pool: List[str] = []
        if self.model:
            self.model_pool.append(self.model)
        for m in self.DEFAULT_GEMINI_POOL:
            if m not in self.model_pool:
                self.model_pool.append(m)

        self.blacklisted_models: set = set()
        self.active_model_idx: int = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.provider == "gemini":
            return self._call_gemini(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
            )
        elif self.provider in ("openai", "openai_compatible"):
            return self._call_openai_compatible(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
            )
        elif self.provider == "ollama":
            return self._call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
            )
        else:
            return self._call_gemini(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
            )

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

        available = [m for m in self.model_pool if m not in self.blacklisted_models]
        if not available:
            # Reset blacklist if all models were exhausted
            self.blacklisted_models.clear()
            available = list(self.model_pool)

        max_cycles = 3
        last_error: Optional[Exception] = None

        for cycle in range(max_cycles):
            for offset in range(len(available)):
                idx = (self.active_model_idx + offset) % len(available)
                candidate_model = available[idx]
                if candidate_model in self.blacklisted_models:
                    continue

                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{candidate_model}:generateContent?key={self.api_key}"
                )

                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )
                payload: Dict[str, Any] = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    },
                }

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                        candidates = body.get("candidates", [])
                        if candidates:
                            parts = (
                                candidates[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                            text = "".join(p.get("text", "") for p in parts)
                            if text:
                                # Lock sticky pointer onto working model
                                self.active_model_idx = idx
                                return text
                except urllib.error.HTTPError as e:
                    last_error = e
                    if e.code == 404:
                        # Permanently blacklist unavailable model endpoint
                        self.blacklisted_models.add(candidate_model)
                        continue
                    elif e.code in (429, 500, 502, 503, 504):
                        # Rate limit or transient overload: rotate immediately
                        time.sleep(min(2 ** cycle, 8))
                        continue
                    else:
                        continue
                except Exception as e:
                    last_error = e
                    time.sleep(1.5)
                    continue

            # Backoff between full pool sweeps
            time.sleep(min(3 * (2 ** cycle), 15))

        raise RuntimeError(
            f"All Gemini models in rotation failed after {max_cycles} sweeps. "
            f"Last error: {last_error}"
        )

    def _call_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        url = self.base_url or "http://localhost:11434/api/generate"
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")
