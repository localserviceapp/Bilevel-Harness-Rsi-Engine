import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str


def extract_python_code(text: str) -> str:
    """Extracts executable Python source code from markdown fences or raw LLM output."""
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip() + "\n"
    return text.strip() + "\n"


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extracts a JSON object from markdown fences or raw text."""
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced[-1].strip() if fenced else text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    return json.loads(candidate)


class LLMClient:
    """
    Unified Multi-Provider LLM Client for the 4-Loop RSI Engine v4.
    Supports:
      - 'gemini': Google AI Studio REST API (GEMINI_API_KEY)
      - 'openai_compatible': DeepSeek / DeepInfra / OpenRouter / OpenAI (OPENAI_API_KEY, OPENAI_BASE_URL)
      - 'ollama': Local Ollama server (OLLAMA_BASE_URL, default http://localhost:11434)
      - 'mock': Deterministic offline discovery simulator
    """

    def __init__(
        self,
        provider: str = "auto",
        loop1_model: str = "gemini-2.5-flash",
        loop2_model: str = "gemini-2.5-pro",
        temperature: float = 0.2,
        timeout_sec: float = 30.0,
    ):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        if provider == "auto":
            if self.gemini_api_key:
                self.provider = "gemini"
            elif self.openai_api_key:
                self.provider = "openai_compatible"
            else:
                self.provider = "mock"
        else:
            self.provider = provider

        self.loop1_model = loop1_model
        self.loop2_model = loop2_model
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    @property
    def is_live(self) -> bool:
        return self.provider in {"gemini", "openai_compatible", "ollama"}

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        target_model = model or self.loop1_model
        if self.provider == "gemini":
            return self._call_gemini(prompt, system_instruction, target_model, json_mode)
        if self.provider == "openai_compatible":
            return self._call_openai_compatible(prompt, system_instruction, target_model, json_mode)
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_instruction, target_model, json_mode)
        return self._call_mock(prompt, system_instruction, target_model)

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        json_mode: bool,
    ) -> LLMResponse:
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        prompt_tok = int(usage.get("promptTokenCount", max(1, len(prompt) // 4)))
        comp_tok = int(usage.get("candidatesTokenCount", max(1, len(text) // 4)))
        total_tok = int(usage.get("totalTokenCount", prompt_tok + comp_tok))

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=total_tok,
            model=model,
            provider="gemini",
        )

    def _call_openai_compatible(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        json_mode: bool,
    ) -> LLMResponse:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        url = f"{self.openai_base_url}/chat/completions"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tok = int(usage.get("prompt_tokens", max(1, len(prompt) // 4)))
        comp_tok = int(usage.get("completion_tokens", max(1, len(text) // 4)))
        total_tok = int(usage.get("total_tokens", prompt_tok + comp_tok))

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=total_tok,
            model=model,
            provider="openai_compatible",
        )

    def _call_ollama(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        json_mode: bool,
    ) -> LLMResponse:
        url = f"{self.ollama_base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system_instruction:
            payload["system"] = system_instruction
        if json_mode:
            payload["format"] = "json"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data.get("response", "")
        prompt_tok = int(data.get("prompt_eval_count", max(1, len(prompt) // 4)))
        comp_tok = int(data.get("eval_count", max(1, len(text) // 4)))

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            model=model,
            provider="ollama",
        )

    def _call_mock(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
    ) -> LLMResponse:
        prompt_tok = max(10, len(prompt) // 4)
        return LLMResponse(
            text="MOCK_RESPONSE",
            prompt_tokens=prompt_tok,
            completion_tokens=45,
            total_tokens=prompt_tok + 45,
            model=model,
            provider="mock",
        )
