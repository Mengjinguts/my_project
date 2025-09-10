# unified_llm.py
import os, re, json, time, random
from typing import List, Dict, Optional, Literal

# 可选：按需导入，避免没装时报错
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    from anthropic import Anthropic, RateLimitError, APIStatusError
except Exception:
    Anthropic = RateLimitError = APIStatusError = None
try:
    from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
except Exception:
    AutoTokenizer = pipeline = None


Provider = Literal["openai", "anthropic", "hf"]

# -------------------- 工具：指数退避 --------------------
def _backoff_sleep(attempt:int, base:float=0.8, cap:float=20.0):
    time.sleep(min(base * (2 ** attempt) + random.uniform(0, 0.5), cap))

# -------------------- 统一 LLM 类 --------------------
class LLM:
    def __init__(self, provider: Provider, model: str,
                 temperature: float = 0.1, max_new_tokens: int = 256,
                 stop: Optional[List[str]] = None,
                 retries: int = 4):
        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.stop = stop or ["```", "\n\n", "\nUser:", "\nSystem:", "\nAssistant:"]
        self.retries = retries

        # 下面三个在对应 provider 初始化时填充
        self._openai_client = None
        self._anthropic_client = None
        self._hf_tokenizer = None
        self._hf_pipeline = None

    # ---- 对外统一接口：messages = [{"role":"system"|"user"|"assistant","content": "..."}] ----
    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "openai":
            return self._chat_openai(messages)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages)
        elif self.provider == "hf":
            return self._chat_hf(messages)
        else:
            raise ValueError(f"Unknown provider {self.provider}")

    def chat_json(self, messages: List[Dict[str, str]]) -> dict:
        """强制 JSON：解析失败自动重试一次，追加ONLY JSON约束。"""
        txt = self.chat(messages)
        try:
            return json.loads(_extract_json(txt))
        except Exception:
            retry = messages + [{"role":"user","content":"Return ONLY minified valid JSON. No prose."}]
            txt2 = self.chat(retry)
            return json.loads(_extract_json(txt2))

