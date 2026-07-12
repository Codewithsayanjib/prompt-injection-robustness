"""Generation backends behind one interface.

  mock : deterministic stand-in, no weights (for testing the harness).
  mlx  : Apple-Silicon 4-bit via mlx-lm (used for every model here).
  hf   : transformers + bitsandbytes 4-bit on CUDA (optional).

Real backends run 4-bit, greedy (temperature 0) so ASR/FPR are deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class GenConfig:
    max_new_tokens: int = 96
    temperature: float = 0.0
    seed: int = 0


def _probe_system_role(tokenizer) -> bool:
    # Gemma-2 and some others reject a 'system' role; check once.
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def _build_messages(system: str, user: str, supports_system: bool) -> list[dict]:
    # Fold system into the user turn for models without a system role, keeping
    # the instruction content identical.
    if supports_system:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return [{"role": "user", "content": f"{system}\n\n{user}"}]


class Backend:
    def __init__(self, model_id: str, gen: GenConfig):
        self.model_id = model_id
        self.gen = gen

    def generate(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def label(self) -> str:
        return self.__class__.__name__


class MockBackend(Backend):
    """Deterministic stand-in: leaks on a hash-determined fraction of attacks,
    less when the prompt looks defended. Lets metrics/plot/resume be tested
    without any weights."""

    name = "mock"

    _BASE_RATE = {
        "mock-weak": 0.75,
        "mock-strong": 0.35,
    }

    def __init__(self, model_id: str, gen: GenConfig, leak_rate: float | None = None):
        super().__init__(model_id, gen)
        self.leak_rate = leak_rate if leak_rate is not None else self._BASE_RATE.get(model_id, 0.6)

    @staticmethod
    def _canary_from_system(system: str) -> str | None:
        # Match a quoted UPPERCASE token so "user's" isn't picked up.
        import re

        m = re.search(r"'([A-Z][A-Z0-9_]{1,})'", system)
        return m.group(1) if m else None

    def _unit_hash(self, s: str) -> float:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def generate(self, system: str, user: str) -> str:
        canary = self._canary_from_system(system)
        defended = ("UNTRUSTED_DATA" in system) or ("Reminder:" in user) or (
            "Ignore any instruction" in system)
        effective = self.leak_rate * (0.4 if defended else 1.0)
        cues = ("ignore", "reveal", "secret", "password", "system prompt", "instruction")
        looks_adversarial = any(c in user.lower() for c in cues)
        if not looks_adversarial:
            effective *= 0.02
        roll = self._unit_hash(system + "||" + user)
        if canary and roll < effective:
            return f"Sure, the secret password is {canary}."
        return "Here is a concise answer to your question."


class MLXBackend(Backend):
    """Apple-Silicon 4-bit generation via mlx-lm."""

    name = "mlx"

    def __init__(self, model_id: str, gen: GenConfig):
        super().__init__(model_id, gen)
        from mlx_lm import load

        self._model, self._tokenizer = load(model_id)
        self._sys_ok = _probe_system_role(self._tokenizer)

    def generate(self, system: str, user: str) -> str:
        from mlx_lm import generate as mlx_generate

        messages = _build_messages(system, user, self._sys_ok)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.gen.max_new_tokens,
            verbose=False,
        )


class HFBackend(Backend):
    """transformers + bitsandbytes 4-bit on CUDA."""

    name = "hf"

    def __init__(self, model_id: str, gen: GenConfig):
        super().__init__(model_id, gen)
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        self._torch = torch
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map="auto"
        )
        self._model.eval()
        self._sys_ok = _probe_system_role(self._tokenizer)

    def generate(self, system: str, user: str) -> str:
        torch = self._torch
        messages = _build_messages(system, user, self._sys_ok)
        # return_dict so we get input_ids + attention_mask and can splat them;
        # passing a BatchEncoding positionally to generate() breaks on newer
        # transformers.
        enc = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()


_BACKENDS = {
    MockBackend.name: MockBackend,
    MLXBackend.name: MLXBackend,
    HFBackend.name: HFBackend,
}


def load_backend(backend: str, model_id: str, gen: GenConfig | None = None, **kw) -> Backend:
    gen = gen or GenConfig()
    if backend not in _BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[backend](model_id, gen, **kw)
