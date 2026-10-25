"""
Answer generation with an open instruct model from Hugging Face.
"""

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation.prompts import PROMPT_VERSION, build_messages


DEFAULT_CONFIG_PATH = Path("configs/generation_config.json")

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_generation_config(path: str | Path | None = None) -> dict:
    with Path(path or DEFAULT_CONFIG_PATH).open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return DTYPES[requested]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def clean_answer(text: str) -> str:
    """
    Keep the first non-empty line and drop a leading "Answer:" label.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    answer = lines[0] if lines else ""
    return re.sub(r"^(final\s+)?answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()


class Reader:
    """
    Wraps a causal LM: builds the prompt from a question and retrieved
    chunks, generates greedily, and returns the answer with metadata.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.revision = config.get("revision")
        self.device = pick_device(config.get("device", "auto"))
        self.dtype = pick_dtype(config.get("dtype", "auto"), self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        # Left padding so batched prompts end where generation starts.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "prompt_version": PROMPT_VERSION,
            "max_new_tokens": self.config["max_new_tokens"],
            "temperature": self.config["temperature"],
            "max_context_tokens": self.config["max_context_tokens"],
        }

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def build_prompt(self, question: str, chunks: list[dict] | None) -> tuple[str, list[dict] | None]:
        """
        The rendered prompt and the chunks it contains. If the prompt is over
        max_context_tokens, the lowest-ranked chunks are dropped first.
        """
        if chunks is None:
            always_answer = self.config.get("closed_book_always_answer", False)
            return self.render(build_messages(question, None, always_answer)), None

        kept = list(chunks)
        prompt = self.render(build_messages(question, kept))
        while len(kept) > 1 and self.n_tokens(prompt) > self.config["max_context_tokens"]:
            kept.pop()
            prompt = self.render(build_messages(question, kept))
        return prompt, kept

    @torch.inference_mode()
    def _generate(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

        settings = {"max_new_tokens": self.config["max_new_tokens"], "pad_token_id": self.tokenizer.pad_token_id}
        if self.config["temperature"] > 0:
            settings.update(do_sample=True, temperature=self.config["temperature"])
        else:
            settings.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        output = self.model.generate(**inputs, **settings)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def answer_batch(self, questions: list[str], chunk_lists: list[list[dict] | None]) -> list[dict]:
        """
        Answer several questions at once. chunk_lists[i] = None means
        closed-book for question i.
        """
        built = [self.build_prompt(q, c) for q, c in zip(questions, chunk_lists)]
        prompts = [p for p, _ in built]

        start = time.perf_counter()
        raw = self._generate(prompts)
        seconds = (time.perf_counter() - start) / len(prompts)

        results = []
        for (prompt, kept), text in zip(built, raw):
            results.append(
                {
                    "answer": clean_answer(text),
                    "raw_output": text,
                    "prompt_tokens": self.n_tokens(prompt),
                    "context_chunk_ids": [c["chunk_id"] for c in kept] if kept is not None else [],
                    "generation_seconds": round(seconds, 3),
                }
            )
        return results

    def answer(self, question: str, chunks: list[dict] | None) -> dict:
        return self.answer_batch([question], [chunks])[0]
