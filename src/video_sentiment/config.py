from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Settings:
    whisper_model: str = "large-v3"
    language: str = "en"
    device: str = "auto"
    batch_size: int = 4
    min_speakers: int | None = None
    max_speakers: int | None = None
    name_hints: list[str] = field(default_factory=list)
    initial_prompt: str = "English conversation."
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    context_sentences: int = 8
    valence_positive: float = 0.65
    valence_negative: float = 0.40
    conflict_confidence_scale: float = 0.75
    tone_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    hf_token: str | None = None

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def compute_type(self) -> str:
        return "float16" if self.resolved_device() == "cuda" else "int8"

    def asr_prompt(self) -> str | None:
        parts: list[str] = []
        if self.initial_prompt and self.initial_prompt.strip():
            parts.append(self.initial_prompt.strip())
        hints = [str(item).strip() for item in self.name_hints or [] if str(item).strip()]
        if hints:
            parts.append("Names that may be spoken: " + ", ".join(hints) + ".")
        text = " ".join(parts).strip()
        return text or None


def load_settings(path: Path | None = None) -> Settings:
    settings = Settings()
    if path is None and Path("config.yaml").exists():
        path = Path("config.yaml")
    if path is not None and path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a mapping of settings.")
        known = {item.name for item in fields(Settings)}
        for key, value in raw.items():
            if key not in known:
                continue
            if key == "name_hints":
                if not value:
                    value = []
                elif isinstance(value, str):
                    value = [value.strip()] if value.strip() else []
                else:
                    value = [str(item).strip() for item in value if str(item).strip()]
            setattr(settings, key, value)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token = user_environment("HF_TOKEN")
    if token:
        settings.hf_token = token
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    if os.environ.get("OLLAMA_MODEL"):
        settings.ollama_model = os.environ["OLLAMA_MODEL"]
    if os.environ.get("OLLAMA_HOST"):
        settings.ollama_url = os.environ["OLLAMA_HOST"]
    return settings


def user_environment(name: str) -> str | None:
    """Read a user-level environment variable saved with setx on Windows."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
