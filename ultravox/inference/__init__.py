# Lazy imports to avoid loading unnecessary dependencies
# Core Ultravox imports are direct, optional modules are lazy-loaded

from ultravox.inference.base import InferenceChunk
from ultravox.inference.base import InferenceGenerator
from ultravox.inference.base import InferenceStats
from ultravox.inference.base import VoiceInference
from ultravox.inference.base import VoiceOutput


def __getattr__(name):
    """Lazy loading for optional and heavy dependencies."""
    # Core Ultravox inference (heavy - requires accelerate, model code)
    if name == "LocalInference":
        from ultravox.inference.infer import LocalInference

        return LocalInference
    elif name == "UltravoxInference":
        from ultravox.inference.ultravox_infer import UltravoxInference

        return UltravoxInference
    elif name == "ModelTypeMismatchError":
        from ultravox.inference.ultravox_infer import ModelTypeMismatchError

        return ModelTypeMismatchError
    # Qwen3-Omni inference (optional - separate from core Ultravox)
    elif name == "Qwen3OmniVLLMInference":
        from ultravox.inference.qwen3_omni_vllm_infer import Qwen3OmniVLLMInference

        return Qwen3OmniVLLMInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core base classes
    "VoiceInference",
    "VoiceOutput",
    "InferenceChunk",
    "InferenceStats",
    "InferenceGenerator",
    # Core Ultravox inference (lazy)
    "LocalInference",
    "UltravoxInference",
    "ModelTypeMismatchError",
    # Qwen3-Omni inference (lazy, optional)
    "Qwen3OmniVLLMInference",
]
