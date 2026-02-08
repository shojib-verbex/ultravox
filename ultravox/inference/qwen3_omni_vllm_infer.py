"""
Qwen3-Omni vLLM inference implementation for the ultravox evaluation pipeline.

This module provides Qwen3OmniVLLMInference class that uses a vLLM server
for fast inference with Qwen3-Omni models.

Requires a running vLLM server with Qwen3-Omni model:
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 7000 --dtype bfloat16
"""

import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import requests
import soundfile as sf

from ultravox import data as datasets
from ultravox.data.types import AUDIO_PLACEHOLDER
from ultravox.inference import base

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 1024
DEFAULT_VLLM_URL = "http://localhost:7000"


class Qwen3OmniVLLMInference(base.VoiceInference):
    """
    Inference class for Qwen3-Omni models using vLLM server.

    Uses the OpenAI-compatible API provided by vLLM for fast inference.
    Much faster than HuggingFace transformers due to:
    - PagedAttention for efficient memory management
    - Continuous batching
    - Optimized CUDA kernels
    """

    def __init__(
        self,
        model_path: str,
        vllm_url: str = DEFAULT_VLLM_URL,
        device: Optional[str] = None,  # Ignored, vLLM handles device
        data_type: Optional[str] = None,  # Ignored, vLLM handles dtype
        use_flash_attention: bool = False,  # Ignored, vLLM handles attention
    ):
        """
        Initialize Qwen3-Omni vLLM inference client.

        Args:
            model_path: Model name (used for validation, vLLM server must have it loaded)
            vllm_url: URL of the vLLM server (default: http://localhost:7000)
            device: Ignored (vLLM handles device placement)
            data_type: Ignored (vLLM handles dtype)
            use_flash_attention: Ignored (vLLM handles attention implementation)
        """
        self.model_path = model_path
        self.vllm_url = vllm_url.rstrip("/")
        self.api_url = f"{self.vllm_url}/v1/chat/completions"

        logging.info(f"Connecting to vLLM server at {self.vllm_url}")
        logging.info(f"Model: {model_path}")

        # Verify server is running
        self._verify_server()

        logging.info("vLLM server connection verified successfully")

    def _verify_server(self):
        """Verify the vLLM server is running and accessible."""
        try:
            response = requests.get(f"{self.vllm_url}/health", timeout=10)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"\n{'='*70}\n"
                f"VLLM SERVER NOT FOUND\n"
                f"{'='*70}\n\n"
                f"Could not connect to vLLM server at: {self.vllm_url}\n\n"
                f"Please start the vLLM server first:\n"
                f"  vllm serve {self.model_path} --port 7000 --dtype bfloat16\n\n"
                f"Or specify a different URL with --vllm_url\n"
                f"{'='*70}\n"
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"vLLM server health check failed: {e}")

    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Convert audio numpy array to base64-encoded WAV string.

        vLLM's OpenAI-compatible API accepts audio as base64-encoded data
        in the message content.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Base64-encoded WAV audio string
        """
        # Write audio to in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)

        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return audio_base64

    def _convert_to_vllm_messages(
        self,
        sample: datasets.VoiceSample,
    ) -> List[dict]:
        """
        Convert VoiceSample messages to vLLM/OpenAI multimodal format.

        VoiceSample format:
            messages = [
                {"role": "user", "content": "<|audio|>\\n\\nWhat is being said?"},
            ]

        vLLM format:
            messages = [
                {"role": "user", "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
                    {"type": "text", "text": "What is being said?"}
                ]}
            ]

        Args:
            sample: VoiceSample with messages and optional audio

        Returns:
            List of messages in vLLM/OpenAI multimodal format
        """
        # Prepare audio data URL if audio is present
        audio_data_url = None
        if sample.audio is not None:
            audio_base64 = self._audio_to_base64(sample.audio, sample.sample_rate)
            audio_data_url = f"data:audio/wav;base64,{audio_base64}"

        vllm_messages = []

        for msg in sample.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant":
                # Assistant messages remain as plain text
                vllm_messages.append({"role": "assistant", "content": content})
            elif role == "user":
                # User messages need multimodal content conversion
                vllm_content = self._convert_user_content(content, audio_data_url)
                vllm_messages.append({"role": "user", "content": vllm_content})
            elif role == "system":
                # System messages as plain text
                vllm_messages.append({"role": "system", "content": content})

        return vllm_messages

    def _convert_user_content(
        self,
        content: str,
        audio_data_url: Optional[str],
    ) -> List[dict]:
        """
        Convert user message content to vLLM multimodal format.

        Args:
            content: Original message content with potential <|audio|> placeholder
            audio_data_url: Base64 data URL for the audio

        Returns:
            List of content items in vLLM/OpenAI format
        """
        vllm_content = []

        if AUDIO_PLACEHOLDER not in content or audio_data_url is None:
            # No audio placeholder or no audio, just return text
            return content  # Return as string for text-only

        # Split content around audio placeholder(s)
        parts = content.split(AUDIO_PLACEHOLDER)

        for i, part in enumerate(parts):
            # Add text part if non-empty
            text = part.strip()
            if text:
                vllm_content.append({"type": "text", "text": text})

            # Add audio after each part except the last one
            if i < len(parts) - 1:
                vllm_content.append({
                    "type": "audio_url",
                    "audio_url": {"url": audio_data_url}
                })

        return vllm_content

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        """
        Perform inference on a single VoiceSample using vLLM server.

        Args:
            sample: VoiceSample containing messages and optional audio
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            VoiceOutput with generated text and token counts
        """
        import time

        # Convert to vLLM message format
        t0 = time.monotonic()
        messages = self._convert_to_vllm_messages(sample)
        t_convert = time.monotonic() - t0

        # Build request payload
        payload = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": max_tokens or MAX_NEW_TOKENS,
            "temperature": temperature if temperature is not None else 0.0,
        }

        # Make request to vLLM server
        t1 = time.monotonic()
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=600,  # 10 min timeout - 30B model on single GPU is slow
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"vLLM request failed: {e}")
            raise
        t_request = time.monotonic() - t1

        # Parse response
        result = response.json()
        choice = result["choices"][0]
        output_text = choice["message"]["content"]

        # Get token counts from usage
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        logging.info(
            f"infer: convert={t_convert:.3f}s, request={t_request:.3f}s, "
            f"in_tokens={input_tokens}, out_tokens={output_tokens}"
        )

        return base.VoiceOutput(
            text=output_text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def infer_batch(
        self,
        samples: List[datasets.VoiceSample],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[base.VoiceOutput]:
        """
        Batch inference - processes samples concurrently for better GPU utilization.

        Uses ThreadPoolExecutor to send multiple requests in parallel,
        allowing vLLM to batch them together using continuous batching.

        Args:
            samples: List of VoiceSample objects
            max_tokens: Maximum tokens to generate per sample
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent requests (default: all samples)

        Returns:
            List of VoiceOutput objects in the same order as input samples
        """
        import time

        if len(samples) <= 1:
            return [self.infer(sample, max_tokens, temperature) for sample in samples]

        # Send all requests concurrently - vLLM's continuous batching handles scheduling
        num_workers = max_concurrent or len(samples)
        results = [None] * len(samples)

        t_batch_start = time.monotonic()
        logging.info(
            f"infer_batch: starting {len(samples)} samples with {num_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with their indices
            future_to_idx = {
                executor.submit(self.infer, sample, max_tokens, temperature): idx
                for idx, sample in enumerate(samples)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logging.error(f"Inference failed for sample {idx}: {e}")
                    # Return empty output on failure
                    results[idx] = base.VoiceOutput(text="", input_tokens=0, output_tokens=0)

        t_batch_total = time.monotonic() - t_batch_start
        logging.info(
            f"infer_batch: {len(samples)} samples done in {t_batch_total:.1f}s "
            f"({t_batch_total/len(samples):.2f}s/sample)"
        )

        return results
