"""vLLM judge using OpenAI-compatible API served by vLLM."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_openai = None


def _get_openai():
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the vLLM judge backend. "
                "Install it with: pip install openai"
            )
    return _openai


class VLLMJudge:
    """Judge using a vLLM server via OpenAI-compatible API.

    The user must start a vLLM server separately, e.g.:
        vllm serve hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \\
            --quantization awq --tensor-parallel-size 2

    Args:
        base_url: Base URL of the vLLM server (default: http://localhost:8000/v1)
        model: Model name as registered in the vLLM server
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        openai = _get_openai()
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't require a real key
            timeout=timeout,
        )
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logger.info(f"VLLMJudge initialized: base_url={base_url}, model={model}")

    def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Run judge inference via vLLM OpenAI-compatible API.

        Args:
            system_prompt: System instruction for the judge.
            user_prompt: User query containing the evaluation task.
            response_format: Optional JSON schema dict for vLLM guided decoding.
                Passed as extra_body={"guided_json": schema}.

        Returns:
            Generated response string, or None if request fails.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        if response_format is not None:
            kwargs["extra_body"] = {"guided_json": response_format}

        try:
            completion = self.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logger.error(f"vLLM judge request failed: {e}")
            return None

    def judge_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> List[Optional[str]]:
        """Run judge inference on multiple prompts sequentially.

        For true batching, increase chunk_size in config instead —
        vLLM handles concurrency internally via continuous batching.
        """
        return [
            self.judge(system_prompt, prompt, response_format)
            for prompt in user_prompts
        ]
