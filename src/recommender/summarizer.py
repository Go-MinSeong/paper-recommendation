"""Paper summarization using OpenAI LLM.

This module provides functionality to generate paper summaries using GPT models.
"""

from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger import log
from config.settings import get_settings


class SummarizationError(Exception):
    """Exception raised for summarization errors."""

    pass


class PaperSummarizer:
    """Paper summarizer using OpenAI LLM.

    This class generates two types of summaries:
    1. Core summary: General abstract of the paper
    2. Contextualized summary: Summary tailored to user's interest

    Attributes:
        model: OpenAI model name
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize paper summarizer.

        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: Model name (uses settings if not provided)
            temperature: Generation temperature (uses settings if not provided)
            max_tokens: Max tokens (uses settings if not provided)

        Examples:
            >>> summarizer = PaperSummarizer()
            >>> summary = await summarizer.generate_core_summary(title, abstract)
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        self._client = AsyncOpenAI(api_key=self.api_key)

        log.info(f"Initialized PaperSummarizer with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_core_summary(self, title: str, abstract: str) -> str:
        """Generate core summary of a paper.

        This summary provides a general overview without specific context.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            str: Core summary (2-3 sentences)

        Raises:
            SummarizationError: If summarization fails

        Examples:
            >>> summarizer = PaperSummarizer()
            >>> summary = await summarizer.generate_core_summary(
            ...     "Visual Language Models for Detection",
            ...     "This paper presents a novel approach..."
            ... )
        """
        try:
            log.debug(f"Generating core summary for: '{title[:50]}...'")

            prompt = f"""다음 논문의 핵심 내용을 2-3문장으로 요약해주세요.
기술적인 내용을 포함하되, 일반 독자도 이해할 수 있게 작성해주세요.

제목: {title}

초록:
{abstract}

요약:"""

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 AI/ML 논문을 명확하고 간결하게 요약하는 전문가입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            summary = response.choices[0].message.content.strip()

            log.debug(f"Generated core summary (length: {len(summary)})")

            return summary

        except Exception as e:
            error_msg = f"Failed to generate core summary: {str(e)}"
            log.error(error_msg)
            raise SummarizationError(error_msg) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_contextualized_summary(
        self,
        title: str,
        abstract: str,
        user_interest: str,
    ) -> str:
        """Generate contextualized summary based on user interest.

        This summary focuses on aspects relevant to the user's research interest.

        Args:
            title: Paper title
            abstract: Paper abstract
            user_interest: User's research interest description

        Returns:
            str: Contextualized summary (2-3 sentences)

        Raises:
            SummarizationError: If summarization fails

        Examples:
            >>> summarizer = PaperSummarizer()
            >>> summary = await summarizer.generate_contextualized_summary(
            ...     "Visual Language Models for Detection",
            ...     "This paper presents...",
            ...     "VLM을 이용한 CCTV 객체 검출"
            ... )
        """
        try:
            log.debug(f"Generating contextualized summary for: '{title[:50]}...'")

            prompt = f"""다음 논문을 사용자의 관심사와 연결하여 2-3문장으로 요약해주세요.
사용자의 관심사와 어떻게 관련되는지, 어떤 도움이 될 수 있는지 중점적으로 설명해주세요.

사용자 관심사: {user_interest}

논문 제목: {title}

논문 초록:
{abstract}

맞춤 요약:"""

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 사용자의 연구 관심사에 맞춰 논문을 분석하고 설명하는 전문가입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            summary = response.choices[0].message.content.strip()

            log.debug(f"Generated contextualized summary (length: {len(summary)})")

            return summary

        except Exception as e:
            error_msg = f"Failed to generate contextualized summary: {str(e)}"
            log.error(error_msg)
            raise SummarizationError(error_msg) from e

    async def close(self) -> None:
        """Close the OpenAI client.

        Examples:
            >>> summarizer = PaperSummarizer()
            >>> await summarizer.close()
        """
        await self._client.close()
        log.debug("PaperSummarizer client closed")

    async def __aenter__(self) -> "PaperSummarizer":
        """Async context manager entry.

        Returns:
            PaperSummarizer: Self instance
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
