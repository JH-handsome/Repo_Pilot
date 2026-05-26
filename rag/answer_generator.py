"""答案生成器模块

提供基于RAG检索结果的答案生成功能，支持多种生成模式：
- judge: 搜索结果判断和总结
- code-understand: 代码理解
- code-generate: 代码生成
- leetcode: LeetCode解题
- api: API使用示例生成
"""

from typing import TYPE_CHECKING

from coding_rag.llm_client import LLMConfig, OpenAICompatibleChatClient

from rag.prompt import (
    GenerationMode,
    build_user_prompt,
    format_results_as_context,
    get_system_prompt,
)

if TYPE_CHECKING:
    from coding_rag.bm25_retriever import SearchResult


class AnswerGenerator:
    """基于RAG检索结果的答案生成器"""

    def __init__(
        self,
        config: LLMConfig,
        mode: GenerationMode = GenerationMode.JUDGE,
        max_context_chars: int = 12000,
    ):
        """初始化答案生成器

        Args:
            config: LLM客户端配置
            mode: 生成模式
            max_context_chars: 发送给LLM的最大上下文字符数
        """
        self.client = OpenAICompatibleChatClient(config)
        self.mode = mode
        self.max_context_chars = max_context_chars

    def generate(
        self,
        query: str,
        results: list["SearchResult"],
    ) -> str:
        """基于检索结果生成答案

        Args:
            query: 用户查询
            results: BM25检索结果列表

        Returns:
            生成的答案

        Raises:
            RuntimeError: LLM请求失败
        """
        # 格式化检索结果为上下文
        context = format_results_as_context(
            results,
            max_context_chars=self.max_context_chars,
        )

        # 构建提示词
        system_prompt = get_system_prompt(self.mode)
        user_prompt = build_user_prompt(self.mode, query, context)

        # 调用LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            answer = self.client.complete(messages)
            return answer
        except Exception as e:
            raise RuntimeError(f"生成答案失败: {e}") from e

    def set_mode(self, mode: GenerationMode) -> None:
        """设置生成模式

        Args:
            mode: 新的生成模式
        """
        self.mode = mode

    def set_max_context_chars(self, max_chars: int) -> None:
        """设置最大上下文字符数

        Args:
            max_chars: 最大上下文字符数
        """
        self.max_context_chars = max_chars


def build_generator(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout: int = 60,
    max_tokens: int = 2000,
    temperature: float | None = None,
    mode: GenerationMode = GenerationMode.JUDGE,
    max_context_chars: int = 12000,
) -> AnswerGenerator:
    """构建答案生成器

    Args:
        provider: LLM提供者名称
        model: 覆盖预设的模型名称
        base_url: 覆盖预设的API基础URL
        api_key_env: 覆盖预设的API密钥环境变量名
        timeout: LLM请求超时时间（秒）
        max_tokens: 最大输出token数
        temperature: 温度参数
        mode: 生成模式
        max_context_chars: 最大上下文字符数

    Returns:
        配置好的AnswerGenerator实例

    Raises:
        ValueError: 缺少必要的配置
    """
    from coding_rag.llm_client import build_llm_config

    config = build_llm_config(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return AnswerGenerator(
        config=config,
        mode=mode,
        max_context_chars=max_context_chars,
    )