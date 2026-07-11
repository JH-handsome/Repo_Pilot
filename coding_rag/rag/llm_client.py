"""
LLM 客户端模块
提供 OpenAI 兼容的聊天补全 API 客户端
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderPreset:
    """
    LLM 服务提供商预设配置
    
    Attributes:
        name: 提供商名称
        base_url: API 基础 URL
        model: 默认模型名称
        api_key_env: API 密钥环境变量名
    """
    name: str
    base_url: str
    model: str
    api_key_env: str


# LLM 服务提供商预设配置
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "kimi": ProviderPreset(
        name="kimi",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k2.6",
        api_key_env="MOONSHOT_API_KEY",
    ),
    "zhipu": ProviderPreset(
        name="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4.7",
        api_key_env="ZHIPU_API_KEY",
    ),
}


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM 客户端配置
    
    Attributes:
        provider: 提供商名称
        base_url: API 基础 URL
        model: 模型名称
        api_key: API 密钥
        api_key_env: API 密钥环境变量名
        timeout: 请求超时时间（秒）
        max_tokens: 最大输出 token 数
        temperature: 温度参数，控制生成随机性
    """
    provider: str
    base_url: str
    model: str
    api_key: str
    api_key_env: str
    timeout: int = 60
    max_tokens: int = 800
    temperature: float | None = None


class OpenAICompatibleChatClient:
    """
    OpenAI 兼容的聊天补全 API 客户端
    
    使用轻量级 HTTP 客户端实现，无需额外依赖
    """

    def __init__(self, config: LLMConfig):
        """初始化客户端
        
        Args:
            config: LLM 配置对象
        """
        self.config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        发送聊天补全请求
        
        Args:
            messages: 消息列表，每个消息包含 role 和 content
            
        Returns:
            LLM 返回的文本内容
            
        Raises:
            RuntimeError: 请求失败时抛出
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        # 如果设置了温度参数，则添加到请求体
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature

        # 将请求体序列化为 JSON
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            chat_completions_url(self.config.base_url),
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        # 发送 HTTP 请求
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        # 处理 HTTP 错误
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {error.code}: {detail}") from error
        # 处理 URL 错误
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error

        # 从响应中提取消息内容
        return extract_message_content(payload)


def build_llm_config(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout: int = 60,
    max_tokens: int = 800,
    temperature: float | None = None,
) -> LLMConfig:
    """
    构建 LLM 配置
    
    支持通过参数、环境变量或预设来构建配置
    
    Args:
        provider: 提供商名称
        model: 覆盖预设的模型名称
        base_url: 覆盖预设的 API 基础 URL
        api_key_env: 覆盖预设的 API 密钥环境变量名
        timeout: 请求超时时间（秒）
        max_tokens: 最大输出 token 数
        temperature: 温度参数
        
    Returns:
        构建好的 LLMConfig 对象
        
    Raises:
        ValueError: 缺少必要配置时抛出
    """
    # 获取提供商预设
    preset = PROVIDER_PRESETS.get(provider)

    # 按优先级确定配置：参数 > 环境变量 > 预设
    final_base_url = first_value(base_url, os.getenv("LLM_BASE_URL"), preset.base_url if preset else None)
    final_model = first_value(model, os.getenv("LLM_MODEL"), preset.model if preset else None)
    final_api_key_env = first_value(
        api_key_env,
        os.getenv("LLM_API_KEY_ENV"),
        preset.api_key_env if preset else None,
    )

    # 获取 API 密钥
    api_key = os.getenv("LLM_API_KEY")
    if not api_key and final_api_key_env:
        # 如果指定了环境变量名，则从该变量获取
        api_key = os.getenv(final_api_key_env)

    # 检查必要配置是否缺失
    missing = []
    if not final_base_url:
        missing.append("base_url")
    if not final_model:
        missing.append("model")
    if not api_key:
        if final_api_key_env:
            missing.append(f"environment variable {final_api_key_env}")
        else:
            missing.append("LLM_API_KEY or --llm-api-key-env")

    # 如果有缺失的配置，抛出异常
    if missing:
        raise ValueError("Missing LLM configuration: " + ", ".join(missing))

    # 此时确保所有必要配置都已存在（已通过上面的检查）
    assert final_base_url is not None, "base_url should not be None after missing check"
    assert final_model is not None, "model should not be None after missing check"
    assert api_key is not None, "api_key should not be None after missing check"

    # 构建并返回配置对象
    return LLMConfig(
        provider=provider,
        base_url=final_base_url,
        model=final_model,
        api_key=api_key,
        api_key_env=final_api_key_env or "LLM_API_KEY",
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def available_provider_names() -> list[str]:
    """获取所有可用的提供商名称列表"""
    return sorted(PROVIDER_PRESETS)


def chat_completions_url(base_url: str) -> str:
    """
    构建聊天补全 API 的完整 URL
    
    Args:
        base_url: API 基础 URL
        
    Returns:
        完整的聊天补全 API URL
    """
    base = base_url.rstrip("/")  # 移除末尾斜杠
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def extract_message_content(payload: dict[str, Any]) -> str:
    """
    从 LLM 响应中提取消息内容
    
    Args:
        payload: LLM 返回的 JSON 响应
        
    Returns:
        提取的消息内容
        
    Raises:
        RuntimeError: 响应格式错误时抛出
    """
    # 获取选择列表
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response has no choices: {payload}")

    # 获取第一个选择的消息
    message = choices[0].get("message") or {}
    # 获取消息内容
    content = message.get("content", "")
    # 处理字符串内容
    if isinstance(content, str):
        return content
    # 处理列表内容（流式响应）
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def first_value(*values: str | None) -> str | None:
    """
    返回第一个非 None 的值
    
    Args:
        *values: 可选值列表
        
    Returns:
        第一个非 None 的值，如果没有则返回 None
    """
    for value in values:
        if value:
            return value
    return None
