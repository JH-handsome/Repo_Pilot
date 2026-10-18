"""
LLM 客户端模块
提供 OpenAI 兼容的聊天补全 API 客户端
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock, local
from typing import Any, Callable, Mapping


LOGGER = logging.getLogger(__name__)
TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
LLM_OBSERVATION_KEYS = frozenset({
    "schema_version", "operation", "provider", "model", "attempts",
    "latency_ms", "outcome", "error_class",
})
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:/+-]{1,128}$")
_SAFE_ERROR_CLASS = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Strict, testable retry limits for explicitly transient transport failures."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        """Reject unbounded or nonsensical retry settings at configuration time."""
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("retry max_attempts must be between 1 and 5")
        if not 0 <= self.base_delay_seconds <= 10:
            raise ValueError("retry base_delay_seconds must be between 0 and 10")
        if not 0 <= self.max_delay_seconds <= 30:
            raise ValueError("retry max_delay_seconds must be between 0 and 30")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("retry base_delay_seconds cannot exceed max_delay_seconds")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("retry jitter_ratio must be between 0 and 1")


@dataclass(frozen=True)
class LLMCallObservation:
    """Sanitized metadata for one logical LLM call across all retry attempts."""

    schema_version: int
    operation: str
    provider: str
    model: str
    attempts: int
    latency_ms: float
    outcome: str
    error_class: str | None


def normalize_llm_call_observation(value: object) -> dict[str, Any] | None:
    """Return a bounded fixed-schema observation, rejecting unsafe or malformed data."""
    if isinstance(value, LLMCallObservation):
        value = asdict(value)
    if not isinstance(value, Mapping) or set(value) != LLM_OBSERVATION_KEYS:
        return None
    schema_version = value.get("schema_version")
    attempts = value.get("attempts")
    latency_ms = value.get("latency_ms")
    outcome = value.get("outcome")
    error_class = value.get("error_class")
    labels = [value.get("operation"), value.get("provider"), value.get("model")]
    if schema_version != 1 or type(attempts) is not int or not 1 <= attempts <= 5:
        return None
    if type(latency_ms) not in (int, float) or not math.isfinite(latency_ms):
        return None
    if not 0 <= float(latency_ms) <= 86_400_000:
        return None
    if outcome not in {"success", "error"}:
        return None
    if not all(isinstance(label, str) and _SAFE_LABEL.fullmatch(label) for label in labels):
        return None
    if error_class is not None and (
        not isinstance(error_class, str) or not _SAFE_ERROR_CLASS.fullmatch(error_class)
    ):
        return None
    if (outcome == "success") != (error_class is None):
        return None
    return {
        "schema_version": 1,
        "operation": labels[0],
        "provider": labels[1],
        "model": labels[2],
        "attempts": attempts,
        "latency_ms": round(float(latency_ms), 3),
        "outcome": outcome,
        "error_class": error_class,
    }


class LLMRequestError(RuntimeError):
    """Safe public error that carries retry classification without response details."""

    def __init__(self, message: str, *, error_class: str, transient: bool):
        """Initialize a sanitized failure with stable classification metadata."""
        super().__init__(message)
        self.error_class = error_class
        self.transient = transient


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
        retry_policy: 有界暂时性故障重试策略
    """
    provider: str
    base_url: str
    model: str
    api_key: str
    api_key_env: str
    timeout: int = 60
    max_tokens: int = 800
    temperature: float | None = None
    retry_policy: LLMRetryPolicy = field(default_factory=LLMRetryPolicy)


class OpenAICompatibleChatClient:
    """
    OpenAI 兼容的聊天补全 API 客户端
    
    使用轻量级 HTTP 客户端实现，无需额外依赖
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        observer: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.perf_counter,
        observation_limit: int = 100,
    ):
        """初始化客户端
        
        Args:
            config: LLM 配置对象
        """
        self.config = config
        if observation_limit < 1:
            raise ValueError("observation_limit must be positive")
        self._observer = observer
        self._sleep = sleep
        self._random_value = random_value
        self._clock = clock
        self._observations: deque[LLMCallObservation] = deque(maxlen=observation_limit)
        self._observation_lock = Lock()
        self._thread_observation = local()

    @property
    def observations(self) -> tuple[LLMCallObservation, ...]:
        """Return a stable snapshot of bounded, sanitized call observations."""
        with self._observation_lock:
            return tuple(self._observations)

    @property
    def last_observation(self) -> LLMCallObservation | None:
        """Return the latest sanitized logical-call observation, if any."""
        with self._observation_lock:
            return self._observations[-1] if self._observations else None

    def take_call_observation(self) -> dict[str, Any] | None:
        """Consume this thread's latest logical-call observation exactly once."""
        observation = getattr(self._thread_observation, "pending", None)
        if hasattr(self._thread_observation, "pending"):
            del self._thread_observation.pending
        return normalize_llm_call_observation(observation)

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

        started = self._clock()
        attempts = 0
        outcome = "error"
        error_class: str | None = None
        try:
            while attempts < self.config.retry_policy.max_attempts:
                attempts += 1
                try:
                    payload = self._request_once(body)
                    content = extract_message_content(payload)
                    outcome = "success"
                    error_class = None
                    return content
                except LLMRequestError as error:
                    error_class = error.error_class
                    if not error.transient or attempts >= self.config.retry_policy.max_attempts:
                        raise
                    self._sleep(self._retry_delay(attempts))
        except Exception as error:
            if error_class is None:
                error_class = type(error).__name__
            raise
        finally:
            observation = LLMCallObservation(
                schema_version=1,
                operation="chat.completions",
                provider=self.config.provider,
                model=self.config.model,
                attempts=attempts,
                latency_ms=round(max(0.0, (self._clock() - started) * 1000), 3),
                outcome=outcome,
                error_class=error_class,
            )
            self._record_observation(observation)

    def _request_once(self, body: dict[str, Any]) -> dict[str, Any]:
        """Perform one HTTP attempt and translate failures into safe classifications."""
        request = urllib.request.Request(
            chat_completions_url(self.config.base_url),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            status = error.code
            transient = status in TRANSIENT_HTTP_STATUS
            try:
                error.close()
            except Exception:
                # Resource cleanup must not replace the stable, sanitized request error.
                pass
            raise LLMRequestError(
                f"LLM request failed with HTTP {status}",
                error_class=f"http_{status}",
                transient=transient,
            ) from error
        except urllib.error.URLError as error:
            transient, classification = classify_url_error(error)
            raise LLMRequestError(
                "LLM request failed at the transport layer",
                error_class=classification,
                transient=transient,
            ) from error
        except OSError as error:
            transient, classification = classify_transport_reason(error)
            raise LLMRequestError(
                "LLM request failed at the transport layer",
                error_class=classification,
                transient=transient,
            ) from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LLMRequestError(
                "LLM response was not valid JSON", error_class="invalid_json", transient=False
            ) from error
        if not isinstance(payload, dict):
            raise LLMRequestError(
                "LLM response was not a JSON object", error_class="invalid_response", transient=False
            )
        return payload

    def _retry_delay(self, failed_attempt: int) -> float:
        """Calculate bounded exponential backoff with injectable symmetric jitter."""
        policy = self.config.retry_policy
        base = min(policy.max_delay_seconds, policy.base_delay_seconds * (2 ** (failed_attempt - 1)))
        multiplier = 1 + policy.jitter_ratio * ((2 * self._random_value()) - 1)
        return min(policy.max_delay_seconds, max(0.0, base * multiplier))

    def _record_observation(self, observation: LLMCallObservation) -> None:
        """Store, log, and optionally emit only the sanitized observation payload."""
        self._thread_observation.pending = observation
        with self._observation_lock:
            self._observations.append(observation)
        payload = asdict(observation)
        LOGGER.info("llm_call_observation %s", json.dumps(payload, sort_keys=True))
        if self._observer is not None:
            try:
                self._observer(dict(payload))
            except Exception:
                LOGGER.warning("llm_call_observer_failed", exc_info=False)


def classify_url_error(error: urllib.error.URLError) -> tuple[bool, str]:
    """Classify a URL failure without copying its possibly sensitive reason text."""
    return classify_transport_reason(error.reason)


def classify_transport_reason(reason: object) -> tuple[bool, str]:
    """Retry only well-known timeout and interrupted-network exception classes."""
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True, "timeout"
    if isinstance(reason, ConnectionResetError):
        return True, "connection_reset"
    if isinstance(reason, ConnectionAbortedError):
        return True, "connection_aborted"
    if isinstance(reason, BrokenPipeError):
        return True, "broken_pipe"
    return False, "url_error" if isinstance(reason, str) else type(reason).__name__.casefold()


def build_llm_config(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout: int = 60,
    max_tokens: int = 800,
    temperature: float | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_seconds: float = 0.25,
    retry_max_delay_seconds: float = 2.0,
    retry_jitter_ratio: float = 0.2,
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
        retry_max_attempts: 单次逻辑调用允许的最大 HTTP 尝试次数
        retry_base_delay_seconds: 指数退避初始秒数
        retry_max_delay_seconds: 单次退避严格上限秒数
        retry_jitter_ratio: 对称抖动比例
        
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
        retry_policy=LLMRetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            max_delay_seconds=retry_max_delay_seconds,
            jitter_ratio=retry_jitter_ratio,
        ),
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
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMRequestError(
            "LLM response has no choices", error_class="invalid_response", transient=False
        )

    # 获取第一个选择的消息
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise LLMRequestError(
            "LLM response message was invalid", error_class="invalid_response", transient=False
        )
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
