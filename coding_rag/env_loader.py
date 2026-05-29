"""本地 .env 配置加载模块。

项目不依赖 python-dotenv，这里只实现当前需要的最小能力：
- 如果 `.env` 不存在，生成一份安全模板；
- 读取 `KEY=VALUE` 或 `export KEY=VALUE` 格式；
- 默认不覆盖命令行里已经设置好的环境变量。
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DOTENV = {
    "DEEPSEEK_API_KEY": "",
    "LLM_BASE_URL": "https://api.deepseek.com",
    "LLM_MODEL": "deepseek-v4-flash",
    "LLM_API_KEY_ENV": "DEEPSEEK_API_KEY",
}


def ensure_dotenv(
    path: str | Path = ".env",
    defaults: dict[str, str] | None = None,
) -> bool:
    """当 `.env` 不存在时创建默认模板。

    返回 True 表示本次创建了文件；返回 False 表示文件已存在。
    """
    env_path = Path(path)
    if env_path.exists():
        return False

    values = defaults or DEFAULT_DOTENV
    lines = ["# 本地 LLM 配置文件，已被 .gitignore 忽略，请不要提交真实密钥。"]
    lines.extend(f"{key}={value}" for key, value in values.items())
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def load_dotenv(path: str | Path = ".env", override: bool = False) -> bool:
    """把 `.env` 中的键值对加载到 `os.environ`。

    默认让 shell 中已有的变量优先，避免本地文件意外覆盖临时密钥。
    """
    env_path = Path(path)
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue

        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value

    return True


def parse_env_line(line: str) -> tuple[str, str] | None:
    """解析一行 `.env` 内容，忽略空行和注释。"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    return key, unquote_env_value(value.strip())


def unquote_env_value(value: str) -> str:
    """去掉单引号或双引号包裹，并处理双引号内的转义字符。"""
    if len(value) < 2:
        return value

    quote = value[0]
    if quote not in {"'", '"'} or value[-1] != quote:
        return value

    unquoted = value[1:-1]
    if quote == '"':
        return unquoted.encode("utf-8").decode("unicode_escape")
    return unquoted
