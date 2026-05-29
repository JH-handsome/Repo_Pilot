"""
分词器模块
负责对代码和查询文本进行分词，支持中文和英文标识符
"""

import re
from dataclasses import dataclass


# 匹配标识符、数字和中文文本的正则表达式
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[\u4e00-\u9fff]+")
# 用于分割驼峰命名（PascalCase -> Pascal Case）
CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
# 用于分割驼峰命名（camelCase -> camel Case）
CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


@dataclass(frozen=True)
class CodeTokenizer:
    """
    代码分词器
    
    设计为轻量级且无依赖：
    - 保留完整标识符，如 `build_binary_tree`
    - 将 snake_case 分割为 `build`、`binary`、`tree`
    - 将 camelCase/PascalCase 分割为更小的单词
    - 保留数字
    - 将中文文本展开为短 n-gram，用于查询如 `二叉树`
    
    Attributes:
        keep_full_identifier: 是否保留完整的标识符
        chinese_ngram_min: 中文 n-gram 的最小长度
        chinese_ngram_max: 中文 n-gram 的最大长度
    """

    keep_full_identifier: bool = True
    chinese_ngram_min: int = 1
    chinese_ngram_max: int = 4

    def tokenize(self, text: str) -> list[str]:
        """
        对文本进行分词
        
        Args:
            text: 要分词的文本
            
        Returns:
            分词后的 token 列表
        """
        tokens: list[str] = []

        for raw_token in IDENTIFIER_RE.findall(text):
            if is_chinese(raw_token):
                tokens.extend(self.tokenize_chinese(raw_token))
                continue

            if raw_token.isdigit():
                tokens.append(raw_token)
                continue

            tokens.extend(self.tokenize_identifier(raw_token))

        return tokens

    def tokenize_identifier(self, identifier: str) -> list[str]:
        """
        对标识符进行分词
        
        Args:
            identifier: 标识符字符串
            
        Returns:
            分词后的 token 列表
        """
        # 移除下划线并转为小写
        normalized = identifier.strip("_").lower()
        if not normalized:
            return []

        tokens: list[str] = []
        if self.keep_full_identifier:
            tokens.append(normalized)

        for part in split_identifier(identifier):
            if part and part not in tokens:
                tokens.append(part)

        return tokens

    def tokenize_chinese(self, text: str) -> list[str]:
        """
        对中文文本进行 n-gram 分词
        
        Args:
            text: 中文文本
            
        Returns:
            n-gram 分词列表
        """
        tokens: list[str] = []
        max_n = min(self.chinese_ngram_max, len(text))

        for n in range(self.chinese_ngram_min, max_n + 1):
            for start in range(0, len(text) - n + 1):
                tokens.append(text[start : start + n])

        return tokens


def tokenize(text: str) -> list[str]:
    """便捷函数：使用默认分词器对文本进行分词"""
    return CodeTokenizer().tokenize(text)


def split_identifier(identifier: str) -> list[str]:
    """
    将标识符分割为更小的部分
    
    处理 snake_case 和 camelCase/PascalCase
    
    Args:
        identifier: 要分割的标识符
        
    Returns:
        分割后的部分列表（小写）
    """
    parts: list[str] = []

    # 先按 snake_case 分割
    for snake_part in identifier.strip("_").split("_"):
        if not snake_part:
            continue

        # 再将驼峰命名转换为空格分隔
        spaced = CAMEL_BOUNDARY_1.sub(r"\1 \2", snake_part)
        spaced = CAMEL_BOUNDARY_2.sub(r"\1 \2", spaced)
        parts.extend(piece.lower() for piece in spaced.split() if piece)

    return parts


def is_chinese(text: str) -> bool:
    """
    判断文本是否全为中文
    
    Args:
        text: 要判断的文本
        
    Returns:
        如果文本全部由中文字符组成则返回 True
    """
    return all("\u4e00" <= char <= "\u9fff" for char in text)
