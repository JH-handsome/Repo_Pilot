"""Code/query tokenizer."""

import re
from dataclasses import dataclass


# 匹配标识符、数字和中文文本的正则表达式
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[\u4e00-\u9fff]+")
# 用于分割驼峰命名（PascalCase -> Pascal Case）
CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
# 用于分割驼峰命名（camelCase -> camel Case）
CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


CHINESE_CODE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "检索": ("retrieval", "search"),
    "搜索": ("search",),
    "评测": ("eval", "evaluation"),
    "评价": ("eval", "evaluation"),
    "召回": ("recall",),
    "结果": ("result",),
    "文件": ("file",),
    "正确": ("relevant", "correct"),
    "相关": ("relevant",),
    "过滤": ("filter",),
    "筛选": ("filter",),
    "排序": ("rank", "sort"),
    "分词": ("tokenize", "tokenizer"),
    "切片": ("chunk", "split"),
    "代码块": ("chunk",),
    "大模型": ("llm",),
    "调用": ("call", "client"),
    "被调用": ("call", "caller", "callee"),
    "导入": ("import", "from"),
    "引用": ("reference", "import"),
    "接口": ("api",),
    "函数": ("function", "def"),
    "方法": ("method", "function", "def"),
    "类": ("class",),
    "签名": ("signature",),
    "继承": ("inherit", "base", "class"),
    "模块": ("module",),
    "路径": ("path", "file"),
    "文件树": ("filetree", "tree", "path"),
    "索引": ("index",),
    "测试": ("test", "testing"),
    "单测": ("test", "unittest"),
    "断言": ("assert",),
}


@dataclass(frozen=True)
class CodeTokenizer:
    """Lightweight tokenizer."""

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

        for term, synonyms in CHINESE_CODE_SYNONYMS.items():
            if term in text:
                tokens.extend(synonym for synonym in synonyms if synonym not in tokens)

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
