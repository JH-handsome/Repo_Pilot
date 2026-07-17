"""
对项目中选定的源码文件进行文档字符串覆盖检查的单元测试。
"""

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_FILES = ("main.py", "frontend.py", "web_ui.py", "leetcode_types.py")


def discover_target_files():
    """发现需要检查文档字符串的目标 .py 文件。

    返回顶层指定的 Python 文件，以及 coding_rag、scripts 两个子目录中
    递归搜索到的所有 .py 文件，按相对项目根目录的 POSIX 路径排序并去重后的列表。
    """
    rel_set = set()
    proj = PROJECT_ROOT.resolve()

    # 顶层文件
    for fname in TOP_LEVEL_FILES:
        target = (PROJECT_ROOT / fname).resolve()
        try:
            rel = target.relative_to(proj).as_posix()
        except ValueError:
            rel = fname
        rel_set.add(rel)

    # coding_rag 与 scripts 子树
    for sub in ("coding_rag", "scripts"):
        sub_dir = PROJECT_ROOT / sub
        if sub_dir.is_dir():
            for py in sorted(sub_dir.rglob("*.py")):
                target = py.resolve()
                try:
                    rel = target.relative_to(proj).as_posix()
                except ValueError:
                    continue
                rel_set.add(rel)

    return [Path(p) for p in sorted(rel_set)]


class DocstringVisitor(ast.NodeVisitor):
    """AST 节点访问器，用于收集缺少文档字符串的模块、类和函数定义。"""

    def __init__(self, relative_path):
        """初始化访问器。

        Args:
            relative_path: 当前源码文件相对项目根目录的路径字符串，
                           用于拼装错误消息。
        """
        self.path = relative_path
        self.stack = []
        self.issues = []

    def visit_Module(self, node):
        """检查模块级文档字符串。

        若模块缺少非空文档字符串，则记录“路径:1:<module>”；
        随后继续遍历子节点。
        """
        raw_doc = ast.get_docstring(node, clean=False)
        if raw_doc is None or raw_doc.strip() == "":
            self.issues.append(f"{self.path}:1:<module>")
        self.generic_visit(node)

    def _check_def(self, node):
        """私有辅助方法：检查类或函数定义的文档字符串。

        若节点缺少非空文档字符串，则追加“路径:行号:限定名”到 issues；
        随后将节点名入栈、遍历子节点、最后出栈，以支持嵌套定义的递归检查。

        Args:
            node: 待检查的 ClassDef、FunctionDef 或 AsyncFunctionDef 节点。
        """
        qualified = ".".join(self.stack + [node.name])
        raw_doc = ast.get_docstring(node, clean=False)
        if raw_doc is None or raw_doc.strip() == "":
            self.issues.append(f"{self.path}:{node.lineno}:{qualified}")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node):
        """检查类定义的文档字符串。"""
        self._check_def(node)

    def visit_FunctionDef(self, node):
        """检查函数定义的文档字符串。"""
        self._check_def(node)

    def visit_AsyncFunctionDef(self, node):
        """检查异步函数定义的文档字符串。"""
        self._check_def(node)


class DocstringCoverageTest(unittest.TestCase):
    """测试选定的源码文件中是否都有必要的文档字符串。"""

    def test_selected_source_docstrings(self):
        """遍历所有目标 .py 文件，检查模块、类、函数的文档字符串覆盖情况。

        对每个文件：
        - 若文件不存在，记录 `<missing-file>`；
        - 若文件存在但无法解析为合法 Python 语法，记录 `<parse-error>`；
        - 否则用 DocstringVisitor 遍历 AST，收集所有缺失文档字符串的位置。
        最后断言 issues 列表为空，否则在一次失败中列出全部缺失项。
        """
        issues = []
        for target in discover_target_files():
            full_path = PROJECT_ROOT / target
            rel = target.as_posix()

            if not full_path.exists():
                issues.append(f"{rel}:1:<missing-file>")
                continue

            try:
                source = full_path.read_text(encoding="utf-8-sig")
                tree = ast.parse(source, filename=str(full_path))
            except SyntaxError as exc:
                lineno = exc.lineno if exc.lineno is not None else 1
                issues.append(f"{rel}:{lineno}:<parse-error>")
                continue

            visitor = DocstringVisitor(rel)
            visitor.visit(tree)
            issues.extend(visitor.issues)

        self.assertFalse(
            issues,
            "缺少源码说明或解析失败：\n" + "\n".join(issues),
        )


if __name__ == "__main__":
    unittest.main()
