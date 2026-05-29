# 数据集目录

这里放检索语料和评测集。

## 推荐语料

建议从这个仓库开始：

- 仓库：`https://github.com/cnkyrpsgl/leetcode`
- 原因：MIT 协议、Python-only、文件结构简单，适合 `file_loader` 读取
- 本地目录：`datasets/leetcode-python`

下载：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_leetcode_corpus.ps1
```

下载后检索：

```powershell
python main.py datasets "two sum hash map" --top-k 5
python main.py datasets "binary tree level order null TreeNode" --top-k 5
python main.py datasets "linked list ListNode head next" --top-k 5
```

## LeetCode 类型参考

`leetcode_reference/leetcode_formats.py` 定义了常见 LeetCode 结构，例如：

- `ListNode`
- `TreeNode`
- `RandomListNode`
- `GraphNode`
- `NaryNode`

如果要把下载下来的 LeetCode 代码单独本地运行，可以用项目根目录的 `leetcode_types.py` 做类型补丁：

```powershell
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

## 评测集

`eval/` 下保存检索评测数据：

- `sample_evalset.json`: 小型示例评测集
- `trace.jsonl`: 20 条中文问题测试集
