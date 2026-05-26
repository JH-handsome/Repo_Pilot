# RAG Corpus

Recommended starter corpus:

- Repository: `https://github.com/cnkyrpsgl/leetcode`
- Reason: MIT license, Python-only LeetCode solutions, easy for `file_loader` to read
- Local target path: `datasets/leetcode-python`

This folder also includes `leetcode_reference/leetcode_formats.py`, which
defines common LeetCode structures such as `ListNode`, `TreeNode`,
`RandomListNode`, `GraphNode`, and `NaryNode`.

For standalone local execution of copied LeetCode solution files, use the root
`leetcode_types.py` shim and patch imports:

```powershell
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

Download:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_leetcode_corpus.ps1
```

Search after download:

```powershell
python main.py datasets "two sum hash map" --top-k 5
python main.py datasets "binary tree level order null TreeNode" --top-k 5
python main.py datasets "linked list ListNode head next" --top-k 5
```
