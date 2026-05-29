# 旧版原型代码

这个目录保存项目早期的实验实现，方便对照学习。当前主流程已经统一到：

- `coding_rag/`: 文件读取、代码切片、BM25 检索、上下文召回和结果过滤
- `rag/`: prompt 拼装和 LLM 答案生成
- `scripts/retrieval_eval.py`: 离线评测和 bad case 分析

新功能请优先改 `coding_rag/` 和 `rag/`，不要在这里继续扩展旧实现。
