"""RepoPilot 桌面前端窗口（Tkinter）。"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from coding_rag.bm25_retriever import BM25Retriever, SearchResult
from coding_rag.code_splitter import split_python_files
from coding_rag.context_recaller import expand_with_neighbor_chunks
from coding_rag.file_loader import load_python_files
from coding_rag.result_filter import filter_recalled_results
from main import run_answer_generator
from scripts.retrieval_eval import EvalCase, optimize, pipeline, serialize_result, mrr_and_recall_at_k, analyze_bad_case


class RepoPilotUI:
    """简单的桌面 GUI，便于交互式使用提示词和 RAG 检索流程。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RepoPilot Prompt UI")
        self.root.geometry("1100x760")

        self.repo_path_var = tk.StringVar(value="datasets")
        self.query_var = tk.StringVar()
        self.top_k_var = tk.IntVar(value=5)
        self.chunk_size_var = tk.IntVar(value=40)
        self.overlap_var = tk.IntVar(value=5)
        self.recall_window_var = tk.IntVar(value=1)
        self.use_llm_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="judge")
        self.provider_var = tk.StringVar(value="deepseek")
        self.status_var = tk.StringVar(value="就绪")
        self.evalset_path_var = tk.StringVar(value="datasets/eval/sample_evalset.json")
        self.last_trace_path: Path | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="仓库路径").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(controls, textvariable=self.repo_path_var, width=72).grid(row=0, column=1, padx=8)
        ttk.Button(controls, text="选择目录", command=self._pick_repo_path).grid(row=0, column=2)

        ttk.Label(controls, text="问题 / 提示词").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(controls, textvariable=self.query_var, width=72).grid(row=1, column=1, padx=8, pady=(8, 0))

        numeric = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        numeric.pack(fill=tk.X)
        ttk.Label(numeric, text="top-k").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(numeric, from_=1, to=100, textvariable=self.top_k_var, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(numeric, text="chunk-size").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(numeric, from_=10, to=300, textvariable=self.chunk_size_var, width=6).grid(row=0, column=3, padx=5)
        ttk.Label(numeric, text="overlap").grid(row=0, column=4, sticky=tk.W)
        ttk.Spinbox(numeric, from_=0, to=100, textvariable=self.overlap_var, width=6).grid(row=0, column=5, padx=5)
        ttk.Label(numeric, text="recall-window").grid(row=0, column=6, sticky=tk.W)
        ttk.Spinbox(numeric, from_=0, to=10, textvariable=self.recall_window_var, width=6).grid(row=0, column=7, padx=5)

        llm_bar = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        llm_bar.pack(fill=tk.X)
        ttk.Checkbutton(llm_bar, text="启用 LLM 生成", variable=self.use_llm_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(llm_bar, text="模式").grid(row=0, column=1, padx=(10, 3))
        ttk.Combobox(
            llm_bar,
            textvariable=self.mode_var,
            values=["judge", "code-understand", "code-generate", "leetcode", "api"],
            state="readonly",
            width=18,
        ).grid(row=0, column=2)
        ttk.Label(llm_bar, text="Provider").grid(row=0, column=3, padx=(10, 3))
        ttk.Combobox(
            llm_bar,
            textvariable=self.provider_var,
            values=["deepseek", "qwen", "kimi", "zhipu", "custom"],
            state="readonly",
            width=12,
        ).grid(row=0, column=4)

        run_bar = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        run_bar.pack(fill=tk.X)
        ttk.Button(run_bar, text="开始检索", command=self._run_async).pack(side=tk.LEFT)
        ttk.Button(run_bar, text="开始评测", command=self._run_eval_async).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(run_bar, text="导出 Trace", command=self._export_last_trace).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(run_bar, text="Bad Case 面板", command=self._show_bad_cases).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(run_bar, text="自动调参", command=self._optimize_async).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(run_bar, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        eval_bar = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        eval_bar.pack(fill=tk.X)
        ttk.Label(eval_bar, text="评测集").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(eval_bar, textvariable=self.evalset_path_var, width=72).grid(row=0, column=1, padx=8)
        ttk.Button(eval_bar, text="选择评测集", command=self._pick_evalset).grid(row=0, column=2)

        output_frame = ttk.Frame(self.root, padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(output_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _pick_repo_path(self) -> None:
        selected = filedialog.askdirectory(title="选择要搜索的仓库")
        if selected:
            self.repo_path_var.set(selected)

    def _run_async(self) -> None:
        if not self.query_var.get().strip():
            messagebox.showwarning("输入不完整", "请先输入问题或提示词。")
            return

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self) -> None:
        self._set_status("处理中...")
        self._set_output("")

        try:
            repo_path = self.repo_path_var.get().strip()
            query = self.query_var.get().strip()
            python_files = load_python_files(repo_path)
            chunks = split_python_files(
                python_files,
                chunk_size=self.chunk_size_var.get(),
                overlap=self.overlap_var.get(),
            )
            if not chunks:
                self._set_output("未找到 Python 代码块，请检查路径是否正确。")
                self._set_status("未找到代码")
                return

            retriever = BM25Retriever(chunks)
            seed = retriever.search(query, top_k=self.top_k_var.get())
            recalled = expand_with_neighbor_chunks(
                chunks=chunks,
                seed_results=seed,
                window=self.recall_window_var.get(),
                max_results=20,
            )
            results = filter_recalled_results(query, recalled, retriever, final_k=self.top_k_var.get())

            lines = [
                f"BM25 种子: {len(seed)} | 召回代码块: {len(recalled)} | 最终代码块: {len(results)}",
                "",
                self._render_results(results),
            ]

            if self.use_llm_var.get():
                args = type("Args", (), {
                    "query": query,
                    "mode": self.mode_var.get(),
                    "llm_provider": self.provider_var.get(),
                    "llm_model": None,
                    "llm_base_url": None,
                    "llm_api_key_env": None,
                    "llm_timeout": 60,
                    "llm_max_tokens": 2000,
                    "llm_temperature": None,
                    "llm_context_chars": 12000,
                })
                answer = run_answer_generator(args, results)
                lines.extend(["", "=" * 80, f"LLM 结果（{self.mode_var.get()}）", "-" * 80, answer.strip()])

            self._set_output("\n".join(lines))
            self._set_status("完成")
        except Exception as exc:
            self._set_output(f"执行失败: {exc}")
            self._set_status("失败")


    def _pick_evalset(self) -> None:
        selected = filedialog.askopenfilename(title="选择评测集 JSON", filetypes=[("JSON", "*.json")])
        if selected:
            self.evalset_path_var.set(selected)

    def _run_eval_async(self) -> None:
        threading.Thread(target=self._run_eval, daemon=True).start()

    def _run_eval(self) -> None:
        self._set_status("评测中...")
        try:
            evalset = json.loads(Path(self.evalset_path_var.get()).read_text(encoding="utf-8"))
            cases = [EvalCase(id=i["id"], query=i["query"], relevant=i["relevant"]) for i in evalset]
            out_rows = []
            for case in cases:
                seed, recalled, final = pipeline(self.repo_path_var.get().strip(), case.query, self.top_k_var.get(), self.chunk_size_var.get(), self.overlap_var.get(), self.recall_window_var.get())
                rr, rec, hits = mrr_and_recall_at_k(final, case.relevant, self.top_k_var.get())
                bad = analyze_bad_case(seed, recalled, final, case.relevant, self.top_k_var.get())
                out_rows.append({"id": case.id, "query": case.query, "metrics": {"rr": rr, "recall_at_k": rec, "hit_count": hits}, "bad_case": bad, "seed": [serialize_result(x) for x in seed], "recalled": [serialize_result(x) for x in recalled], "final": [serialize_result(x) for x in final]})

            trace_path = Path("artifacts") / "ui_retrieval_trace.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows), encoding="utf-8")
            self.last_trace_path = trace_path
            bad_cnt = sum(1 for r in out_rows if not r["bad_case"]["final_hit"])
            self._set_output(f"评测完成: {len(out_rows)} cases\nBad cases: {bad_cnt}\nTrace: {trace_path}")
            self._set_status("评测完成")
        except Exception as exc:
            self._set_output(f"评测失败: {exc}")
            self._set_status("评测失败")

    def _export_last_trace(self) -> None:
        if not self.last_trace_path or not self.last_trace_path.exists():
            messagebox.showwarning("无 Trace", "请先点击“开始评测”生成 Trace。")
            return
        target = filedialog.asksaveasfilename(title="保存 Trace", defaultextension=".jsonl", filetypes=[("JSONL", "*.jsonl")])
        if not target:
            return
        Path(target).write_text(self.last_trace_path.read_text(encoding="utf-8"), encoding="utf-8")
        self._set_status("Trace 已导出")

    def _show_bad_cases(self) -> None:
        if not self.last_trace_path or not self.last_trace_path.exists():
            messagebox.showwarning("无数据", "请先点击“开始评测”。")
            return
        rows = [json.loads(line) for line in self.last_trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        bad = [r for r in rows if not r["bad_case"]["final_hit"]]
        if not bad:
            self._set_output("没有 bad case。")
            return
        blocks = []
        for r in bad:
            blocks.append(f"[{r['id']}] {r['query']}\n- reasons: {', '.join(r['bad_case']['reasons'])}\n- prompt_hints: {'; '.join(r['bad_case']['prompt_hints'])}")
        self._set_output("\n\n".join(blocks))
        self._set_status("Bad Case 展示完成")

    def _optimize_async(self) -> None:
        threading.Thread(target=self._run_optimize, daemon=True).start()

    def _run_optimize(self) -> None:
        self._set_status("调参中...")
        try:
            evalset = json.loads(Path(self.evalset_path_var.get()).read_text(encoding="utf-8"))
            cases = [EvalCase(id=i["id"], query=i["query"], relevant=i["relevant"]) for i in evalset]
            class Args: pass
            args = Args()
            args.repo_path = self.repo_path_var.get().strip()
            args.top_k = self.top_k_var.get()
            # optimize prints to stdout; give user guidance in UI
            optimize(args, cases)
            self._set_output("自动调参已执行（结果已输出到终端日志）。")
            self._set_status("调参完成")
        except Exception as exc:
            self._set_output(f"调参失败: {exc}")
            self._set_status("调参失败")

    @staticmethod
    def _render_results(results: list[SearchResult]) -> str:
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            blocks.extend(
                [
                    "=" * 80,
                    f"结果 {index} | 分数: {result.score:.4f}",
                    f"Source: {result.source}",
                    f"来源: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
                    "-" * 80,
                    chunk.text.rstrip(),
                ]
            )
        return "\n".join(blocks)

    def _set_output(self, text: str) -> None:
        self.output.after(0, self._replace_output, text)

    def _replace_output(self, text: str) -> None:
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)


def main() -> None:
    app = tk.Tk()
    RepoPilotUI(app)
    app.mainloop()


if __name__ == "__main__":
    if not Path("main.py").exists():
        raise SystemExit("请在项目根目录运行: python web_ui.py")
    main()
