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
from coding_rag.env_loader import ensure_dotenv, load_dotenv
from coding_rag.file_loader import load_python_files
from coding_rag.result_filter import filter_recalled_results
from main import run_answer_generator
from scripts.retrieval_eval import (
    aggregate_stage_diagnostics,
    analyze_bad_case,
    build_eval_generator,
    build_stage_file_record,
    default_recall_max_results,
    generate_eval_answer,
    load_evalset,
    mrr_and_recall_at_k,
    optimize,
    pipeline,
    stage_files_path_for_trace,
    stage_summary_path_for_trace,
    write_json,
    write_jsonl,
    render_stage_diagnostics_summary,
)
from rag.trace import build_retrieval_trace, render_trace_report


class RepoPilotUI:
    """简单的桌面 GUI，便于交互式使用提示词和 RAG 检索流程。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RepoPilot Prompt UI")
        self.root.geometry("1100x760")

        self.repo_path_var = tk.StringVar(value=".")
        self.query_var = tk.StringVar()
        self.top_k_var = tk.IntVar(value=5)
        self.chunk_size_var = tk.IntVar(value=40)
        self.overlap_var = tk.IntVar(value=5)
        self.recall_window_var = tk.IntVar(value=2)
        self.use_llm_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="judge")
        self.provider_var = tk.StringVar(value="deepseek")
        self.status_var = tk.StringVar(value="就绪")
        self.evalset_path_var = tk.StringVar(value="datasets/eval/sample_evalset.json")
        self.last_trace_path: Path | None = None
        self.last_stage_files_path: Path | None = None
        self.last_stage_summary_path: Path | None = None

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
        ttk.Label(llm_bar, text="提供商").grid(row=0, column=3, padx=(10, 3))
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
                max_results=default_recall_max_results(self.top_k_var.get(), self.recall_window_var.get()),
            )
            results = filter_recalled_results(query, recalled, retriever, final_k=self.top_k_var.get())
            trajectory = build_retrieval_trace(
                query=query,
                seed_results=seed,
                recalled_results=recalled,
                final_results=results,
                params={
                    "repo_path": repo_path,
                    "top_k": self.top_k_var.get(),
                    "chunk_size": self.chunk_size_var.get(),
                    "overlap": self.overlap_var.get(),
                    "recall_window": self.recall_window_var.get(),
                },
            )

            lines = [
                f"Hybrid 种子: {len(seed)} | 召回代码块: {len(recalled)} | 最终代码块: {len(results)}",
                "",
                render_trace_report(trajectory, limit=8),
                "",
                "=" * 80,
                "最终代码块",
                "-" * 80,
                self._render_results(results),
            ]

            if self.use_llm_var.get():
                ensure_dotenv()
                load_dotenv()
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
        selected = filedialog.askopenfilename(
            title="选择评测集",
            filetypes=[("评测集", "*.json *.jsonl"), ("JSON", "*.json"), ("JSONL", "*.jsonl")],
        )
        if selected:
            self.evalset_path_var.set(selected)

    def _run_eval_async(self) -> None:
        threading.Thread(target=self._run_eval, daemon=True).start()

    def _run_eval(self) -> None:
        self._set_status("评测中...")
        try:
            cases = load_evalset(self.evalset_path_var.get())
            out_rows = []
            stage_file_rows = []
            path_roots = [
                Path(self.repo_path_var.get().strip()).resolve(),
                Path(self.evalset_path_var.get()).resolve().parent,
            ]
            generator = None
            llm_setup_error = None
            if self.use_llm_var.get():
                ensure_dotenv()
                load_dotenv()
                try:
                    generator = build_eval_generator(self._build_llm_args())
                except ValueError as error:
                    llm_setup_error = f"LLM 配置错误: {error}"

            for case in cases:
                seed, recalled, final = pipeline(self.repo_path_var.get().strip(), case.query, self.top_k_var.get(), self.chunk_size_var.get(), self.overlap_var.get(), self.recall_window_var.get())
                rr, rec, hits = mrr_and_recall_at_k(final, case.relevant, self.top_k_var.get(), path_roots)
                bad = analyze_bad_case(seed, recalled, final, case.relevant, self.top_k_var.get(), path_roots)
                trajectory = build_retrieval_trace(
                    query=case.query,
                    seed_results=seed,
                    recalled_results=recalled,
                    final_results=final,
                    params={
                        "repo_path": self.repo_path_var.get().strip(),
                        "top_k": self.top_k_var.get(),
                        "chunk_size": self.chunk_size_var.get(),
                        "overlap": self.overlap_var.get(),
                        "recall_window": self.recall_window_var.get(),
                    },
                )
                row = {
                    "id": case.id,
                    "query": case.query,
                    "metrics": {"rr": rr, "recall_at_k": rec, "hit_count": hits},
                    "bad_case": bad,
                    "trajectory": trajectory,
                    "seed": trajectory["stages"]["initial_search"],
                    "recalled": trajectory["stages"]["neighbor_recall"],
                    "final": trajectory["stages"]["final_filter"],
                    "context": trajectory["stages"]["context_compaction"],
                }
                if self.use_llm_var.get():
                    row["llm"] = generate_eval_answer(generator, llm_setup_error, case.query, final)
                out_rows.append(row)
                stage_file_rows.append(build_stage_file_record(case, seed, recalled, final, self.top_k_var.get(), path_roots))

            trace_path = Path("artifacts") / "ui_retrieval_trace.jsonl"
            stage_files_path = stage_files_path_for_trace(trace_path)
            stage_summary_path = stage_summary_path_for_trace(trace_path)
            stage_summary = aggregate_stage_diagnostics(stage_file_rows, out_rows)
            write_jsonl(trace_path, out_rows)
            write_jsonl(stage_files_path, stage_file_rows)
            write_json(stage_summary_path, stage_summary)
            self.last_trace_path = trace_path
            self.last_stage_files_path = stage_files_path
            self.last_stage_summary_path = stage_summary_path
            bad_cnt = sum(1 for r in out_rows if not r["bad_case"]["final_hit"])
            self._set_output(self._render_eval_output(out_rows, bad_cnt, trace_path, stage_files_path, stage_summary_path, stage_summary))
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
            cases = load_evalset(self.evalset_path_var.get())
            class Args: pass
            args = Args()
            args.repo_path = self.repo_path_var.get().strip()
            args.top_k = self.top_k_var.get()
            report = optimize(args, cases)
            self._set_output(report)
            self._set_status("调参完成")
        except Exception as exc:
            self._set_output(f"调参失败: {exc}")
            self._set_status("调参失败")

    def _build_llm_args(self):
        return type("Args", (), {
            "mode": self.mode_var.get(),
            "llm_provider": self.provider_var.get(),
            "llm_model": None,
            "llm_base_url": None,
            "llm_api_key_env": None,
            "llm_timeout": 60,
            "llm_max_tokens": 1200,
            "llm_temperature": None,
            "llm_context_chars": 12000,
        })

    @staticmethod
    def _render_eval_output(
        rows: list[dict],
        bad_count: int,
        trace_path: Path,
        stage_files_path: Path | None = None,
        stage_summary_path: Path | None = None,
        stage_summary: dict | None = None,
    ) -> str:
        lines = [
            f"评测完成: {len(rows)} cases",
            f"Bad cases: {bad_count}",
            f"Trace: {trace_path}",
        ]
        if stage_files_path:
            lines.append(f"Stage files: {stage_files_path}")
        if stage_summary_path:
            lines.append(f"Stage summary: {stage_summary_path}")
        if stage_summary:
            lines.extend(["", render_stage_diagnostics_summary(stage_summary)])
        for row in rows:
            metrics = row["metrics"]
            lines.extend([
                "",
                "=" * 80,
                f"[{row['id']}] {row['query']}",
                f"MRR: {metrics['rr']:.4f} | Recall@K: {metrics['recall_at_k']:.4f} | Hits: {metrics['hit_count']}",
            ])

            llm = row.get("llm")
            if not llm:
                continue

            lines.extend(["", "LLM 输出:", "-" * 80])
            if llm.get("error"):
                lines.append(f"LLM 调用失败: {llm['error']}")
            else:
                lines.append((llm.get("answer") or "").strip() or "LLM 输出为空。")

        return "\n".join(lines)

    @staticmethod
    def _render_results(results: list[SearchResult]) -> str:
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            blocks.extend(
                [
                    "=" * 80,
                    f"结果 {index} | 分数: {result.score:.4f}",
                    f"来源标签: {result.source}",
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
