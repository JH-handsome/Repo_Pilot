"""LLM-based review for ReAct Agent plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from coding_rag.agent.planner import AgentPlanRun, format_tool_specs


class ChatClient(Protocol):
    """聊天客户端协议。"""
    def complete(self, messages: list[dict[str, str]]) -> str:
        """完成消息对话。"""
        ...


@dataclass(frozen=True)
class AgentPlanReview:
    """智能体计划审查结果。"""
    verdict: str
    score: float
    issues: list[str]
    suggestions: list[str]
    raw_text: str

    @property
    def passed(self) -> bool:
        """审查是否通过。"""
        return self.verdict == "pass" and self.score >= 0.7


def review_agent_plan(run: AgentPlanRun, client: ChatClient) -> AgentPlanReview:
    """Call an LLM to review whether a generated Agent plan is reasonable."""
    messages = build_agent_plan_review_messages(run)
    raw_text = client.complete(messages).strip()
    return parse_agent_plan_review(raw_text)


def build_agent_plan_review_messages(run: AgentPlanRun) -> list[dict[str, str]]:
    """构造审查提示消息。"""
    system_prompt = """你是 RepoPilot 的 Agent 计划审查员。你的任务是判断一个 ReAct Agent 工作流计划是否合理、可执行、可审查。
审查规则：
1. 只审查计划质量，不补写新的完整计划。
2. 检查计划是否基于任务、检索上下文和工具接口。
3. 检查 Action 是否只使用给定工具名。
4. 检查每一步是否包含目的、工具输入、预期观察结果和失败备选动作。
5. 检查是否包含候选文件、验证命令、风险和交付物。
6. 如果计划声称已经执行工具或已经修改代码，应判为 fail。
必须只输出 JSON，不要输出 markdown。格式：
{
  "verdict": "pass" | "fail",
  "score": 0.0,
  "issues": ["..."],
  "suggestions": ["..."]
}
"""
    user_prompt = f"""## 用户任务
{run.task}

## 可用工具接口
{format_tool_specs(run.tools)}

## 检索摘要
- seed_count: {len(run.seed_results)}
- recalled_count: {len(run.recalled_results)}
- final_count: {len(run.final_results)}
- candidate_files: {", ".join(candidate_files(run))}

## 待审查计划
{run.plan_text or "(no plan text)"}

请按系统要求输出 JSON 审查结果。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_agent_plan_review(raw_text: str) -> AgentPlanReview:
    """解析审查器响应。"""
    payload = parse_json_object(raw_text)
    verdict = str(payload.get("verdict", "fail")).casefold()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    return AgentPlanReview(
        verdict=verdict,
        score=clamp_score(payload.get("score", 0.0)),
        issues=string_list(payload.get("issues")),
        suggestions=string_list(payload.get("suggestions")),
        raw_text=raw_text,
    )


def parse_json_object(text: str) -> dict:
    """从响应文本中提取 JSON 对象。"""
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    inline = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if inline:
        try:
            payload = json.loads(inline.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    return {"verdict": "fail", "score": 0.0, "issues": ["LLM review did not return valid JSON"], "suggestions": []}


def clamp_score(value) -> float:
    """将评分限制在 [0.0, 1.0] 范围内。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))


def string_list(value) -> list[str]:
    """将列表中每个元素转为 str，非列表时返回 []。"""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def candidate_files(run: AgentPlanRun) -> list[str]:
    """从 run.final_results 的检索块中收集唯一文件路径，而非从计划步骤中收集。"""
    files: list[str] = []
    for result in run.final_results:
        path = str(result.chunk.file_path)
        if path not in files:
            files.append(path)
    return files


def render_agent_plan_review(review: AgentPlanReview) -> str:
    """将审查结论、评分、通过状态、问题与建议格式化为 Markdown 风格的审查文本。"""
    lines = [
        "## Agent 计划审查",
        f"- verdict: {review.verdict}",
        f"- score: {review.score:.2f}",
        f"- passed: {review.passed}",
        "- issues:",
    ]
    lines.extend(f"  - {issue}" for issue in review.issues or ["(none)"])
    lines.append("- suggestions:")
    lines.extend(f"  - {suggestion}" for suggestion in review.suggestions or ["(none)"])
    return "\n".join(lines)
