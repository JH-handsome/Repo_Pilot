"""定义 Learning Mode 的严格结构化输出和公开运行结果。"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


STEP_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
MAX_VERIFICATION_ITEMS = 20
WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
LearningErrorCode = Literal[
    "repository_error",
    "empty_repository",
    "model_error",
    "invalid_model_output",
    "invalid_evidence",
]
LearningSessionAction = Literal["start", "submit", "reflect"]
LearningSessionPhase = Literal[
    "awaiting_submission",
    "needs_revision",
    "awaiting_reflection",
    "completed",
]
LearningSessionErrorCode = Literal[
    "invalid_request",
    "invalid_session",
    "action_phase_mismatch",
    "planning_failed",
    "model_error",
    "invalid_model_output",
    "session_error",
]
NextLearningAction = Literal["submit", "reflect", "none"]
NonEmptyText = Annotated[str, Field(min_length=1)]
ReviewText = Annotated[str, Field(min_length=1, max_length=1_000)]


class StrictLearningModel(BaseModel):
    """拒绝额外字段和隐式类型转换的 Learning Mode 模型基类。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class EvidenceRef(StrictLearningModel):
    """把一个明确事实绑定到本次只读检索已经观察到的源码范围。"""

    claim: str = Field(min_length=1)
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        """规范仓库相对路径，并拒绝绝对路径与父目录穿越。"""
        return normalize_repo_relative_path(value)

    @model_validator(mode="after")
    def validate_line_range(self) -> "EvidenceRef":
        """保证证据范围的结束行不早于开始行。"""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ProjectComponent(StrictLearningModel):
    """项目中的一个可解释组件及其源码依据。"""

    name: str = Field(min_length=1)
    responsibility: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)


class ProjectProfile(StrictLearningModel):
    """面向初学者的项目目标、技术栈、入口和组件画像。"""

    project_name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tech_stack: list[NonEmptyText] = Field(min_length=1)
    prerequisites: list[NonEmptyText] = Field(min_length=1)
    entry_points: list[EvidenceRef] = Field(min_length=1)
    components: list[ProjectComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_components(self) -> "ProjectProfile":
        """拒绝大小写不同但含义相同的重复组件名称。"""
        names = [component.name.casefold() for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        return self


class ReproductionStep(StrictLearningModel):
    """从空目录复现项目时的一步教学任务。"""

    step_id: str = Field(min_length=1, pattern=STEP_ID_PATTERN)
    title: str = Field(min_length=1)
    learning_goal: str = Field(min_length=1)
    depends_on: list[str]
    files_to_create: list[str] = Field(min_length=1)
    tasks: list[NonEmptyText] = Field(min_length=1)
    why: str = Field(min_length=1)
    benefits: list[NonEmptyText] = Field(min_length=1)
    verification: list[NonEmptyText] = Field(
        min_length=1,
        max_length=MAX_VERIFICATION_ITEMS,
    )
    evidence: list[EvidenceRef] = Field(min_length=1)
    common_pitfalls: list[NonEmptyText] = Field(min_length=1)
    optimization_question: str = Field(min_length=2)

    @field_validator("depends_on")
    @classmethod
    def validate_dependency_shape(cls, values: list[str]) -> list[str]:
        """拒绝空依赖、重复依赖以及不稳定的步骤标识。"""
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            if not item or re.fullmatch(STEP_ID_PATTERN, item) is None:
                raise ValueError("depends_on entries must be valid step ids")
            if item in normalized:
                raise ValueError("depends_on entries must be unique")
            normalized.append(item)
        return normalized

    @field_validator("files_to_create")
    @classmethod
    def normalize_created_files(cls, values: list[str]) -> list[str]:
        """把待创建文件限制为新项目内的相对路径。"""
        normalized = [normalize_repo_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("files_to_create entries must be unique")
        return normalized

    @field_validator("optimization_question")
    @classmethod
    def validate_optimization_question(cls, value: str) -> str:
        """确保优化思考字段确实以问题形式呈现。"""
        if not value.endswith(("?", "？")) or not value.rstrip("?？").strip():
            raise ValueError("optimization_question must end with a question mark")
        return value

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "ReproductionStep":
        """单步层面拒绝依赖自身，完整顺序由 LearningPlan 校验。"""
        if self.step_id in self.depends_on:
            raise ValueError("a step cannot depend on itself")
        return self


class LearningPlan(StrictLearningModel):
    """一次项目拆解生成的项目画像和四到六步复现路线。"""

    project_profile: ProjectProfile
    steps: list[ReproductionStep] = Field(min_length=4, max_length=6)

    @model_validator(mode="after")
    def validate_step_order(self) -> "LearningPlan":
        """保证步骤标识唯一，并且依赖只指向已经出现的前序步骤。"""
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError("step ids must be unique")
            unknown = [dependency for dependency in step.depends_on if dependency not in seen]
            if unknown:
                raise ValueError("step dependencies must refer to earlier steps")
            seen.add(step.step_id)
        return self


class StepSubmission(StrictLearningModel):
    """学习者为当前步骤提交的实现说明和测试输出。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=False,
    )

    implementation_summary: str = Field(max_length=12_000)
    test_output: str = Field(max_length=20_000)


class StepReviewFinding(StrictLearningModel):
    """一次审查对单条验收要求作出的判断。"""

    verification_index: int = Field(ge=0)
    satisfied: StrictBool
    reason: ReviewText


class StepReview(StrictLearningModel):
    """只基于学习者报告证据形成的结构化步骤审查。"""

    passed: StrictBool
    evidence_sufficient: StrictBool
    findings: list[StepReviewFinding] = Field(
        min_length=1,
        max_length=MAX_VERIFICATION_ITEMS,
    )
    gaps: list[ReviewText] = Field(max_length=MAX_VERIFICATION_ITEMS)
    hint: ReviewText | None
    verification_scope: Literal["learner_reported_evidence"]

    @model_validator(mode="after")
    def validate_review_conclusion(self) -> "StepReview":
        """保证逐项判断、证据充分性和最终结论彼此一致。"""
        indexes = [finding.verification_index for finding in self.findings]
        if len(indexes) != len(set(indexes)):
            raise ValueError("verification_index values must be unique")

        expected_passed = self.evidence_sufficient and all(
            finding.satisfied for finding in self.findings
        )
        if self.passed != expected_passed:
            raise ValueError(
                "passed must equal evidence_sufficient and all findings satisfied"
            )

        if self.passed:
            if self.gaps or self.hint is not None:
                raise ValueError("a passed review must have empty gaps and no hint")
            return self

        if not self.gaps:
            raise ValueError("a failed review requires at least one gap")
        if self.hint is None or not self.hint.strip():
            raise ValueError("a failed review requires a non-empty hint")
        return self


class RevisionFeedback(StrictLearningModel):
    """向学习者公开的最小修改反馈。"""

    gaps: list[ReviewText] = Field(
        min_length=1,
        max_length=MAX_VERIFICATION_ITEMS,
    )
    hint: ReviewText


class StepReviewRecord(StrictLearningModel):
    """保存一次步骤审查及其不含原始提交文本的审计元数据。"""

    step_id: str = Field(min_length=1, pattern=STEP_ID_PATTERN)
    attempt: int = Field(ge=1)
    review: StepReview
    review_source: Literal["precheck", "model"]
    submission_digest: str = Field(pattern=SHA256_PATTERN)
    implementation_chars: int = Field(ge=0)
    test_output_chars: int = Field(ge=0)
    execution_performed: Literal[False] = False


class StepReflectionRecord(StrictLearningModel):
    """保存学习者完成一个步骤后的反思。"""

    step_id: str = Field(min_length=1, pattern=STEP_ID_PATTERN)
    answer: str = Field(min_length=1, max_length=12_000)


LearningSessionEventOutcome = Literal[
    "started",
    "insufficient_submission",
    "needs_revision",
    "awaiting_reflection",
    "advanced",
    "completed",
]


class LearningSessionEvent(StrictLearningModel):
    """会话的一次公开状态转换记录。"""

    turn: int = Field(ge=1)
    action: LearningSessionAction
    from_phase: LearningSessionPhase | None
    to_phase: LearningSessionPhase
    step_id: str = Field(min_length=1, pattern=STEP_ID_PATTERN)
    outcome: LearningSessionEventOutcome


class LearningSession(StrictLearningModel):
    """由调用方保存并在每轮请求中回传的完整学习会话。"""

    schema_version: Literal["1.0"] = "1.0"
    session_id: str = Field(min_length=1)
    learning_goal: str = Field(min_length=1)
    learner_level: Literal["beginner"] = "beginner"
    plan: LearningPlan
    current_step_index: int = Field(ge=0)
    phase: LearningSessionPhase
    reviews: list[StepReviewRecord] = Field(default_factory=list)
    reflections: list[StepReflectionRecord] = Field(default_factory=list)
    history: list[LearningSessionEvent] = Field(min_length=1)
    integrity_token: str = Field(pattern=SHA256_PATTERN)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        """会话标识保持字符串形态，同时必须是合法 UUID。"""
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("session_id must be a valid UUID string") from exc
        return value

    @model_validator(mode="after")
    def validate_session_state(self) -> "LearningSession":
        """拒绝无法由公开 action 流程产生的会话状态组合。"""
        steps = self.plan.steps
        if self.current_step_index >= len(steps):
            raise ValueError("current_step_index must identify a planned step")

        step_ids = [step.step_id for step in steps]
        step_positions = {step_id: index for index, step_id in enumerate(step_ids)}
        current_step_id = step_ids[self.current_step_index]

        reflection_ids = [record.step_id for record in self.reflections]
        if reflection_ids != step_ids[: len(reflection_ids)]:
            raise ValueError("reflections must be a strict prefix of planned steps")

        reviews_by_step: dict[str, list[StepReviewRecord]] = {
            step_id: [] for step_id in step_ids
        }
        highest_review_position = -1
        passed_steps: set[str] = set()
        for record in self.reviews:
            position = step_positions.get(record.step_id)
            if position is None or position > self.current_step_index:
                raise ValueError("reviews may only refer to current or past steps")
            if position < highest_review_position:
                raise ValueError("review records must follow planned step order")
            highest_review_position = position

            records = reviews_by_step[record.step_id]
            if record.attempt != len(records) + 1:
                raise ValueError("review attempts must be consecutive for each step")
            if record.step_id in passed_steps:
                raise ValueError("a passed step cannot be reviewed again")

            verification_count = len(steps[position].verification)
            finding_indices = sorted(
                finding.verification_index for finding in record.review.findings
            )
            if finding_indices != list(range(verification_count)):
                raise ValueError(
                    "every review must cover each verification item exactly once"
                )
            if record.review_source == "precheck" and record.review.passed:
                raise ValueError("a precheck review cannot pass a step")

            records.append(record)
            if record.review.passed:
                passed_steps.add(record.step_id)

        for index in range(self.current_step_index):
            records = reviews_by_step[step_ids[index]]
            if not records or not records[-1].review.passed:
                raise ValueError("every past step must have a passing review")
            if index >= len(self.reflections):
                raise ValueError("every past step must have one reflection")

        current_reviews = reviews_by_step[current_step_id]
        if self.phase == "completed":
            if self.current_step_index != len(steps) - 1:
                raise ValueError("a completed session must remain on the last step index")
            if reflection_ids != step_ids:
                raise ValueError("a completed session requires every step reflection")
            for step_id in step_ids:
                records = reviews_by_step[step_id]
                if not records or not records[-1].review.passed:
                    raise ValueError("a completed session requires every step to pass")
        else:
            if len(self.reflections) != self.current_step_index:
                raise ValueError("only past steps may have reflections before completion")
            if self.phase == "needs_revision":
                if not current_reviews or current_reviews[-1].review.passed:
                    raise ValueError(
                        "needs_revision requires a failed current-step review"
                    )
            elif self.phase == "awaiting_reflection":
                if not current_reviews or not current_reviews[-1].review.passed:
                    raise ValueError(
                        "awaiting_reflection requires a passing current-step review"
                    )
            elif self.phase == "awaiting_submission" and current_reviews:
                raise ValueError(
                    "awaiting_submission cannot already contain a current-step review"
                )

        expected_turns = list(range(1, len(self.history) + 1))
        if [event.turn for event in self.history] != expected_turns:
            raise ValueError("history turns must be consecutive and start at one")
        first_event = self.history[0]
        if (
            first_event.action != "start"
            or first_event.from_phase is not None
            or first_event.to_phase != "awaiting_submission"
            or first_event.step_id != step_ids[0]
            or first_event.outcome != "started"
        ):
            raise ValueError("history must begin with the initial start transition")
        if self.history[-1].to_phase != self.phase:
            raise ValueError("the last history event must end in the current phase")
        if len(self.history) != 1 + len(self.reviews) + len(self.reflections):
            raise ValueError("history must contain one event per review and reflection")

        review_index = 0
        reflection_index = 0
        for index, event in enumerate(self.history):
            if event.step_id not in step_positions:
                raise ValueError("history events must refer to planned steps")
            if index == 0:
                continue
            if event.from_phase != self.history[index - 1].to_phase:
                raise ValueError("history phase transitions must be contiguous")
            if event.action == "submit":
                if review_index >= len(self.reviews):
                    raise ValueError("submit history event requires a review record")
                record = self.reviews[review_index]
                review_index += 1
                if event.step_id != record.step_id or event.from_phase not in {
                    "awaiting_submission",
                    "needs_revision",
                }:
                    raise ValueError("submit history event has invalid step or source phase")
                expected_phase = (
                    "awaiting_reflection" if record.review.passed else "needs_revision"
                )
                expected_outcome = (
                    "insufficient_submission"
                    if record.review_source == "precheck"
                    else expected_phase
                )
                if event.to_phase != expected_phase or event.outcome != expected_outcome:
                    raise ValueError("submit history event does not match its review")
                continue
            if event.action == "reflect":
                if reflection_index >= len(self.reflections):
                    raise ValueError("reflect history event requires a reflection record")
                record = self.reflections[reflection_index]
                step_position = step_positions[record.step_id]
                reflection_index += 1
                is_last = step_position == len(steps) - 1
                expected_phase = "completed" if is_last else "awaiting_submission"
                expected_outcome = "completed" if is_last else "advanced"
                if (
                    event.step_id != record.step_id
                    or event.from_phase != "awaiting_reflection"
                    or event.to_phase != expected_phase
                    or event.outcome != expected_outcome
                ):
                    raise ValueError("reflect history event does not match its reflection")
                continue
            raise ValueError("only the first history event may use the start action")
        if review_index != len(self.reviews) or reflection_index != len(self.reflections):
            raise ValueError("history must account for every review and reflection")
        return self


class StartLearningSessionRequest(StrictLearningModel):
    """启动一条新的学习路线。"""

    action: Literal["start"]
    learning_goal: str = Field(min_length=1)
    learner_level: Literal["beginner"] = "beginner"


class SubmitLearningSessionRequest(StrictLearningModel):
    """提交当前步骤的学习者报告证据。"""

    action: Literal["submit"]
    session: LearningSession
    submission: StepSubmission


class ReflectLearningSessionRequest(StrictLearningModel):
    """记录反思，并在安全时推进至下一步骤。"""

    action: Literal["reflect"]
    session: LearningSession
    reflection: str = Field(max_length=12_000)


LearningSessionRequest = Annotated[
    StartLearningSessionRequest
    | SubmitLearningSessionRequest
    | ReflectLearningSessionRequest,
    Field(discriminator="action"),
]


class LearningSessionError(StrictLearningModel):
    """学习会话入口对调用方公开的稳定错误结构。"""

    code: LearningSessionErrorCode
    message: str = Field(min_length=1)


class LearningSessionResult(StrictLearningModel):
    """一次 start、submit 或 reflect 调用的公开结果。"""

    status: Literal["success", "failed"]
    action: LearningSessionAction | None = None
    session: LearningSession | None = None
    current_step: ReproductionStep | None = None
    next_action: NextLearningAction = "none"
    review: StepReview | None = None
    feedback: RevisionFeedback | None = None
    graph_steps: list[str]
    trace: dict[str, Any]
    error: LearningSessionError | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "LearningSessionResult":
        """保证结果字段和会话阶段一致，失败时仍可安全重试。"""
        if self.status == "success":
            if self.action is None or self.session is None:
                raise ValueError("a successful result requires an action and session")
            if self.error is not None:
                raise ValueError("a successful result cannot include an error")
        elif self.error is None:
            raise ValueError("a failed result requires an error")

        if self.session is None:
            if any(
                value is not None
                for value in (
                    self.current_step,
                    self.review,
                    self.feedback,
                )
            ):
                raise ValueError("a result without a session cannot expose session data")
            if self.next_action != "none":
                raise ValueError("a result without a session must have next_action none")
            return self

        if (
            self.status == "success"
            and self.action != self.session.history[-1].action
        ):
            raise ValueError("a successful action must match the latest session event")

        if self.session.phase == "completed":
            expected_step = None
            expected_action = "none"
        else:
            expected_step = self.session.plan.steps[self.session.current_step_index]
            expected_action = (
                "reflect"
                if self.session.phase == "awaiting_reflection"
                else "submit"
            )
        if self.current_step != expected_step or self.next_action != expected_action:
            raise ValueError("current_step and next_action must match the session phase")

        if self.status == "failed":
            if self.review is not None or self.feedback is not None:
                raise ValueError("a failed result cannot expose review output")
            return self

        current_reviews = [
            record
            for record in self.session.reviews
            if record.step_id
            == self.session.plan.steps[self.session.current_step_index].step_id
        ]
        if self.session.phase == "needs_revision":
            latest_review = current_reviews[-1].review
            expected_feedback = RevisionFeedback(
                gaps=latest_review.gaps,
                hint=latest_review.hint,
            )
            if self.review != latest_review or self.feedback != expected_feedback:
                raise ValueError(
                    "needs_revision must expose its latest review and feedback"
                )
        elif self.session.phase == "awaiting_reflection":
            latest_review = current_reviews[-1].review
            if self.review != latest_review or self.feedback is not None:
                raise ValueError(
                    "awaiting_reflection must expose its passing review without feedback"
                )
        elif self.review is not None or self.feedback is not None:
            raise ValueError(
                "review output is only exposed immediately before revision or reflection"
            )
        return self


class LearningError(StrictLearningModel):
    """Learning Mode 对调用方公开的稳定错误结构。"""

    code: LearningErrorCode
    message: str = Field(min_length=1)


class LearningWorkflowResult(StrictLearningModel):
    """Learning Mode 一次启动请求的公开、可 JSON 序列化结果。"""

    status: Literal["success", "failed"]
    learning_goal: str = Field(min_length=1)
    learner_level: Literal["beginner"]
    project_profile: ProjectProfile | None = None
    steps: list[ReproductionStep] = Field(default_factory=list)
    first_step: ReproductionStep | None = None
    graph_steps: list[str]
    trace: dict[str, Any]
    error: LearningError | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "LearningWorkflowResult":
        """避免失败结果泄露未校验计划，并保证成功结果字段完整。"""
        expected_steps = ["collect_evidence", "analyze_and_plan", "present_step"]
        if self.status == "success":
            if self.project_profile is None or not 4 <= len(self.steps) <= 6:
                raise ValueError("success result requires a complete learning plan")
            if self.first_step != self.steps[0]:
                raise ValueError("first_step must equal the first planned step")
            if self.graph_steps != expected_steps or self.error is not None:
                raise ValueError("success result has inconsistent workflow state")
            return self

        if self.project_profile is not None or self.steps or self.first_step is not None:
            raise ValueError("failed result must not expose an unvalidated plan")
        if self.error is None:
            raise ValueError("failed result requires an error")
        return self


def normalize_repo_relative_path(value: str) -> str:
    """返回稳定 POSIX 相对路径，并拒绝冒号、绝对路径和 ``..``。"""
    normalized = value.strip().replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or WINDOWS_DRIVE_PATTERN.match(normalized)
        or ":" in normalized
    ):
        raise ValueError("path must be repository-relative")
    candidate = PurePosixPath(normalized)
    if any(part == ".." for part in candidate.parts):
        raise ValueError("path must not escape the repository")
    result = str(candidate)
    if result in {"", "."}:
        raise ValueError("path must identify a file")
    return result
