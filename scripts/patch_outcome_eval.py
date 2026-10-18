"""Run the trusted-local, real-source patch outcome harness and independent gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.agent_patch_provider import AgentCandidateProvider
from coding_rag.evaluation.patch_outcome import (
    OraclePatchProvider,
    PatchOutcomeValidationError,
    StaticPatchProvider,
    compare_patch_scorecards,
    evaluate_patch_suite,
    gate_exit_code,
    load_and_validate_patch_suite,
    validate_patch_gate_preflight,
)
from coding_rag.rag.llm_client import (
    OpenAICompatibleChatClient,
    available_provider_names,
    build_llm_config,
)
from coding_rag.tools.env import load_dotenv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse real-source manifest, trusted candidate, artifact, and gate controls."""
    parser = argparse.ArgumentParser(
        description="Evaluate final patch/test outcomes; this is not an LLM intelligence benchmark"
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--manifest", default="datasets/eval/django_patch_outcome_v2.json")
    parser.add_argument(
        "--candidate-mode", choices=("oracle", "static", "agent-live"), default="oracle",
    )
    parser.add_argument("--candidate-patches", help="JSON object mapping task IDs to unified diffs")
    parser.add_argument(
        "--allow-local-code-execution", action="store_true",
        help="Required for external patches: tests execute trusted local candidate code without an OS sandbox",
    )
    parser.add_argument("--allow-network-llm", action="store_true")
    parser.add_argument("--accept-provider-charges", action="store_true")
    parser.add_argument(
        "--llm-provider", choices=available_provider_names() + ["custom"], default="deepseek",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=2000)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--agent-max-steps", type=int, default=6)
    parser.add_argument("--agent-max-approvals", type=int, default=2)
    parser.add_argument("--agent-context-chars", type=int, default=12000)
    parser.add_argument("--baseline", default="benchmarks/patch-outcome-baseline-v2.json")
    parser.add_argument("--policy", default="benchmarks/patch-outcome-gate-v2.json")
    parser.add_argument("--out", default="artifacts/eval/patch-outcome-scorecard.json")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    """Write a deterministic human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, payload)


def _provider(args: argparse.Namespace, suite, *, client=None):
    """Build an oracle, trusted static, or explicitly enabled live Agent provider."""
    if args.candidate_mode == "oracle":
        if args.candidate_patches:
            raise ValueError("--candidate-patches is only valid with --candidate-mode static")
        return OraclePatchProvider(suite.tasks)
    if args.candidate_mode == "agent-live":
        if args.candidate_patches:
            raise ValueError("--candidate-patches is only valid with --candidate-mode static")
        if not args.allow_local_code_execution:
            raise ValueError("agent-live requires explicit --allow-local-code-execution")
        if not args.allow_network_llm or not args.accept_provider_charges:
            raise ValueError(
                "agent-live requires --allow-network-llm and --accept-provider-charges"
            )
        if client is None:
            llm_config = build_llm_config(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
            )
            client = OpenAICompatibleChatClient(llm_config)
        else:
            llm_config = getattr(client, "config", None)
            if llm_config is None:
                raise ValueError("injected agent-live client must expose safe provider/model config")
        return AgentCandidateProvider(
            client,
            provider_label=llm_config.provider,
            model_label=llm_config.model,
            max_steps=args.agent_max_steps,
            max_approvals=args.agent_max_approvals,
            max_context_chars=args.agent_context_chars,
            max_output_tokens_per_call=llm_config.max_tokens,
        )
    if not args.allow_local_code_execution:
        raise ValueError(
            "static candidates require explicit --allow-local-code-execution; "
            "workspace isolation is not an OS sandbox"
        )
    if not args.candidate_patches:
        raise ValueError("static candidate mode requires --candidate-patches")
    payload = json.loads(Path(args.candidate_patches).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("candidate patches file must be a JSON object of task-id to diff strings")
    if set(payload) - {task.id for task in suite.tasks}:
        raise ValueError("candidate patches file contains unknown task IDs")
    return StaticPatchProvider(payload)


def main(argv: list[str] | None = None) -> int:
    """Validate, evaluate, optionally update oracle baseline, and enforce the gate."""
    args = parse_args(argv)
    if args.gate and args.update_baseline:
        print("Evaluation error: --gate and --update-baseline are mutually exclusive")
        return 2
    if args.update_baseline and args.candidate_mode != "oracle":
        print("Evaluation error: baseline updates are restricted to explicit oracle self-check mode")
        return 2
    if args.candidate_mode == "agent-live" and args.gate:
        print(
            "Evaluation error: agent-live gate is not supported for oracle baselines; "
            "run without --gate to report live repair outcomes"
        )
        return 2
    if args.candidate_mode == "agent-live" and not (
        args.allow_local_code_execution
        and args.allow_network_llm
        and args.accept_provider_charges
    ):
        print(
            "Evaluation error: agent-live requires --allow-local-code-execution, "
            "--allow-network-llm, and --accept-provider-charges"
        )
        return 2
    if args.candidate_mode == "agent-live":
        if not 1 <= args.llm_timeout <= 300:
            print("Evaluation error: --llm-timeout must be between 1 and 300")
            return 2
        if not 1 <= args.llm_max_tokens <= 8192:
            print("Evaluation error: --llm-max-tokens must be between 1 and 8192")
            return 2
        if args.llm_temperature is not None and not 0 <= args.llm_temperature <= 2:
            print("Evaluation error: --llm-temperature must be between 0 and 2")
            return 2
    try:
        suite = load_and_validate_patch_suite(args.workspace_root, args.manifest)
        baseline = policy = None
        baseline_path = Path(args.baseline)
        policy_path = Path(args.policy)
        if args.gate:
            if not baseline_path.is_file() or not policy_path.is_file():
                raise ValueError("baseline or policy does not exist")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            validate_patch_gate_preflight(suite, baseline, policy)
        if args.candidate_mode == "agent-live":
            load_dotenv(args.env_file)
        provider = _provider(args, suite)
        if args.candidate_mode == "agent-live":
            upper = len(suite.tasks) * (args.agent_max_steps + 2) * args.llm_max_tokens
            print(
                "agent-live disclosure: task requirements and selected public Django source "
                f"may be sent to provider={args.llm_provider}; provider charges may apply. "
                f"Logical calls are bounded to {args.agent_max_steps + 2} per task and the "
                f"theoretical output-token cap for this suite is {upper}; exact billing is unavailable."
            )
        scorecard = evaluate_patch_suite(suite, provider)
        _write_json(Path(args.out), scorecard)
        print(json.dumps({
            "label": scorecard["benchmark"],
            "self_check_only": args.candidate_mode == "oracle",
            "score": scorecard["score"],
            "metrics": scorecard["metrics"], "cases": scorecard["cases"],
        }, ensure_ascii=False, indent=2))
        if args.update_baseline:
            if (
                scorecard["metrics"]["harness_valid_rate"] != 1.0
                or scorecard["metrics"]["test_pass_rate"] != 1.0
            ):
                raise ValueError("oracle baseline update requires every harness and test outcome to pass")
            _write_json(baseline_path, scorecard)
            print(f"Baseline updated: {baseline_path}")
            return 0
        if not args.gate:
            return 0
        assert baseline is not None and policy is not None
        result = compare_patch_scorecards(scorecard, baseline, policy)
        print("PASS" if result.passed else "FAIL")
        for failure in result.failures:
            print(f"- {failure}")
        return gate_exit_code(result)
    except PatchOutcomeValidationError as error:
        print("Validation failed:")
        for finding in error.errors:
            print(f"- {finding}")
        return 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Evaluation error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
