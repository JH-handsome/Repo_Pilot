"""Validate a RepoPilot retrieval evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.evaluation.retrieval import validate_evalset


def main() -> int:
    """Print validation findings and return zero for valid data or two otherwise."""
    parser = argparse.ArgumentParser(description="Validate retrieval benchmark data")
    parser.add_argument("repo_path")
    parser.add_argument("evalset")
    args = parser.parse_args()
    try:
        result = validate_evalset(args.repo_path, args.evalset)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
