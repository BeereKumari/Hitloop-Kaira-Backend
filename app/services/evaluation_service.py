"""
Reusable Code Evaluation Service

Executes candidate code against hidden test cases via Judge0,
compares actual vs expected output after whitespace trimming,
and returns a structured EvaluationReport with pass/fail counts,
percentage score, and per-test-case details.

Usage:
    from app.services.evaluation_service import evaluate_code

    report = await evaluate_code(
        code="def solve(nums): ...",
        language="python",
        test_cases=[{"input": "1 2 3", "expected_output": "6", "is_hidden": True}],
        timeout_seconds=10,
    )
    print(report.percentage)  # 100.0
    print(report.score)       # 10.0
"""

import asyncio
from typing import List, Dict

from app.models.coding import EvaluationTestResult, EvaluationReport
from app.services.judge0_service import (
    LANGUAGE_MAP,
    JUDGE0_BASE_URL,
    submit_code,
    poll_result,
    parse_result,
)


async def evaluate_code(
    code: str,
    language: str,
    test_cases: List[Dict],
    timeout_seconds: int = 10,
    score_scale: int = 10,
    hidden_only: bool = True,
) -> EvaluationReport:
    """
    Evaluate code against test cases and return a structured report.

    For every test case the code is executed via Judge0. The actual stdout
    is compared with the expected output after stripping leading/trailing
    whitespace on both sides.

    Args:
        code:         Source code to evaluate.
        language:     Language identifier — one of python, javascript, java, c, cpp.
        test_cases:   List of dicts with keys 'input', 'expected_output',
                      and optional 'is_hidden', 'explanation'.
        timeout_seconds: Max seconds to wait per test case execution.
        score_scale:  Maximum score value (default 10).
        hidden_only:  If True only evaluate test cases where is_hidden=True.
                      Falls back to all cases when none are marked hidden.

    Returns:
        EvaluationReport containing pass/fail counts, percentage, score,
        per-test-case results, aggregated metrics, and any errors.
    """
    eval_cases = _filter_test_cases(test_cases, hidden_only)
    total = len(eval_cases)

    if not JUDGE0_BASE_URL:
        return _unavailable_report(total, score_scale)

    if language not in LANGUAGE_MAP:
        return EvaluationReport(
            passed=0,
            total=total,
            percentage=0.0,
            score=0.0,
            score_scale=score_scale,
            results=[],
            compile_error=f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}",
            judge0_status="Unsupported Language",
        )

    judge0_language_id = LANGUAGE_MAP[language]

    results: List[EvaluationTestResult] = []
    passed = 0
    compile_error = None
    runtime_error = None
    total_time_ms = 0
    peak_memory_kb = 0
    overall_status = "Accepted"

    for idx, tc in enumerate(eval_cases):
        result = await _evaluate_single(
            code=code,
            language_id=judge0_language_id,
            test_case=tc,
            index=idx,
            timeout=timeout_seconds,
        )
        results.append(result)

        if result.passed:
            passed += 1
        else:
            status = result.judge0_status or ""
            if "Compile" in status and not compile_error:
                compile_error = result.actual
                overall_status = status
            elif not runtime_error and status in (
                "Runtime Error", "Time Limit Exceeded",
                "Output Limit Exceeded", "Execution Error",
            ):
                runtime_error = result.actual
                overall_status = status
            elif overall_status == "Accepted":
                overall_status = status or "Wrong Answer"

        if result.time_ms is not None:
            total_time_ms += result.time_ms
        if result.memory_kb is not None and result.memory_kb > peak_memory_kb:
            peak_memory_kb = result.memory_kb

    percentage = round((passed / total) * 100, 1) if total > 0 else 0.0
    score = round((passed / total) * score_scale, 1) if total > 0 else 0.0

    return EvaluationReport(
        passed=passed,
        total=total,
        percentage=percentage,
        score=score,
        score_scale=score_scale,
        results=results,
        compile_error=compile_error,
        runtime_error=runtime_error,
        execution_time_ms=total_time_ms if any(r.time_ms is not None for r in results) else None,
        memory_usage_kb=peak_memory_kb if peak_memory_kb > 0 else None,
        judge0_status=overall_status,
    )


# ─── Internals ──────────────────────────────────────────────────────────────


def _filter_test_cases(test_cases: List[Dict], hidden_only: bool) -> List[Dict]:
    """Return the subset of test cases to evaluate against."""
    if hidden_only:
        hidden = [tc for tc in test_cases if tc.get("is_hidden", False)]
        if hidden:
            return hidden
    return test_cases


async def _evaluate_single(
    code: str,
    language_id: int,
    test_case: dict,
    index: int,
    timeout: int,
) -> EvaluationTestResult:
    """Execute code against one test case and compare trimmed outputs."""
    stdin_data = test_case.get("input", "")
    expected_raw = test_case.get("expected_output", "")
    explanation = test_case.get("explanation")

    try:
        token = await submit_code(code, language_id, stdin_data)
        raw = await poll_result(token, timeout)
        parsed = parse_result(raw, {
            "input": stdin_data,
            "expected": expected_raw,
            "explanation": explanation,
        })
    except asyncio.TimeoutError:
        return EvaluationTestResult(
            test_case_index=index,
            input=stdin_data,
            expected=expected_raw.strip(),
            actual="Time Limit Exceeded",
            passed=False,
            explanation=explanation,
            judge0_status="Time Limit Exceeded",
        )
    except Exception as e:
        return EvaluationTestResult(
            test_case_index=index,
            input=stdin_data,
            expected=expected_raw.strip(),
            actual=f"Execution error: {str(e)[:300]}",
            passed=False,
            explanation=explanation,
            judge0_status="Execution Error",
        )

    actual_trimmed = (parsed.get("actual") or "").strip()
    expected_trimmed = expected_raw.strip()
    passed = actual_trimmed == expected_trimmed and parsed.get("passed", False)

    return EvaluationTestResult(
        test_case_index=index,
        input=stdin_data,
        expected=expected_trimmed,
        actual=actual_trimmed,
        passed=passed,
        explanation=explanation,
        time_ms=parsed.get("time_ms"),
        memory_kb=parsed.get("memory_kb"),
        judge0_status=parsed.get("judge0_status"),
    )


def _unavailable_report(total: int, score_scale: int) -> EvaluationReport:
    """Return a report when Judge0 is not configured."""
    return EvaluationReport(
        passed=0,
        total=total,
        percentage=0.0,
        score=0.0,
        score_scale=score_scale,
        results=[],
        compile_error="Code execution engine is not configured. Set JUDGE0_URL in environment.",
        judge0_status="Unavailable",
    )
