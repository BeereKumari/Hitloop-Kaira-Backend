import os
import json
import asyncio
from typing import List, Dict, Any

from cerebras.cloud.sdk import Cerebras

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

__all__ = [
    "generate_coding_problem",
    "generate_coding_evaluation",
    "generate_multi_question_evaluation",
]


def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)


# ─── Problem Generation ────────────────────────────────────────────────────

PROBLEM_GENERATION_PROMPT = """Generate a coding problem as JSON. Schema:
{"title":"str","description":"markdown with constraints","difficulty":"easy|medium|hard|expert","category":"str","function_signature":"def solve(...) — the function the candidate implements","examples":[{"input":"str (JSON)","output":"str (what the code should print)","explanation":"str"}],"test_cases":[{"input":"str — one JSON value per line for multiple args","expected_output":"str — exact string the code should print","is_hidden":false}],"hints":["str"],"time_complexity_expected":"O(n)","space_complexity_expected":"O(n)"}

CRITICAL RULES:
- function_signature MUST always use function name "solve"
- test_case input: single JSON value on one line (e.g. "[1,2,3]" or "[2,7,11,15]\\n9" for two args — newline-separated)
- test_case expected_output: the EXACT string the program should print (e.g. "6" not [6], "[0,1]" not "indices")
- examples follow the same format as test_cases
- 5 test cases total (3 visible, 2 hidden), all deterministic
- Keep description under 300 words
- Return ONLY valid JSON."""


async def generate_coding_problem(
    candidate_skills: str,
    applied_role: str,
    difficulty: str,
    category: str = None,
) -> dict:
    client = get_cerebras_client()

    category_line = f"Preferred category: {category}" if category else "Choose the most relevant category based on the candidate's skills."

    user_content = (
        f"Skills: {candidate_skills[:200]}\n"
        f"Role: {applied_role}\n"
        f"Difficulty: {difficulty}\n"
        f"{category_line}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": PROBLEM_GENERATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()

    # Retry up to 3 times with backoff to handle rate limits
    last_error = None
    for attempt in range(3):
        try:
            raw = await loop.run_in_executor(None, _call)
            if not raw:
                last_error = ValueError("Cerebras returned empty response")
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                continue
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
    else:
        raise ValueError(f"Failed to generate problem after 3 attempts: {last_error}")

    import re

    # Helper 1: escapes literal control chars in double-quoted JSON strings
    def sanitize_json_control_chars(s: str) -> str:
        in_str = False
        esc = False
        cleaned = []
        for char in s:
            if esc:
                cleaned.append(char)
                esc = False
                continue
            if char == "\\":
                cleaned.append(char)
                esc = True
                continue
            if char == '"':
                cleaned.append(char)
                in_str = not in_str
                continue
            if in_str:
                if char == "\n":
                    cleaned.append("\\n")
                elif char == "\r":
                    cleaned.append("\\r")
                elif char == "\t":
                    cleaned.append("\\t")
                else:
                    cleaned.append(char)
            else:
                cleaned.append(char)
        return "".join(cleaned)

    # Helper 2: balance braces/brackets on truncated string
    def repair_truncated_json(s: str) -> str:
        s = s.strip()
        if not s:
            return "{}"
        
        # Strip trailing commas
        if s.endswith(","):
            s = s[:-1].strip()

        stack = []
        in_str = False
        esc = False
        cleaned = []
        for char in s:
            cleaned.append(char)
            if esc:
                esc = False
                continue
            if char == "\\":
                esc = True
                continue
            if char == '"':
                in_str = not in_str
                continue
            if not in_str:
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char == "}":
                    if stack and stack[-1] == "}":
                        stack.pop()
                elif char == "]":
                    if stack and stack[-1] == "]":
                        stack.pop()

        repaired_str = "".join(cleaned)
        if in_str:
            repaired_str += '"'
        while stack:
            repaired_str += stack.pop()
        return repaired_str

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Repair common LLM bolding key/value formatting errors
    raw = re.sub(r'\*\*\s*"(.*?)"\s*\*\*(\s*:)', r'"\1"\2', raw)
    raw = re.sub(r'"\s*\*\*(.*?)\*\*\s*"(\s*:)', r'"\1"\2', raw)
    raw = re.sub(r'\*\*(.*?)\*\*(\s*:)', r'"\1"\2', raw)
    raw = re.sub(r':\s*\*\*\s*"(.*?)"\s*\*\*', r': "\1"', raw)
    raw = re.sub(r':\s*"\*\*(.*?)\*\*"', r': "\1"', raw)

    # Sanitize control chars first
    raw_sanitized = sanitize_json_control_chars(raw)

    try:
        problem = json.loads(raw_sanitized)
    except json.JSONDecodeError:
        # Attempt to repair truncated JSON
        try:
            repaired = repair_truncated_json(raw)
            repaired_san = sanitize_json_control_chars(repaired)
            problem = json.loads(repaired_san)
        except Exception:
            raise ValueError(f"Failed to parse problem generation response\nRaw: {raw[:500]}")

    # ── Validate: ensure test cases exist ──
    test_cases = problem.get("test_cases", [])
    if len(test_cases) < 2:
        # Retry once with a stronger prompt
        retry_content = (
            f"{user_content}\n\n"
            "IMPORTANT: Your previous response was missing test_cases. "
            "You MUST include the 'test_cases' array with exactly 5 entries "
            "(3 with is_hidden:false, 2 with is_hidden:true). "
            "Each test case must have 'input', 'expected_output', and 'is_hidden' fields."
        )

        def _retry_call():
            resp = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[
                    {"role": "system", "content": PROBLEM_GENERATION_PROMPT},
                    {"role": "user", "content": retry_content},
                ],
                temperature=0.3,
                max_tokens=2500,
            )
            return resp.choices[0].message.content

        raw2 = await loop.run_in_executor(None, _retry_call)
        raw2 = raw2.strip()
        if raw2.startswith("```"):
            raw2 = raw2.split("```")[1]
            if raw2.startswith("json"):
                raw2 = raw2[4:]
            raw2 = raw2.strip()

        try:
            problem = json.loads(raw2)
        except (json.JSONDecodeError, ValueError):
            pass  # keep the original problem

        test_cases = problem.get("test_cases", [])

    # ── Last resort: derive test cases from examples ──
    if len(test_cases) < 2:
        examples = problem.get("examples", [])
        derived = []
        for i, ex in enumerate(examples):
            derived.append({
                "input": ex.get("input", ""),
                "expected_output": ex.get("output", ""),
                "is_hidden": i >= len(examples) - 2,
            })
        if derived:
            problem["test_cases"] = derived

    # ── Ensure function_signature always says "solve" ──
    sig = problem.get("function_signature", "")
    if sig and "solve" not in sig:
        problem["function_signature"] = sig

    return problem


# ─── AI Evaluation ──────────────────────────────────────────────────────────

CODE_EVALUATION_PROMPT = """You are a senior software engineer generating a detailed coding assessment review.
Analyze the candidate's code submission against the problem, execution results, and hidden test case outcomes.
Return a valid JSON object matching this exact schema:

{
  "overall_score": 8.5,
  "recommendation": "Strong Hire",
  "overall_summary": "3-4 sentence comprehensive evaluation of the candidate's coding ability.",
  "correctness": {
    "score": 9,
    "assessment": "Detailed assessment of whether the solution correctly solves the problem, handles edge cases, and passes all test cases."
  },
  "complexity": {
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "assessment": "Analysis of the solution's time and space complexity relative to the optimal approach."
  },
  "code_quality": {
    "score": 8,
    "assessment": "Evaluation of code structure, modularity, readability, and adherence to best practices."
  },
  "naming_conventions": {
    "score": 7,
    "assessment": "Review of variable, function, and class naming clarity and consistency."
  },
  "edge_case_handling": {
    "score": 8,
    "assessment": "Assessment of how well the code handles edge cases such as empty inputs, single elements, boundaries, and large values."
  },
  "optimization_suggestions": [
    "Specific actionable suggestion to improve performance or correctness"
  ],
  "strengths": [
    "Specific strength demonstrated in the code with evidence"
  ],
  "weaknesses": [
    "Specific weakness or area for improvement"
  ],
  "hiring_notes": "Private notes for the hiring panel.",
  "skills_demonstrated": ["Array Manipulation", "Hash Map", "Recursion"],
  "recommended_topics": ["Dynamic Programming", "Graph Traversal"],
  "recruiter_insight": "2-3 sentence note for non-technical recruiters summarizing what this score means for the role fit."
}

Rules:
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- All score fields (overall_score, correctness.score, code_quality.score, naming_conventions.score, edge_case_handling.score) are on a 0-10 scale.
- optimization_suggestions should be concrete and actionable (e.g. "Use a hash map instead of linear scan for O(1) lookup").
- skills_demonstrated: list 3-6 specific technical skills the candidate showed (e.g. "Hash Map", "Two Pointers", "Recursion", "Edge Case Handling").
- recommended_topics: list 2-4 topics the candidate should study to improve (e.g. "Dynamic Programming", "Graph Algorithms").
- recruiter_insight: write in plain English, avoid jargon, focus on role-readiness and growth potential.
- Return ONLY valid JSON with no markdown fences."""


async def generate_coding_evaluation(
    problem: dict,
    code: str,
    language: str,
    execution_results: dict,
    candidate_name: str,
    applied_role: str,
) -> dict:
    """
    Send the coding question, submitted code, execution summary, and hidden
    test case results to the AI model and return a structured review.
    """
    client = get_cerebras_client()

    # ── Build execution summary with per-test-case details ──
    test_cases = execution_results.get("results", [])
    passed = execution_results.get("passed", 0)
    total = execution_results.get("total", 0)

    test_summary = f"Test Results: {passed}/{total} passed\n"
    for idx, r in enumerate(test_cases):
        status = "PASS" if r.get("passed") else "FAIL"
        test_summary += (
            f"  [{idx + 1}] {status}"
            f" | Input: {str(r.get('input', ''))[:100]}"
            f" | Expected: {str(r.get('expected', ''))[:100]}"
            f" | Got: {str(r.get('actual', ''))[:100]}\n"
        )
        if r.get("time_ms") is not None:
            test_summary += f"       Time: {r['time_ms']}ms"
        if r.get("memory_kb") is not None:
            test_summary += f" | Memory: {r['memory_kb']}KB"
        if r.get("judge0_status"):
            test_summary += f" | Status: {r['judge0_status']}"
        test_summary += "\n"

    if execution_results.get("compile_error"):
        test_summary += f"\nCompile Error:\n{execution_results['compile_error'][:500]}\n"
    if execution_results.get("runtime_error"):
        test_summary += f"\nRuntime Error:\n{execution_results['runtime_error'][:500]}\n"

    exec_metrics = ""
    if execution_results.get("execution_time_ms") is not None:
        exec_metrics += f"Total Execution Time: {execution_results['execution_time_ms']}ms\n"
    if execution_results.get("memory_usage_kb") is not None:
        exec_metrics += f"Peak Memory Usage: {execution_results['memory_usage_kb']}KB\n"
    if execution_results.get("judge0_status"):
        exec_metrics += f"Overall Judge0 Status: {execution_results['judge0_status']}\n"

    # ── Include expected complexities from the problem ──
    expected_time = problem.get("time_complexity_expected", "N/A")
    expected_space = problem.get("space_complexity_expected", "N/A")

    user_content = (
        f"Candidate: {candidate_name}\n"
        f"Applied Role: {applied_role}\n"
        f"Language: {language}\n\n"
        f"=== Problem ===\n"
        f"Title: {problem.get('title', 'N/A')}\n"
        f"Description:\n{problem.get('description', 'N/A')[:1500]}\n\n"
        f"Expected Optimal Complexity: Time {expected_time}, Space {expected_space}\n\n"
        f"=== Candidate's Code ===\n```{language}\n{code[:6000]}\n```\n\n"
        f"=== Execution Summary ===\n{exec_metrics}\n"
        f"=== Hidden Test Case Results ===\n{test_summary}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": CODE_EVALUATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call)

    if not raw:
        return _fallback_evaluation("")

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return _fallback_evaluation(raw)


def _fallback_evaluation(raw_text: str) -> dict:
    """Return a structured fallback when AI response cannot be parsed."""
    _empty_section = {"score": 0, "assessment": "Could not be determined."}
    return {
        "overall_score": 0,
        "recommendation": "Hold",
        "overall_summary": raw_text[:500] if raw_text else "Evaluation could not be parsed.",
        "correctness": {**_empty_section},
        "complexity": {"time_complexity": "N/A", "space_complexity": "N/A", "assessment": "Could not be determined."},
        "code_quality": {**_empty_section},
        "naming_conventions": {**_empty_section},
        "edge_case_handling": {**_empty_section},
        "optimization_suggestions": [],
        "strengths": [],
        "weaknesses": [],
        "hiring_notes": "Auto-generated evaluation failed. Review submission manually.",
        "skills_demonstrated": [],
        "recommended_topics": [],
        "recruiter_insight": "Evaluation could not be generated. Please review the submission manually.",
    }


# ─── Multi-Question AI Evaluation ──────────────────────────────────────────

MULTI_QUESTION_EVAL_PROMPT = """You are a senior software engineer generating a comprehensive coding assessment review for a candidate who completed a 4-question coding challenge (Easy, Medium, Hard, Expert).

Analyze ALL four questions, their code submissions, execution results, and test case outcomes.
Return a valid JSON object matching this exact schema:

{
  "overall_score": 7.5,
  "recommendation": "Hire",
  "overall_summary": "4-5 sentence comprehensive evaluation of the candidate's coding ability across all difficulty levels.",
  "correctness": {
    "score": 8,
    "assessment": "Detailed assessment of correctness across all problems, noting which were solved optimally and which had issues."
  },
  "complexity": {
    "time_complexity": "Varies by problem",
    "space_complexity": "Varies by problem",
    "assessment": "Analysis of the candidate's understanding of time/space complexity across difficulty levels."
  },
  "code_quality": {
    "score": 7,
    "assessment": "Evaluation of code structure, readability, and best practices across all submissions."
  },
  "naming_conventions": {
    "score": 8,
    "assessment": "Review of naming consistency and clarity across all code."
  },
  "edge_case_handling": {
    "score": 6,
    "assessment": "How well the candidate handled edge cases across different difficulty levels."
  },
  "optimization_suggestions": [
    "Specific actionable suggestion for improvement"
  ],
  "strengths": [
    "Specific strength demonstrated with evidence"
  ],
  "weaknesses": [
    "Specific weakness or area for improvement"
  ],
  "hiring_notes": "Private notes for the hiring panel.",
  "skills_demonstrated": ["Array Manipulation", "Hash Map", "Recursion"],
  "recommended_topics": ["Dynamic Programming", "Graph Traversal"],
  "recruiter_insight": "2-3 sentence note for non-technical recruiters."
}

Rules:
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- All score fields are on a 0-10 scale.
- Overall score should reflect performance across ALL questions (weighted by difficulty).
- Skills demonstrated should cover the full range of problems solved.
- recommended_topics should focus on areas where the candidate struggled.
- Return ONLY valid JSON with no markdown fences."""


async def generate_multi_question_evaluation(
    question_results: list,
    candidate_name: str,
    applied_role: str,
    language: str,
) -> dict:
    """
    Generate a comprehensive AI evaluation for a 4-question coding assessment.
    Analyzes all questions together to produce an overall assessment.
    """
    client = get_cerebras_client()

    # ── Build summary for all questions ──
    questions_summary = ""
    total_passed = 0
    total_cases = 0

    for i, qr in enumerate(question_results):
        passed = qr.get("passed", 0)
        total = qr.get("total", 0)
        total_passed += passed
        total_cases += total

        questions_summary += f"\n{'='*60}\n"
        questions_summary += f"Question {i + 1}: {qr.get('title', 'N/A')} ({qr.get('difficulty', 'N/A').upper()})\n"
        questions_summary += f"Category: {qr.get('category', 'N/A')}\n"
        questions_summary += f"Status: {qr.get('status', 'N/A')}\n"
        questions_summary += f"Submissions: {qr.get('submissions_count', 0)}\n"
        questions_summary += f"Test Results: {passed}/{total} passed\n"

        if qr.get("code"):
            questions_summary += f"\nCandidate's Code:\n```{qr.get('language', language)}\n{qr['code'][:3000]}\n```\n"

        exec_results = qr.get("execution_results")
        if exec_results:
            results = exec_results.get("results", [])
            for idx, r in enumerate(results):
                status = "PASS" if r.get("passed") else "FAIL"
                questions_summary += (
                    f"  [{idx + 1}] {status}"
                    f" | Input: {str(r.get('input', ''))[:80]}"
                    f" | Expected: {str(r.get('expected', ''))[:80]}"
                    f" | Got: {str(r.get('actual', ''))[:80]}\n"
                )
                if r.get("time_ms") is not None:
                    questions_summary += f"       Time: {r['time_ms']}ms"
                if r.get("memory_kb") is not None:
                    questions_summary += f" | Memory: {r['memory_kb']}KB"
                if r.get("judge0_status"):
                    questions_summary += f" | Status: {r['judge0_status']}"
                questions_summary += "\n"

            if exec_results.get("compile_error"):
                questions_summary += f"\nCompile Error:\n{exec_results['compile_error'][:300]}\n"
            if exec_results.get("runtime_error"):
                questions_summary += f"\nRuntime Error:\n{exec_results['runtime_error'][:300]}\n"

    user_content = (
        f"Candidate: {candidate_name}\n"
        f"Applied Role: {applied_role}\n"
        f"Language: {language}\n"
        f"Total Test Cases: {total_passed}/{total_cases} passed\n"
        f"\n=== Question Details ===\n{questions_summary}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": MULTI_QUESTION_EVAL_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call)

    if not raw:
        return _fallback_evaluation("")

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return _fallback_evaluation(raw)
