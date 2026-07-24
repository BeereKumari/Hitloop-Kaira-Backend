import os
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger("judge0")

JUDGE0_BASE_URL = os.getenv("JUDGE0_URL", "").rstrip("/")
JUDGE0_AUTH_TOKEN = os.getenv("JUDGE0_AUTH_TOKEN", "")
MAX_OUTPUT_LIMIT = 10240

# ─── Judge0 Language IDs ────────────────────────────────────────────────────
# https://judge0.com/api/languages

LANGUAGE_MAP = {
    "python": 71,      # Python 3.11
    "javascript": 63,   # Node.js 20
    "java": 62,         # Java 17
    "c": 50,            # GCC 12
    "cpp": 54,          # G++ 12
}

# ─── Judge0 Execution Status Codes ──────────────────────────────────────────

STATUS_IN_QUEUE = 1
STATUS_PROCESSING = 2
STATUS_ACCEPTED = 3
STATUS_WRONG_ANSWER = 4
STATUS_TLE = 5
STATUS_COMPILE_ERROR = 6
STATUS_RUNTIME_ERROR_SIGSEGV = 7
STATUS_RUNTIME_ERROR_SIGXFSZ = 8
STATUS_RUNTIME_ERROR_SIGFPE = 9
STATUS_RUNTIME_ERROR_SIGABRT = 10
STATUS_RUNTIME_ERROR_NZEC = 11
STATUS_RUNTIME_ERROR_OTHER = 12
STATUS_INTERNAL_ERROR = 13
STATUS_EXEC_FORMAT_ERROR = 14

# ─── Human-readable status labels ───────────────────────────────────────────

STATUS_LABELS = {
    STATUS_IN_QUEUE: "In Queue",
    STATUS_PROCESSING: "Processing",
    STATUS_ACCEPTED: "Accepted",
    STATUS_WRONG_ANSWER: "Wrong Answer",
    STATUS_TLE: "Time Limit Exceeded",
    STATUS_COMPILE_ERROR: "Compilation Error",
    STATUS_RUNTIME_ERROR_SIGSEGV: "Runtime Error (SIGSEGV)",
    STATUS_RUNTIME_ERROR_SIGXFSZ: "Runtime Error (SIGXFSZ)",
    STATUS_RUNTIME_ERROR_SIGFPE: "Runtime Error (SIGFPE)",
    STATUS_RUNTIME_ERROR_SIGABRT: "Runtime Error (SIGABRT)",
    STATUS_RUNTIME_ERROR_NZEC: "Runtime Error (NZEC)",
    STATUS_RUNTIME_ERROR_OTHER: "Runtime Error",
    STATUS_INTERNAL_ERROR: "Internal Error",
    STATUS_EXEC_FORMAT_ERROR: "Exec Format Error",
}


def get_status_label(status_id: int) -> str:
    """Map a Judge0 status ID to a human-readable label."""
    return STATUS_LABELS.get(status_id, f"Unknown Status ({status_id})")


def is_runtime_error(status_id: int) -> bool:
    """Check if status ID corresponds to a runtime error."""
    return status_id in (
        STATUS_RUNTIME_ERROR_SIGSEGV,
        STATUS_RUNTIME_ERROR_SIGXFSZ,
        STATUS_RUNTIME_ERROR_SIGFPE,
        STATUS_RUNTIME_ERROR_SIGABRT,
        STATUS_RUNTIME_ERROR_NZEC,
        STATUS_RUNTIME_ERROR_OTHER,
        STATUS_EXEC_FORMAT_ERROR,
    )


def _get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if JUDGE0_AUTH_TOKEN:
        headers["X-Auth-Token"] = JUDGE0_AUTH_TOKEN
    return headers


# ─── Code Harness Wrapping ──────────────────────────────────────────────────
# Wraps user function code with stdin-reading / stdout-writing boilerplate
# so Judge0 can execute it. Modeled after LeetCode's test harness approach.

def _extract_func_name(func_signature: str) -> str:
    """Extract the function name from a signature string like 'def solve(nums):'"""
    if not func_signature:
        return "solve"
    m = re.search(r'(?:def|function)\s+(\w+)\s*\(|(?:public\s+\w+\s+)?(\w+)\s*\(', func_signature)
    if m:
        return m.group(1) or m.group(2)
    return "solve"


def wrap_code_with_harness(code: str, language: str, problem: dict) -> str:
    """
    Wrap user-written function code with a language-specific harness that:
      1. Reads stdin (one JSON value per line)
      2. Calls the user's function
      3. Prints the result as JSON to stdout

    This bridges the gap between "implement this function" problem style
    and Judge0's stdin/stdout execution model.
    """
    func_name = _extract_func_name(problem.get("function_signature", ""))

    if language == "python":
        return _wrap_python(code, func_name)
    elif language == "javascript":
        return _wrap_javascript(code, func_name)
    elif language == "java":
        return _wrap_java(code, func_name)
    elif language == "c":
        return _wrap_c(code, func_name)
    elif language == "cpp":
        return _wrap_cpp(code, func_name)
    return code


def _wrap_python(code: str, func_name: str) -> str:
    return f'''import sys, json, ast

def __harness__():
    raw = sys.stdin.read().strip()
    if not raw:
        return
    lines = [l.strip() for l in raw.split("\\n") if l.strip()]
    args = []
    for line in lines:
        try:
            args.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            try:
                args.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                args.append(line)
    result = {func_name}(*args)
    if isinstance(result, (list, dict)):
        print(json.dumps(result, separators=(",", ":")))
    elif isinstance(result, bool):
        print("true" if result else "false")
    elif result is None:
        print("null")
    else:
        print(result)

{code}

if __name__ == "__main__":
    __harness__()
'''


def _wrap_javascript(code: str, func_name: str) -> str:
    return f'''const __fs = require("fs");
const __raw = __fs.readFileSync("/dev/stdin", "utf8").trim();
const __lines = __raw.split("\\n").filter(l => l.trim());
const __args = __lines.map(l => {{
  try {{ return JSON.parse(l); }} catch {{ return l; }}
}});

{code}

const __result = {func_name}(...__args);
if (Array.isArray(__result) || (typeof __result === "object" && __result !== null)) {{
  console.log(JSON.stringify(__result));
}} else if (typeof __result === "boolean") {{
  console.log(__result ? "true" : "false");
}} else if (__result === null || __result === undefined) {{
  console.log("null");
}} else {{
  console.log(__result);
}}
'''


def _extract_java_imports(code: str) -> tuple[str, str]:
    """Extract import statements from Java code and return (imports, remaining_code)."""
    import_lines = []
    other_lines = []
    for line in code.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") and trimmed.endswith(";"):
            import_lines.append(line)
        else:
            other_lines.append(line)
    return "\n".join(import_lines), "\n".join(other_lines)


def _wrap_java(code: str, func_name: str) -> str:
    user_imports, user_code = _extract_java_imports(code)
    return f'''import java.util.*;
import java.io.*;
{user_imports}

public class Main {{
    public static void main(String[] args) throws Exception {{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        List<Object> parsedArgs = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {{
            line = line.trim();
            if (line.isEmpty()) continue;
            parsedArgs.add(parseValue(line));
        }}

        Solution sol = new Solution();
        Object[] methodArgs = parsedArgs.toArray();
        
        java.lang.reflect.Method solveMethod = null;
        for (java.lang.reflect.Method m : Solution.class.getDeclaredMethods()) {{
            if (m.getName().equals("{func_name}")) {{
                solveMethod = m;
                break;
            }}
        }}
        
        if (solveMethod == null) {{
            throw new RuntimeException("Method '{func_name}' not found in class Solution");
        }}

        Class<?>[] paramTypes = solveMethod.getParameterTypes();
        Object[] convertedArgs = new Object[paramTypes.length];
        for (int i = 0; i < paramTypes.length; i++) {{
            if (i < methodArgs.length) {{
                convertedArgs[i] = convertType(methodArgs[i], paramTypes[i]);
            }} else {{
                convertedArgs[i] = null;
            }}
        }}

        Object result = solveMethod.invoke(sol, convertedArgs);
        System.out.println(formatResult(result));
    }}

    private static Object parseValue(String val) {{
        val = val.trim();
        if (val.equals("true")) return true;
        if (val.equals("false")) return false;
        if (val.equals("null")) return null;
        if (val.startsWith("[") && val.endsWith("]")) {{
            String content = val.substring(1, val.length() - 1).trim();
            if (content.isEmpty()) return new ArrayList<Object>();
            List<Object> list = new ArrayList<>();
            List<String> tokens = splitJsonArray(content);
            for (String tok : tokens) {{
                list.add(parseValue(tok));
            }}
            return list;
        }}
        if (val.startsWith("\\\"") && val.endsWith("\\\"")) {{
            return val.substring(1, val.length() - 1);
        }}
        if (val.startsWith("'") && val.endsWith("'")) {{
            return val.substring(1, val.length() - 1);
        }}
        try {{
            if (val.contains(".")) {{
                return Double.parseDouble(val);
            }} else {{
                return Integer.parseInt(val);
            }}
        }} catch (NumberFormatException e) {{
            return val;
        }}
    }}

    private static List<String> splitJsonArray(String content) {{
        List<String> result = new ArrayList<>();
        int bracketDepth = 0;
        boolean inQuotes = false;
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < content.length(); i++) {{
            char c = content.charAt(i);
            if (c == '"' && (i == 0 || content.charAt(i - 1) != '\\\\')) {{
                inQuotes = !inQuotes;
            }}
            if (!inQuotes) {{
                if (c == '[' || c == '{{') bracketDepth++;
                if (c == ']' || c == '}}') bracketDepth--;
            }}
            if (c == ',' && bracketDepth == 0 && !inQuotes) {{
                result.add(current.toString().trim());
                current = new StringBuilder();
            }} else {{
                current.append(c);
            }}
        }}
        if (current.length() > 0) {{
            result.add(current.toString().trim());
        }}
        return result;
    }}

    private static Object convertType(Object obj, Class<?> targetType) {{
        if (obj == null) return null;
        if (targetType.isInstance(obj)) return obj;

        if (obj instanceof List) {{
            List<?> list = (List<?>) obj;
            if (targetType == int[].class) {{
                int[] arr = new int[list.size()];
                for (int i = 0; i < list.size(); i++) {{
                    arr[i] = ((Number) list.get(i)).intValue();
                }}
                return arr;
            }}
            if (targetType == double[].class) {{
                double[] arr = new double[list.size()];
                for (int i = 0; i < list.size(); i++) {{
                    arr[i] = ((Number) list.get(i)).doubleValue();
                }}
                return arr;
            }}
            if (targetType == String[].class) {{
                String[] arr = new String[list.size()];
                for (int i = 0; i < list.size(); i++) {{
                    arr[i] = list.get(i).toString();
                }}
                return arr;
            }}
            if (targetType == List.class || targetType == Collection.class) {{
                return list;
            }}
        }}
        
        if (obj instanceof Number) {{
            Number num = (Number) obj;
            if (targetType == int.class || targetType == Integer.class) return num.intValue();
            if (targetType == double.class || targetType == Double.class) return num.doubleValue();
            if (targetType == long.class || targetType == Long.class) return num.longValue();
        }}

        return obj;
    }}

    private static String formatResult(Object result) {{
        if (result == null) return "null";
        if (result instanceof Boolean) return result.toString();
        if (result instanceof int[]) return Arrays.toString((int[]) result).replace(" ", "");
        if (result instanceof double[]) return Arrays.toString((double[]) result).replace(" ", "");
        if (result instanceof Object[]) return Arrays.deepToString((Object[]) result).replace(" ", "");
        if (result instanceof Collection) {{
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            boolean first = true;
            for (Object item : (Collection<?>) result) {{
                if (!first) sb.append(",");
                sb.append(formatResult(item));
                first = false;
            }}
            sb.append("]");
            return sb.toString();
        }}
        return result.toString();
    }}
}}

{user_code}
'''


def _wrap_c(code: str, func_name: str) -> str:
    return f'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

{code}

int main() {{
    solve();
    return 0;
}}
'''


def _wrap_cpp(code: str, func_name: str) -> str:
    return f'''#include <bits/stdc++.h>
using namespace std;

{code}

int main() {{
    solve();
    return 0;
}}
'''


async def run_code_with_test_cases(
    code: str,
    language: str,
    test_cases: List[Dict[str, str]],
    timeout_seconds: int = 10,
) -> dict:
    """
    Execute code against test cases via Judge0.

    Returns:
    {
        passed: int,
        total: int,
        results: [{ input, expected, actual, passed, explanation, time_ms, memory_kb, judge0_status }],
        compile_error: str | None,
        runtime_error: str | None,
        execution_time_ms: int | None,    # sum of all test case times
        memory_usage_kb: int | None,       # peak memory across all test cases
        judge0_status: str,                # overall status description
    }
    """
    if not JUDGE0_BASE_URL:
        return _fallback_unavailable(language, test_cases)

    if language not in LANGUAGE_MAP:
        return {
            "passed": 0,
            "total": len(test_cases),
            "results": [],
            "compile_error": f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}",
            "runtime_error": None,
            "execution_time_ms": None,
            "memory_usage_kb": None,
            "judge0_status": "Unsupported Language",
        }

    judge0_language_id = LANGUAGE_MAP[language]
    results = []
    passed = 0
    compile_error = None
    runtime_error = None
    total_time_ms = 0
    peak_memory_kb = 0
    overall_status = "Accepted"

    for tc in test_cases:
        result = await _execute_single_test(
            code, judge0_language_id, tc, timeout_seconds
        )
        results.append(result)
        if result["passed"]:
            passed += 1
        else:
            if result.get("error_type") == "compile" and not compile_error:
                compile_error = result["actual"]
                overall_status = result.get("judge0_status", "Compilation Error")
            elif result.get("error_type") == "runtime" and not runtime_error:
                runtime_error = result["actual"]
                overall_status = result.get("judge0_status", "Runtime Error")
            elif not overall_status or overall_status == "Accepted":
                overall_status = result.get("judge0_status", "Wrong Answer")

        t = result.get("time_ms")
        if t is not None:
            total_time_ms += t
        m = result.get("memory_kb")
        if m is not None and m > peak_memory_kb:
            peak_memory_kb = m

    return {
        "passed": passed,
        "total": len(test_cases),
        "results": [
            {
                "input": r["input"],
                "expected": r["expected"],
                "actual": r["actual"],
                "passed": r["passed"],
                "explanation": r.get("explanation"),
                "time_ms": r.get("time_ms"),
                "memory_kb": r.get("memory_kb"),
                "judge0_status": r.get("judge0_status"),
            }
            for r in results
        ],
        "compile_error": compile_error,
        "runtime_error": runtime_error,
        "execution_time_ms": total_time_ms if any(r.get("time_ms") is not None for r in results) else None,
        "memory_usage_kb": peak_memory_kb if peak_memory_kb > 0 else None,
        "judge0_status": overall_status,
    }


async def run_code_once(
    code: str,
    language: str,
    stdin_data: str = "",
    timeout_seconds: int = 10,
) -> dict:
    """
    Execute code once against custom stdin via Judge0.
    Returns { stdout, compile_error, runtime_error, stderr, time_ms, memory_kb, judge0_status }.
    """
    if not JUDGE0_BASE_URL:
        return {
            "stdout": "",
            "compile_error": "Code execution engine is not configured. Set JUDGE0_BASE_URL in environment.",
            "runtime_error": None,
            "stderr": None,
            "time_ms": None,
            "memory_kb": None,
            "judge0_status": "Unavailable",
        }

    if language not in LANGUAGE_MAP:
        return {
            "stdout": "",
            "compile_error": f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}",
            "runtime_error": None,
            "stderr": None,
            "time_ms": None,
            "memory_kb": None,
            "judge0_status": "Unsupported Language",
        }

    judge0_language_id = LANGUAGE_MAP[language]

    try:
        submission_token = await submit_code(code, judge0_language_id, stdin_data)
        result = await poll_result(submission_token, timeout_seconds)
    except asyncio.TimeoutError:
        return {
            "stdout": "",
            "compile_error": None,
            "runtime_error": "Time Limit Exceeded",
            "stderr": None,
            "time_ms": None,
            "memory_kb": None,
            "judge0_status": "Time Limit Exceeded",
        }
    except Exception as e:
        return {
            "stdout": "",
            "compile_error": f"Execution error: {str(e)[:500]}",
            "runtime_error": None,
            "stderr": None,
            "time_ms": None,
            "memory_kb": None,
            "judge0_status": "Execution Error",
        }

    status_id = result.get("status", {}).get("id", 0)
    status_desc = get_status_label(status_id)
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    compile_output = (result.get("compile_output") or "").strip()
    message = (result.get("message") or "").strip()
    time_s = result.get("time")
    mem_kb = result.get("memory")

    time_ms = int(float(time_s) * 1000) if time_s is not None else None

    if status_id == STATUS_ACCEPTED:
        return {
            "stdout": stdout,
            "compile_error": None,
            "runtime_error": None,
            "stderr": None,
            "time_ms": time_ms,
            "memory_kb": mem_kb,
            "judge0_status": status_desc,
        }

    if status_id == STATUS_COMPILE_ERROR:
        detail = compile_output or stderr or status_desc
        return {
            "stdout": "",
            "compile_error": detail[:1500],
            "runtime_error": None,
            "stderr": detail[:1500],
            "time_ms": time_ms,
            "memory_kb": mem_kb,
            "judge0_status": status_desc,
        }

    if status_id == STATUS_TLE:
        return {
            "stdout": stdout[:500] if stdout else "",
            "compile_error": None,
            "runtime_error": "Time Limit Exceeded",
            "stderr": None,
            "time_ms": time_ms,
            "memory_kb": mem_kb,
            "judge0_status": status_desc,
        }

    if is_runtime_error(status_id):
        detail = stderr or compile_output or message or status_desc
        return {
            "stdout": stdout[:500] if stdout else "",
            "compile_error": None,
            "runtime_error": detail[:1500],
            "stderr": detail[:1500],
            "time_ms": time_ms,
            "memory_kb": mem_kb,
            "judge0_status": status_desc,
        }

    if status_id == STATUS_WRONG_ANSWER:
        return {
            "stdout": stdout or "",
            "compile_error": None,
            "runtime_error": None,
            "stderr": None,
            "time_ms": time_ms,
            "memory_kb": mem_kb,
            "judge0_status": status_desc,
        }

    detail = stderr or compile_output or message or status_desc
    return {
        "stdout": stdout[:500] if stdout else "",
        "compile_error": None,
        "runtime_error": detail[:1500] or f"Execution failed: {status_desc}",
        "stderr": detail[:1500],
        "time_ms": time_ms,
        "memory_kb": mem_kb,
        "judge0_status": status_desc,
    }


async def _execute_single_test(
    code: str,
    language_id: int,
    test_case: dict,
    timeout: int,
    max_retries: int = 2,
) -> dict:
    stdin_data = test_case.get("input", "")
    expected = test_case.get("expected_output", "")
    explanation = test_case.get("explanation")

    base_result = {
        "input": stdin_data,
        "expected": expected,
        "explanation": explanation,
    }

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            submission_token = await submit_code(code, language_id, stdin_data)
            result = await poll_result(submission_token, timeout)

            status_id = result.get("status", {}).get("id", 0)
            return parse_result(result, base_result)

        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.warning("Judge0 poll timeout (attempt %d/%d), retrying", attempt + 1, max_retries + 1)
                await asyncio.sleep(0.5)
                continue
            return {
                **base_result,
                "actual": "Time Limit Exceeded",
                "passed": False,
                "error_type": "runtime",
                "time_ms": None,
                "memory_kb": None,
                "judge0_status": "Time Limit Exceeded",
            }
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning("Judge0 error (attempt %d/%d): %s", attempt + 1, max_retries + 1, str(e)[:200])
                await asyncio.sleep(0.5)
                continue
            break

    return {
        **base_result,
        "actual": f"Execution error: {str(last_exception)[:300]}",
        "passed": False,
        "error_type": "compile",
        "time_ms": None,
        "memory_kb": None,
        "judge0_status": "Execution Error",
    }


async def submit_code(code: str, language_id: int, stdin_data: str) -> str:
    url = f"{JUDGE0_BASE_URL}/submissions?base64_encoded=false&wait=false&max_output_limit={MAX_OUTPUT_LIMIT}"
    payload = {
        "source_code": code,
        "language_id": language_id,
        "stdin": stdin_data,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=_get_headers())
        resp.raise_for_status()
        return resp.json()["token"]


async def poll_result(token: str, timeout: int) -> dict:
    url = f"{JUDGE0_BASE_URL}/submissions/{token}?base64_encoded=false&max_output_limit={MAX_OUTPUT_LIMIT}"
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            if asyncio.get_event_loop().time() >= deadline:
                raise asyncio.TimeoutError("Judge0 polling timed out")

            resp = await client.get(url, headers=_get_headers())
            resp.raise_for_status()
            data = resp.json()

            status_id = data.get("status", {}).get("id", 0)
            if status_id not in (STATUS_IN_QUEUE, STATUS_PROCESSING):
                return data

            await asyncio.sleep(0.3)


def parse_result(data: dict, base_result: dict) -> dict:
    status_id = data.get("status", {}).get("id", 0)
    status_desc = get_status_label(status_id)
    stdout = (data.get("stdout") or "").strip()
    stderr = (data.get("stderr") or "").strip()
    compile_output = (data.get("compile_output") or "").strip()
    time_s = data.get("time")
    mem_kb = data.get("memory")

    time_ms = int(float(time_s) * 1000) if time_s is not None else None

    metrics = {
        "time_ms": time_ms,
        "memory_kb": mem_kb,
        "judge0_status": status_desc,
    }

    if status_id == STATUS_ACCEPTED:
        expected = base_result["expected"].strip()
        passed = stdout == expected
        return {
            **base_result,
            "actual": stdout,
            "passed": passed,
            "error_type": None,
            **metrics,
            "judge0_status": status_desc if passed else "Wrong Answer",
        }

    if status_id == STATUS_COMPILE_ERROR:
        error_detail = compile_output or stderr or status_desc
        return {
            **base_result,
            "actual": error_detail[:500],
            "passed": False,
            "error_type": "compile",
            **metrics,
        }

    if status_id == STATUS_TLE:
        return {
            **base_result,
            "actual": "Time Limit Exceeded",
            "passed": False,
            "error_type": "runtime",
            **metrics,
        }

    if is_runtime_error(status_id):
        error_detail = stderr or status_desc
        return {
            **base_result,
            "actual": error_detail[:500] or f"Runtime error: {status_desc}",
            "passed": False,
            "error_type": "runtime",
            **metrics,
        }

    if status_id == STATUS_WRONG_ANSWER:
        return {
            **base_result,
            "actual": stdout or "Wrong Answer (empty output)",
            "passed": False,
            "error_type": None,
            **metrics,
        }

    # Any other status (Internal Error, Exec Format Error, etc.)
    return {
        **base_result,
        "actual": f"Execution failed: {status_desc}",
        "passed": False,
        "error_type": "runtime",
        **metrics,
    }


async def run_code_against_sample_cases(
    code: str,
    language: str,
    test_cases: List[Dict[str, str]],
    timeout_seconds: int = 10,
) -> dict:
    """
    Execute code against sample (visible) test cases for the 'Run' button.
    Only runs non-hidden test cases. Returns structured results with per-case details.
    """
    if not JUDGE0_BASE_URL:
        return _fallback_unavailable(language, test_cases)

    if language not in LANGUAGE_MAP:
        return {
            "passed": 0,
            "total": len(test_cases),
            "results": [],
            "compile_error": f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}",
            "runtime_error": None,
            "execution_time_ms": None,
            "memory_usage_kb": None,
            "judge0_status": "Unsupported Language",
        }

    judge0_language_id = LANGUAGE_MAP[language]
    results = []
    passed = 0
    compile_error = None
    runtime_error = None
    total_time_ms = 0
    peak_memory_kb = 0
    overall_status = "Accepted"

    for tc in test_cases:
        result = await _execute_single_test(
            code, judge0_language_id, tc, timeout_seconds
        )
        results.append(result)
        if result["passed"]:
            passed += 1
        else:
            if result.get("error_type") == "compile" and not compile_error:
                compile_error = result["actual"]
                overall_status = result.get("judge0_status", "Compilation Error")
            elif result.get("error_type") == "runtime" and not runtime_error:
                runtime_error = result["actual"]
                overall_status = result.get("judge0_status", "Runtime Error")
            elif overall_status == "Accepted":
                overall_status = result.get("judge0_status", "Wrong Answer")

        t = result.get("time_ms")
        if t is not None:
            total_time_ms += t
        m = result.get("memory_kb")
        if m is not None and m > peak_memory_kb:
            peak_memory_kb = m

    total = len(test_cases)
    score = round((passed / total) * 10, 1) if total > 0 else 0.0

    return {
        "passed": passed,
        "total": total,
        "score": score,
        "results": [
            {
                "input": r["input"],
                "expected": r["expected"],
                "actual": r["actual"],
                "passed": r["passed"],
                "explanation": r.get("explanation"),
                "time_ms": r.get("time_ms"),
                "memory_kb": r.get("memory_kb"),
                "judge0_status": r.get("judge0_status"),
            }
            for r in results
        ],
        "compile_error": compile_error,
        "runtime_error": runtime_error,
        "execution_time_ms": total_time_ms if any(r.get("time_ms") is not None for r in results) else None,
        "memory_usage_kb": peak_memory_kb if peak_memory_kb > 0 else None,
        "judge0_status": overall_status,
    }


def _fallback_unavailable(language: str, test_cases: List[Dict]) -> dict:
    """Return when JUDGE0_BASE_URL is not configured."""
    return {
        "passed": 0,
        "total": len(test_cases),
        "results": [],
        "compile_error": "Code execution engine is not configured. Set JUDGE0_BASE_URL in environment.",
        "runtime_error": None,
        "time_ms": None,
    }


def get_supported_languages() -> list:
    return list(LANGUAGE_MAP.keys())
