from pydantic import BaseModel, Field
from typing import Optional, List


# ─── Request Models ──────────────────────────────────────────────────────────

class StartAssessmentRequest(BaseModel):
    difficulty: str = Field("medium", description="easy, medium, hard, expert")
    language: str = Field("python", description="python, javascript, java, c, cpp")
    category: Optional[str] = Field(None, description="e.g. arrays, trees, strings, system_design")


class StartFullAssessmentRequest(BaseModel):
    language: str = Field("python", description="python, javascript, java, c, cpp")
    category: Optional[str] = Field(None, description="e.g. arrays, trees, strings, system_design")


class SubmitCodeRequest(BaseModel):
    session_id: str
    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field("python")
    custom_input: Optional[str] = Field(None, description="If provided, run once against this stdin and return raw output instead of running against test cases")


class RunSampleRequest(BaseModel):
    session_id: str
    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field("python")
    question_index: int = Field(0, description="Index of the question to run against (0-based)")


class SubmitQuestionRequest(BaseModel):
    session_id: str
    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field("python")
    question_index: int = Field(0, description="Index of the question to submit (0-based)")


class EndAssessmentRequest(BaseModel):
    session_id: str


# ─── Execution Types (used by services) ─────────────────────────────────────

class TestCase(BaseModel):
    input: str
    expected_output: str
    is_hidden: bool = False
    explanation: Optional[str] = None


class TestResult(BaseModel):
    input: str
    expected: str
    actual: str
    passed: bool
    explanation: Optional[str] = None


class CodeExecutionResult(BaseModel):
    passed: int
    total: int
    results: List[TestResult]
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None
    time_ms: Optional[int] = None


# ─── Evaluation Service Models ─────────────────────────────────────────────

class EvaluationTestResult(BaseModel):
    """Result of evaluating code against a single test case."""
    test_case_index: int
    input: str
    expected: str
    actual: str
    passed: bool
    explanation: Optional[str] = None
    time_ms: Optional[int] = None
    memory_kb: Optional[int] = None
    judge0_status: Optional[str] = None


class EvaluationReport(BaseModel):
    """Structured report returned by the evaluation service."""
    passed: int
    total: int
    percentage: float = Field(..., description="Pass percentage 0–100")
    score: float = Field(..., description="Score on configured scale (default 0–10)")
    score_scale: int = Field(10, description="Maximum score value")
    results: List[EvaluationTestResult]
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    memory_usage_kb: Optional[int] = None
    judge0_status: str = "Unknown"


# ─── Question Document Models ───────────────────────────────────────────────

class SampleTestCase(BaseModel):
    input: str
    output: str
    explanation: Optional[str] = None


class HiddenTestCase(BaseModel):
    input: str
    expected_output: str
    explanation: Optional[str] = None


class CodingQuestion(BaseModel):
    title: str
    description: str
    difficulty: str = Field("medium", description="easy, medium, hard, expert")
    languages: List[str] = Field(default_factory=lambda: ["python"])
    starter_code: Optional[str] = None
    sample_test_cases: List[SampleTestCase] = Field(default_factory=list)
    test_cases: List[HiddenTestCase] = Field(default_factory=list)
    constraints: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    time_limit_seconds: int = 10
    memory_limit_mb: int = 256
    category: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


# ─── Submission Document Models ─────────────────────────────────────────────

class ExecutionResult(BaseModel):
    passed: int = 0
    total: int = 0
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None


class CodingSubmission(BaseModel):
    candidate_id: str
    question_id: str
    language: str
    code: str
    execution_result: ExecutionResult = ExecutionResult()
    passed_test_cases: int = 0
    failed_test_cases: int = 0
    execution_time_ms: Optional[int] = None
    memory_usage_kb: Optional[int] = None
    submitted_at: str = ""
    status: str = Field("pending", description="pending, running, completed, failed")
