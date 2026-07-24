"""
Coding Assessment API — AI-generated coding problems with automated test execution.

Endpoints (Legacy - single question):
  POST /api/coding/start        — Generate a new coding problem and start session
  POST /api/coding/submit       — Submit code for evaluation against test cases
  POST /api/coding/end          — End session, generate AI evaluation of submission

Endpoints (Full Assessment - 4 questions):
  POST /api/coding/start-assessment  — Start a full 4-question assessment
  POST /api/coding/run-sample        — Run code against visible sample test cases
  POST /api/coding/submit-question   — Submit code for a specific question
  GET  /api/coding/question-status   — Get current question state and progress

Shared:
  GET  /api/coding/session/{id}         — Retrieve assessment session history
  GET  /api/coding/evaluation/{session_id} — Retrieve final evaluation
"""

import uuid
import asyncio
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends

from app.database.database import db
from app.middleware.auth import get_current_user_optional
from app.models.coding import (
    StartAssessmentRequest,
    StartFullAssessmentRequest,
    SubmitCodeRequest,
    RunSampleRequest,
    SubmitQuestionRequest,
    EndAssessmentRequest,
)
from app.services.coding_service import (
    generate_coding_problem,
    generate_coding_evaluation,
    generate_multi_question_evaluation,
)
from app.services.evaluation_service import evaluate_code
from app.services.judge0_service import (
    get_supported_languages,
    run_code_once,
    run_code_against_sample_cases,
    wrap_code_with_harness,
)

router = APIRouter(prefix="/api/coding", tags=["Coding Assessment"])

# ─── Difficulty Config ──────────────────────────────────────────────────────

DIFFICULTY_CONFIG = {
    "easy": {
        "label": "Easy",
        "description": "Basic data structures, simple algorithms, fundamental syntax",
        "time_limit_seconds": 15,
        "timer_minutes": 15,
    },
    "medium": {
        "label": "Medium",
        "description": "Intermediate algorithms, data structure manipulation, moderate complexity",
        "time_limit_seconds": 12,
        "timer_minutes": 25,
    },
    "hard": {
        "label": "Hard",
        "description": "Advanced algorithms, optimization, complex data structures",
        "time_limit_seconds": 10,
        "timer_minutes": 40,
    },
    "expert": {
        "label": "Expert",
        "description": "System-level problems, advanced optimization, novel algorithm design",
        "time_limit_seconds": 10,
        "timer_minutes": 60,
    },
}

# Question sequence for full assessment
QUESTION_SEQUENCE = ["easy", "medium", "hard", "expert"]


# ─── Helpers ────────────────────────────────────────────────────────────────

async def _load_candidate_context(user_id: str) -> dict:
    """Load candidate profile and resume analysis context."""
    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    analysis_doc = await db["resume_analyses"].find_one(
        {"user_id": {"$in": [user_id, "anonymous"]}},
        sort=[("created_at", -1)],
    )
    analysis = analysis_doc.get("analysis", {}) if analysis_doc else {}

    personal = (profile or {}).get("personal", {})
    skills_obj = (profile or {}).get("skills", {})

    candidate_skills = skills_obj.get("core_skills") or ", ".join(
        analysis.get("extracted_skills", [])
    )
    applied_role = personal.get("applied_role") or analysis.get("target_role", "Software Engineer")
    candidate_name = personal.get("full_name") or analysis.get("candidate_name", "Candidate")

    return {
        "candidate_skills": candidate_skills,
        "applied_role": applied_role,
        "candidate_name": candidate_name,
    }


def _format_problem(problem: dict) -> dict:
    """Extract frontend-friendly problem fields."""
    return {
        "title": problem.get("title", ""),
        "description": problem.get("description", ""),
        "difficulty": problem.get("difficulty", "medium"),
        "category": problem.get("category", ""),
        "function_signature": problem.get("function_signature", ""),
        "examples": problem.get("examples", []),
        "hints": problem.get("hints", []),
        "time_complexity_expected": problem.get("time_complexity_expected", ""),
        "space_complexity_expected": problem.get("space_complexity_expected", ""),
        "test_cases": problem.get("test_cases", []),
    }


# ─── Full Assessment Endpoints (New) ───────────────────────────────────────

@router.post("/start-assessment")
async def start_full_assessment(
    req: StartFullAssessmentRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Start a full multi-question coding assessment.
    Generates tailored problems and creates a session with per-question state.
    """
    if req.language not in get_supported_languages():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Choose from: {get_supported_languages()}",
        )

    user_id = str(current_user["_id"]) if current_user else "anonymous"
    ctx = await _load_candidate_context(user_id)

    # ── Enforce recruiter-scheduled configurations ──
    sequence = ["easy", "medium", "hard", "expert"]
    if user_id != "anonymous":
        schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
        
        # Rule 1: Video interview must be completed and shortlisted
        if not schedule or not schedule.get("video"):
            raise HTTPException(
                status_code=403,
                detail="Coding assessment is locked. You must complete the Video interview and be shortlisted first."
            )
        video_conf = schedule["video"]
        if not video_conf.get("analysis_completed") or video_conf.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Coding assessment is locked. You must complete the Video interview and be shortlisted first."
            )

        # Rule 2: Coding must be configured/scheduled
        if not schedule.get("coding"):
            raise HTTPException(
                status_code=403,
                detail="Coding assessment has not been scheduled by your recruiter yet."
            )
        coding_conf = schedule["coding"]

        # Enforce scheduled time
        scheduled_time = coding_conf.get("scheduled_time")
        if scheduled_time:
            now_str = datetime.utcnow().isoformat()
            if now_str < scheduled_time:
                raise HTTPException(
                    status_code=403,
                    detail=f"Your Coding assessment is scheduled for {scheduled_time}. It has not started yet."
                )

        # Enforce deadline
        deadline = coding_conf.get("deadline")
        if deadline:
            now_str = datetime.utcnow().isoformat()
            if now_str > deadline:
                raise HTTPException(
                    status_code=403,
                    detail=f"The deadline for this Coding assessment has passed ({deadline}). You can no longer start it."
                )

        # Enforce attempt limit
        completed_count = await db["coding_evaluations"].count_documents({"user_id": user_id})
        max_attempts = int(coding_conf.get("max_attempts", 3))
        if completed_count >= max_attempts:
            raise HTTPException(
                status_code=403,
                detail=f"Attempt limit reached. You have used {completed_count}/{max_attempts} allowed attempts. Contact your recruiter to reset."
            )

        # Load complexities from schedule
        complexities = coding_conf.get("complexities")
        if complexities:
            sequence = list(complexities)
        else:
            single_comp = coding_conf.get("complexity", "medium")
            num_q = int(coding_conf.get("num_questions", 1))
            sequence = [single_comp] * num_q

    # ── Generate problems in sequence ──
    problems = []
    for i, difficulty in enumerate(sequence):
        try:
            problem = await generate_coding_problem(
                candidate_skills=ctx["candidate_skills"],
                applied_role=ctx["applied_role"],
                difficulty=difficulty,
                category=req.category,
            )
            problems.append(problem)
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate {difficulty} problem: {str(e)}",
            )
        # Brief delay between Cerebras calls to avoid rate limits
        if i < len(sequence) - 1:
            await asyncio.sleep(1)

    # ── Create session with multi-question state ──
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    questions = []
    for i, (difficulty, problem) in enumerate(zip(sequence, problems)):
        cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
        questions.append({
            "index": i,
            "difficulty": difficulty,
            "timer_minutes": cfg["timer_minutes"],
            "problem": problem,
            "status": "active" if i == 0 else "locked",
            "submissions": [],
            "best_result": None,
            "completed_at": None,
        })

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "candidate_name": ctx["candidate_name"],
        "applied_role": ctx["applied_role"],
        "language": req.language,
        "mode": "full_assessment",
        "questions": questions,
        "current_question_index": 0,
        "status": "active",
        "start_time": now,
        "created_at": now,
        "updated_at": now,
    }

    await db["coding_sessions"].insert_one(session_doc)

    # ── Return first question ──
    first = questions[0]
    first_problem = _format_problem(first["problem"])

    return {
        "status": "success",
        "session_id": session_id,
        "language": req.language,
        "total_questions": len(questions),
        "current_question_index": 0,
        "questions": [
            {
                "index": q["index"],
                "difficulty": q["difficulty"],
                "timer_minutes": q["timer_minutes"],
                "status": q["status"],
                "title": q["problem"].get("title", ""),
            }
            for q in questions
        ],
        "current_question": {
            **first_problem,
            "difficulty": first["difficulty"],
            "timer_minutes": first["timer_minutes"],
            "question_index": 0,
        },
    }


@router.post("/run-sample")
async def run_sample_test_cases(
    req: RunSampleRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Run code against visible sample test cases for the 'Run' button.
    Only executes against non-hidden test cases and returns LeetCode-style results.
    """
    session = await db["coding_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Assessment session not found")
    if session["status"] == "completed":
        raise HTTPException(status_code=400, detail="This assessment session has already been completed")

    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    qi = req.question_index
    questions = session.get("questions", [])

    if qi < 0 or qi >= len(questions):
        raise HTTPException(status_code=400, detail=f"Invalid question index: {qi}")

    question = questions[qi]
    if question.get("status") == "locked":
        raise HTTPException(status_code=400, detail="This question is locked. Complete the previous question first.")

    problem = question.get("problem", {})
    test_cases = problem.get("test_cases", [])

    # Filter to visible (non-hidden) sample test cases
    sample_cases = [tc for tc in test_cases if not tc.get("is_hidden", False)]
    if not sample_cases:
        # Fallback: use examples as sample cases
        sample_cases = [
            {"input": ex.get("input", ""), "expected_output": ex.get("output", ""), "is_hidden": False}
            for ex in problem.get("examples", [])
        ]

    if not sample_cases:
        raise HTTPException(status_code=400, detail="No sample test cases available for this problem")

    difficulty = question.get("difficulty", "medium")
    difficulty_cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
    wrapped_code = wrap_code_with_harness(req.code, req.language, problem)

    results = await run_code_against_sample_cases(
        code=wrapped_code,
        language=req.language,
        test_cases=sample_cases,
        timeout_seconds=difficulty_cfg["time_limit_seconds"],
    )

    return {
        "status": "success",
        "mode": "run",
        "question_index": qi,
        **results,
    }


@router.post("/submit-question")
async def submit_question(
    req: SubmitQuestionRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Submit code for a specific question against all (hidden) test cases.
    Records the submission and returns results. Does NOT trigger AI evaluation.
    """
    session = await db["coding_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Assessment session not found")
    if session["status"] == "completed":
        raise HTTPException(status_code=400, detail="This assessment session has already been completed")

    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    qi = req.question_index
    questions = list(session.get("questions", []))

    if qi < 0 or qi >= len(questions):
        raise HTTPException(status_code=400, detail=f"Invalid question index: {qi}")

    question = questions[qi]
    if question.get("status") == "locked":
        raise HTTPException(status_code=400, detail="This question is locked. Complete the previous question first.")

    problem = question.get("problem", {})
    test_cases = problem.get("test_cases", [])

    if not test_cases:
        raise HTTPException(status_code=500, detail="No test cases found for this problem")

    difficulty = question.get("difficulty", "medium")
    difficulty_cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
    wrapped_code = wrap_code_with_harness(req.code, req.language, problem)

    # ── Evaluate against all test cases ──
    report = await evaluate_code(
        code=wrapped_code,
        language=req.language,
        test_cases=test_cases,
        timeout_seconds=difficulty_cfg["time_limit_seconds"],
        score_scale=10,
        hidden_only=False,
    )

    # ── Record submission for this question ──
    submission_record = {
        "submission_id": str(uuid.uuid4()),
        "code": req.code,
        "language": req.language,
        "passed": report.passed,
        "total": report.total,
        "failed": report.total - report.passed,
        "score": report.score,
        "results": [r.model_dump() for r in report.results],
        "compile_error": report.compile_error,
        "runtime_error": report.runtime_error,
        "execution_time_ms": report.execution_time_ms,
        "memory_usage_kb": report.memory_usage_kb,
        "judge0_status": report.judge0_status,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    now = datetime.utcnow().isoformat()
    q_submissions = list(question.get("submissions", []))
    q_submissions.append(submission_record)

    # ── Track best result for this question ──
    best = question.get("best_result")
    if not best or report.passed > best.get("passed", 0):
        best = {
            "passed": report.passed,
            "total": report.total,
            "failed": report.total - report.passed,
            "score": report.score,
            "submission_id": submission_record["submission_id"],
        }

    # ── Check if question is solved (at least one test case passed) ──
    is_solved = report.passed >= 1

    # Update question state
    question["submissions"] = q_submissions
    question["best_result"] = best

    if is_solved:
        question["status"] = "completed"
        question["completed_at"] = now

        # Unlock next question if exists
        next_idx = qi + 1
        if next_idx < len(questions):
            questions[next_idx]["status"] = "active"

    questions[qi] = question

    # Determine if all questions are completed
    all_completed = all(q["status"] == "completed" for q in questions)

    await db["coding_sessions"].update_one(
        {"session_id": req.session_id},
        {"$set": {
            "questions": questions,
            "current_question_index": qi if not is_solved else min(qi + 1, len(questions) - 1),
            "updated_at": now,
        }},
    )

    return {
        "status": "success",
        "submission_id": submission_record["submission_id"],
        "question_index": qi,
        "passed": report.passed,
        "failed": report.total - report.passed,
        "total": report.total,
        "score": report.score,
        "percentage": report.percentage,
        "results": [r.model_dump() for r in report.results],
        "compile_error": report.compile_error,
        "runtime_error": report.runtime_error,
        "execution_time_ms": report.execution_time_ms,
        "memory_usage_kb": report.memory_usage_kb,
        "judge0_status": report.judge0_status,
        "is_optimal": report.passed == report.total,
        "is_solved": is_solved,
        "question_status": question["status"],
        "total_submissions": len(q_submissions),
        "next_question_index": qi + 1 if is_solved and qi + 1 < len(questions) else None,
        "all_completed": all_completed,
    }


@router.get("/question-status")
async def get_question_status(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Get current question state and progress for a full assessment session."""
    session = await db["coding_sessions"].find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.get("questions", [])
    current_idx = session.get("current_question_index", 0)

    return {
        "status": "success",
        "session_id": session_id,
        "current_question_index": current_idx,
        "total_questions": len(questions),
        "assessment_status": session.get("status", "active"),
        "questions": [
            {
                "index": q["index"],
                "difficulty": q["difficulty"],
                "timer_minutes": q["timer_minutes"],
                "status": q["status"],
                "title": q["problem"].get("title", ""),
                "submissions_count": len(q.get("submissions", [])),
                "best_result": q.get("best_result"),
            }
            for q in questions
        ],
    }


# ─── Legacy Endpoints (Backward Compatible) ───────────────────────────────

@router.post("/start")
async def start_assessment(
    req: StartAssessmentRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Generate a new coding problem tailored to the candidate's profile and start a session.
    (Legacy single-question mode)
    """
    if req.difficulty not in DIFFICULTY_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty. Choose from: {list(DIFFICULTY_CONFIG.keys())}",
        )

    if req.language not in get_supported_languages():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Choose from: {get_supported_languages()}",
        )

    user_id = str(current_user["_id"]) if current_user else "anonymous"
    ctx = await _load_candidate_context(user_id)

    # ── Generate problem via Cerebras ──
    try:
        problem = await generate_coding_problem(
            candidate_skills=ctx["candidate_skills"],
            applied_role=ctx["applied_role"],
            difficulty=req.difficulty,
            category=req.category,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate problem: {str(e)}")

    # ── Create session ──
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "candidate_name": ctx["candidate_name"],
        "applied_role": ctx["applied_role"],
        "difficulty": req.difficulty,
        "language": req.language,
        "mode": "single_question",
        "problem": problem,
        "submissions": [],
        "best_result": None,
        "status": "active",
        "start_time": now,
        "created_at": now,
        "updated_at": now,
    }

    await db["coding_sessions"].insert_one(session_doc)

    return {
        "status": "success",
        "session_id": session_id,
        "difficulty": req.difficulty,
        "language": req.language,
        "problem": _format_problem(problem),
    }


@router.post("/submit")
async def submit_code(
    req: SubmitCodeRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Submit code for evaluation. Runs against all test cases and returns results.
    (Legacy mode - supports both custom_input and test case evaluation)
    """
    session = await db["coding_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Assessment session not found")
    if session["status"] == "completed":
        raise HTTPException(status_code=400, detail="This assessment session has already been completed")

    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Submitted code cannot be empty")

    # ── Custom input mode: run once and return raw stdout ──
    if req.custom_input is not None:
        difficulty = session.get("difficulty", "medium")
        difficulty_cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
        problem = session.get("problem", {})
        wrapped_code = wrap_code_with_harness(req.code, req.language, problem)
        run_result = await run_code_once(
            code=wrapped_code,
            language=req.language,
            stdin_data=req.custom_input,
            timeout_seconds=difficulty_cfg["time_limit_seconds"],
        )
        return {
            "status": "success",
            "mode": "run",
            **run_result,
        }

    problem = session.get("problem", {})
    test_cases = problem.get("test_cases", [])

    if not test_cases:
        raise HTTPException(status_code=500, detail="No test cases found for this problem")

    # ── Evaluate code against hidden test cases ──
    difficulty = session.get("difficulty", "medium")
    difficulty_cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
    wrapped_code = wrap_code_with_harness(req.code, req.language, problem)

    report = await evaluate_code(
        code=wrapped_code,
        language=req.language,
        test_cases=test_cases,
        timeout_seconds=difficulty_cfg["time_limit_seconds"],
        score_scale=10,
        hidden_only=False,
    )

    # ── Record submission ──
    submission_record = {
        "submission_id": str(uuid.uuid4()),
        "code": req.code,
        "language": req.language,
        "passed": report.passed,
        "total": report.total,
        "failed": report.total - report.passed,
        "score": report.score,
        "results": [r.model_dump() for r in report.results],
        "compile_error": report.compile_error,
        "runtime_error": report.runtime_error,
        "execution_time_ms": report.execution_time_ms,
        "memory_usage_kb": report.memory_usage_kb,
        "judge0_status": report.judge0_status,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    now = datetime.utcnow().isoformat()
    submissions = list(session.get("submissions", []))
    submissions.append(submission_record)

    # ── Track best result ──
    best = session.get("best_result")
    if not best or report.passed > best.get("passed", 0):
        best = {
            "passed": report.passed,
            "total": report.total,
            "failed": report.total - report.passed,
            "score": report.score,
            "submission_id": submission_record["submission_id"],
        }

    await db["coding_sessions"].update_one(
        {"session_id": req.session_id},
        {"$set": {
            "submissions": submissions,
            "best_result": best,
            "updated_at": now,
        }},
    )

    return {
        "status": "success",
        "submission_id": submission_record["submission_id"],
        "passed": report.passed,
        "failed": report.total - report.passed,
        "total": report.total,
        "score": report.score,
        "percentage": report.percentage,
        "results": [r.model_dump() for r in report.results],
        "compile_error": report.compile_error,
        "runtime_error": report.runtime_error,
        "execution_time_ms": report.execution_time_ms,
        "memory_usage_kb": report.memory_usage_kb,
        "judge0_status": report.judge0_status,
        "is_optimal": report.passed == report.total,
        "total_submissions": len(submissions),
    }


@router.post("/end")
async def end_assessment(
    req: EndAssessmentRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    End the coding assessment session, generate AI evaluation of code quality and approach.
    Supports both single-question and full-assessment modes.
    """
    session = await db["coding_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    mode = session.get("mode", "single_question")

    if mode == "full_assessment":
        return await _end_full_assessment(session, req.session_id)
    else:
        return await _end_single_question(session, req.session_id)


async def _send_coding_completed_email(session: dict, score: float):
    try:
        from bson import ObjectId
        from app.services.email_service import notify_stage_completed
        import asyncio
        user_id = session.get("user_id")
        if user_id and user_id != "anonymous":
            candidate_user = await db["users"].find_one({"_id": ObjectId(user_id)})
            if candidate_user:
                candidate_email = candidate_user.get("email")
                candidate_name = candidate_user.get("full_name") or "Candidate"
                if candidate_email:
                    completed_count = await db["coding_evaluations"].count_documents({"user_id": user_id})
                    attempt_num = completed_count
                    asyncio.create_task(notify_stage_completed(
                        to_email=candidate_email,
                        candidate_name=candidate_name,
                        stage_name="Coding",
                        attempt_num=attempt_num,
                        score=score
                    ))
    except Exception as email_err:
        print(f"Error sending coding completion email: {email_err}")


async def _end_single_question(session: dict, session_id: str) -> dict:
    """End a single-question assessment and generate AI evaluation."""
    submissions = session.get("submissions", [])
    best_result = session.get("best_result")

    if not submissions:
        raise HTTPException(status_code=400, detail="No submissions recorded. Submit code before ending the assessment.")

    # ── Use the best submission for evaluation ──
    best_sub = submissions[-1]
    if best_result:
        for sub in submissions:
            if sub.get("submission_id") == best_result.get("submission_id"):
                best_sub = sub
                break

    execution_results = {
        "passed": best_sub.get("passed", 0),
        "total": best_sub.get("total", 0),
        "results": best_sub.get("results", []),
        "compile_error": best_sub.get("compile_error"),
        "runtime_error": best_sub.get("runtime_error"),
    }

    problem = session.get("problem", {})

    try:
        evaluation = await generate_coding_evaluation(
            problem=problem,
            code=best_sub.get("code", ""),
            language=best_sub.get("language", session.get("language", "python")),
            execution_results=execution_results,
            candidate_name=session.get("candidate_name", "Candidate"),
            applied_role=session.get("applied_role", "Software Engineer"),
        )
    except Exception:
        evaluation = _empty_evaluation()

    now = datetime.utcnow().isoformat()
    solving_time_seconds = _compute_solving_time(session)

    eval_doc = {
        "session_id": session_id,
        "user_id": session["user_id"],
        "candidate_name": session.get("candidate_name", "Candidate"),
        "applied_role": session.get("applied_role", "Software Engineer"),
        "difficulty": session.get("difficulty", "medium"),
        "language": best_sub.get("language", session.get("language", "python")),
        "mode": "single_question",
        "problem": {
            "title": problem.get("title", ""),
            "category": problem.get("category", ""),
            "description": problem.get("description", ""),
        },
        "code": best_sub.get("code", ""),
        "submissions_count": len(submissions),
        "best_result": best_result,
        "solving_time_seconds": solving_time_seconds,
        "evaluation": evaluation,
        "created_at": now,
    }
    await db["coding_evaluations"].insert_one(eval_doc)

    await db["coding_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {"status": "completed", "updated_at": now}},
    )

    await _send_coding_completed_email(session, evaluation.get("overall_score"))

    return {
        "status": "success",
        "evaluation": evaluation,
        "best_result": best_result,
        "solving_time_seconds": solving_time_seconds,
        "submissions_count": len(submissions),
    }


async def _end_full_assessment(session: dict, session_id: str) -> dict:
    """End a full 4-question assessment and generate comprehensive AI evaluation."""
    questions = session.get("questions", [])

    # ── Collect per-question results ──
    question_results = []
    total_passed = 0
    total_cases = 0
    total_submissions = 0

    for q in questions:
        best = q.get("best_result")
        subs = q.get("submissions", [])
        total_submissions += len(subs)

        q_passed = best.get("passed", 0) if best else 0
        q_total = best.get("total", 0) if best else 0
        total_passed += q_passed
        total_cases += q_total

        # Get the best submission code
        best_code = ""
        best_language = session.get("language", "python")
        for sub in subs:
            if best and sub.get("submission_id") == best.get("submission_id"):
                best_code = sub.get("code", "")
                best_language = sub.get("language", session.get("language", "python"))
                break
        if not best_code and subs:
            best_code = subs[-1].get("code", "")

        question_results.append({
            "difficulty": q["difficulty"],
            "title": q["problem"].get("title", ""),
            "category": q["problem"].get("category", ""),
            "description": q["problem"].get("description", ""),
            "passed": q_passed,
            "total": q_total,
            "status": q["status"],
            "submissions_count": len(subs),
            "code": best_code,
            "language": best_language,
            "execution_results": {
                "passed": q_passed,
                "total": q_total,
                "results": subs[-1].get("results", []) if subs else [],
                "compile_error": subs[-1].get("compile_error") if subs else None,
                "runtime_error": subs[-1].get("runtime_error") if subs else None,
            } if subs else None,
        })

    # ── Generate multi-question AI evaluation ──
    try:
        evaluation = await generate_multi_question_evaluation(
            question_results=question_results,
            candidate_name=session.get("candidate_name", "Candidate"),
            applied_role=session.get("applied_role", "Software Engineer"),
            language=session.get("language", "python"),
        )
    except Exception:
        evaluation = _empty_evaluation()

    now = datetime.utcnow().isoformat()
    solving_time_seconds = _compute_solving_time(session)

    eval_doc = {
        "session_id": session_id,
        "user_id": session["user_id"],
        "candidate_name": session.get("candidate_name", "Candidate"),
        "applied_role": session.get("applied_role", "Software Engineer"),
        "language": session.get("language", "python"),
        "mode": "full_assessment",
        "question_results": [
            {
                "difficulty": qr["difficulty"],
                "title": qr["title"],
                "description": qr.get("description", ""),
                "passed": qr["passed"],
                "total": qr["total"],
                "status": qr["status"],
                "submissions_count": qr["submissions_count"],
                "code": qr.get("code", ""),
                "language": qr.get("language", ""),
            }
            for qr in question_results
        ],
        "total_passed": total_passed,
        "total_cases": total_cases,
        "total_submissions": total_submissions,
        "solving_time_seconds": solving_time_seconds,
        "evaluation": evaluation,
        "created_at": now,
    }
    await db["coding_evaluations"].insert_one(eval_doc)

    await db["coding_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {"status": "completed", "updated_at": now}},
    )

    await _send_coding_completed_email(session, evaluation.get("overall_score"))

    return {
        "status": "success",
        "evaluation": evaluation,
        "question_results": [
            {
                "difficulty": qr["difficulty"],
                "title": qr["title"],
                "description": qr.get("description", ""),
                "passed": qr["passed"],
                "total": qr["total"],
                "status": qr["status"],
                "code": qr.get("code", ""),
                "language": qr.get("language", ""),
            }
            for qr in question_results
        ],
        "total_passed": total_passed,
        "total_cases": total_cases,
        "solving_time_seconds": solving_time_seconds,
        "total_submissions": total_submissions,
    }


def _compute_solving_time(session: dict) -> Optional[float]:
    """Compute solving time from session start_time."""
    start_time_str = session.get("start_time")
    if start_time_str:
        try:
            start_dt = datetime.fromisoformat(start_time_str)
            end_dt = datetime.utcnow()
            return round((end_dt - start_dt).total_seconds(), 1)
        except (ValueError, TypeError):
            return None
    return None


def _empty_evaluation() -> dict:
    """Return a fallback empty evaluation."""
    _empty = {"score": 0, "assessment": "Evaluation generation failed."}
    return {
        "overall_score": 0,
        "recommendation": "Hold",
        "overall_summary": "Evaluation generation failed.",
        "correctness": {**_empty},
        "complexity": {"time_complexity": "N/A", "space_complexity": "N/A", "assessment": "N/A"},
        "code_quality": {**_empty},
        "naming_conventions": {**_empty},
        "edge_case_handling": {**_empty},
        "optimization_suggestions": [],
        "strengths": [],
        "weaknesses": [],
        "hiring_notes": "Auto-generated evaluation failed.",
        "skills_demonstrated": [],
        "recommended_topics": [],
        "recruiter_insight": "Evaluation could not be generated.",
    }


# ─── Shared Endpoints ─────────────────────────────────────────────────────

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Retrieve coding assessment session history."""
    session = await db["coding_sessions"].find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["id"] = str(session["_id"])
    del session["_id"]

    # ── Truncate stored code in submissions for response ──
    for sub in session.get("submissions", []):
        if len(sub.get("code", "")) > 2000:
            sub["code"] = sub["code"][:2000] + "\n... [truncated]"

    for q in session.get("questions", []):
        for sub in q.get("submissions", []):
            if len(sub.get("code", "")) > 2000:
                sub["code"] = sub["code"][:2000] + "\n... [truncated]"

    return {"status": "success", "data": session}


@router.get("/evaluation/{session_id}")
async def get_evaluation(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Retrieve the final evaluation report for a completed coding assessment."""
    eval_doc = await db["coding_evaluations"].find_one({"session_id": session_id})
    if not eval_doc:
        raise HTTPException(status_code=404, detail="Evaluation not found for this session")

    eval_doc["id"] = str(eval_doc["_id"])
    del eval_doc["_id"]
    return {"status": "success", "data": eval_doc}
