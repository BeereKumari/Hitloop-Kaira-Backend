"""
Interview API — AI-powered chat interview backed by RAG + Cerebras gpt-oss-120b.

Endpoints:
  POST /api/interview/start            — Begin a new interview session
  POST /api/interview/message          — Send a candidate message, get AI reply
  POST /api/interview/end              — End session, generate & store evaluation
  GET  /api/interview/attempts         — List previous attempts for current user
  GET  /api/interview/session/{id}     — Retrieve session history
  GET  /api/interview/evaluation/{id}  — Retrieve final evaluation
"""

import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.database.database import db
from app.middleware.auth import get_current_user_optional
from app.services.vector_service import retrieve_relevant_chunks, store_interview_vectors
from app.services.cerebras_service import run_interview_turn, generate_interview_evaluation

router = APIRouter(prefix="/api/interview", tags=["Interview"])

MAX_FOLLOWUPS = 2       # Hard cap: max 2 follow-ups per question

# ─── Complexity Level Config ─────────────────────────────────────────────────

COMPLEXITY_CONFIG = {
    "easy": {
        "level": "L1-L2",
        "label": "Easy",
        "description": "Foundational concepts and basic application of skills",
        "instruction": "Ask foundational questions about core concepts. 'What is' and 'how does' style. Keep straightforward and concept-focused.",
    },
    "medium": {
        "level": "L3-L4",
        "label": "Medium",
        "description": "Applied knowledge, design patterns, and trade-offs",
        "instruction": "Ask application-level questions about design patterns, trade-offs, and real-world scenarios. Probe practical experience and decision-making.",
    },
    "hard": {
        "level": "L5-L6",
        "label": "Hard",
        "description": "Complex system design, optimization, and failure modes",
        "instruction": "Ask complex system design questions involving scale, concurrency, failure modes, and optimization. Require reasoning about trade-offs under constraints.",
    },
    "expert": {
        "level": "L7-L8",
        "label": "Expert",
        "description": "Research-level, novel architectures, and deep architectural reasoning",
        "instruction": "Ask expert-level questions about novel architectures, research insights, and deep edge cases. Explore the boundaries of the candidate's knowledge.",
    },
}


# ─── System Prompt Builder ───────────────────────────────────────────────────

def build_interview_system_prompt(
    candidate_data: dict,
    context_chunks: list,
    complexity: str,
    num_questions: int,
    max_followups: int = MAX_FOLLOWUPS,
) -> str:
    cfg = COMPLEXITY_CONFIG.get(complexity, COMPLEXITY_CONFIG["medium"])
    chunks_text = "\n---\n".join([c.get("content", "") for c in context_chunks[:6]])

    strengths = candidate_data.get("strengths", [])
    areas = candidate_data.get("areas_to_explore", [])
    projects = candidate_data.get("project_names", [])

    return f"""You are Kaira, a strict senior technical interviewer at a top AI company. You are conducting a {cfg["label"]}-level ({cfg["level"]}) structured technical interview.

CANDIDATE PROFILE:
- Name: {candidate_data.get("name", "Candidate")}
- Applied Role: {candidate_data.get("role", "Software Engineer")}
- Experience: {candidate_data.get("experience", "Not specified")} years at {candidate_data.get("company", "N/A")}
- Core Skills: {candidate_data.get("skills", "Not specified")}
- Known Projects: {", ".join(projects) if projects else "See resume context below"}

RESUME & PROJECT CONTEXT (from candidate's actual documents):
{chunks_text[:4500] if chunks_text else "No resume context available."}

AI ANALYSIS INSIGHTS:
- Demonstrated Strengths: {", ".join(strengths[:4]) if strengths else "Assess from resume context"}
- Areas to Probe Deeply: {", ".join(areas[:4]) if areas else "General technical depth"}

INTERVIEW PARAMETERS:
- Total Questions: {num_questions}
- Max Follow-ups per Question: {max_followups} (not mandatory — only when genuinely needed)
- Complexity Level: {cfg["level"]} — {cfg["description"]}
- Instruction: {cfg["instruction"]}

STRICT INTERVIEWER RULES (DO NOT VIOLATE):
1. Ask ONLY ONE focused technical question at a time, based SOLELY on this candidate's resume, projects, and skills
2. NEVER explain concepts, hint, or help the candidate — you are an EVALUATOR, not a tutor
3. If asked to explain/hint/help: reply ONLY "I'm here to evaluate your expertise, not guide you. Please answer from your own experience."
4. If off-topic/casual message: reply ONLY "Let's stay focused on the interview. [repeat current question]"
5. After each answer:
   - Strong, detailed answer → ask follow-up only if it reveals a genuinely interesting gap to probe
   - Vague/incomplete answer → ask one follow-up to clarify, or move on
   - Wrong/no answer → note it, move to next topic
6. Follow-ups are NOT mandatory — prefer moving to the next question over forced follow-ups
7. Always acknowledge the answer briefly (1 sentence max) before the next question
8. Ground every question in the candidate's actual work and specific tech stack

SCORING RULES — Apply these when transitioning away from a question (is_followup=false):
- question_score 1.0 = Complete, accurate, well-explained answer with depth
- question_score 0.5 = Partially correct, or correct direction but missing key details
- question_score 0.0 = Incorrect, vague, refused to answer, or completely off-topic

RESPONSE FORMAT — ALWAYS return valid JSON only, no extra text:
{{
  "reply": "Your complete interviewer response: brief acknowledgment + next question",
  "is_followup": true_or_false,
  "depth_level": 3,
  "signal": "One brief observation about answer quality, or empty string",
  "question_score": 0.5
}}

IMPORTANT for question_score:
- When is_followup=false (moving to next question): include the score (0.0, 0.5, or 1.0) for the question just answered
- When is_followup=true: set question_score to null (still being evaluated)
- On the very first question (no answer yet): set question_score to null

depth_level scale: 1=basic, 2=foundational, 3=applied, 4=design, 5=advanced, 6=expert, 7=research
"""


# ─── Request / Response Models ────────────────────────────────────────────────

class StartInterviewRequest(BaseModel):
    complexity: str = "medium"
    num_questions: int = 8
    max_attempts: int = 3    # Total allowed attempts for this user (enforced by backend)


class MessageRequest(BaseModel):
    session_id: str
    message: str
    tab_switches_count: Optional[int] = None


class EndInterviewRequest(BaseModel):
    session_id: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _calc_score_from_questions(question_scores: List[float], num_questions: int) -> float:
    """
    Calculate final score as fraction of total possible points.
    Returns a value 0-10.
    Each question is worth 1 point (1.0 full, 0.5 partial, 0.0 none).
    Score = (total_earned / num_questions) * 10
    """
    if not num_questions:
        return 0.0
    earned = sum(question_scores)
    return round((earned / num_questions) * 10, 1)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/schedule")
async def get_candidate_schedule(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Retrieve the candidate's interview and assessment schedules, attempt counts, and latest scores."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    user_id = str(current_user["_id"])

    schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
    if not schedule:
        schedule = {
            "chat": None,
            "audio": None,
            "video": None,
            "coding": None,
            "live_project": None,
            "ai_fluency": None,
            "behaviour": None,
        }
    else:
        schedule["id"] = str(schedule["_id"])
        del schedule["_id"]
        for field in ["chat", "audio", "video", "coding", "live_project", "ai_fluency", "behaviour"]:
            if field not in schedule:
                schedule[field] = None

    chat_completed = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": {"$exists": False}
    })
    chat_completed_specific = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "chat"
    })
    chat_count = chat_completed + chat_completed_specific

    audio_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "audio"
    })

    video_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "video"
    })

    coding_count = await db["coding_evaluations"].count_documents({
        "user_id": user_id
    })

    lp_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "live_project"
    })

    ai_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "ai_fluency"
    })

    beh_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "behaviour"
    })

    chat_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "$or": [{"interview_type": "chat"}, {"interview_type": {"$exists": False}}]},
        sort=[("created_at", -1)]
    )
    audio_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "audio"},
        sort=[("created_at", -1)]
    )
    video_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "video"},
        sort=[("created_at", -1)]
    )
    coding_latest = await db["coding_evaluations"].find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)]
    )
    lp_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "live_project"},
        sort=[("created_at", -1)]
    )
    ai_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "ai_fluency"},
        sort=[("created_at", -1)]
    )
    beh_latest = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "behaviour"},
        sort=[("created_at", -1)]
    )

    coding_score = coding_latest.get("evaluation", {}).get("overall_score") if coding_latest else None
    lp_score = lp_latest.get("final_score") or lp_latest.get("evaluation", {}).get("final_score") if lp_latest else None
    ai_score = ai_latest.get("final_score") or ai_latest.get("evaluation", {}).get("final_score") if ai_latest else None
    beh_score = beh_latest.get("final_score") or beh_latest.get("evaluation", {}).get("final_score") if beh_latest else None

    return {
        "status": "success",
        "schedule": schedule,
        "attempts": {
            "chat": chat_count,
            "audio": audio_count,
            "video": video_count,
            "coding": coding_count,
            "live_project": lp_count,
            "ai_fluency": ai_count,
            "behaviour": beh_count,
        },
        "scores": {
            "chat": chat_latest.get("evaluation", {}).get("final_score") if chat_latest else None,
            "audio": audio_latest.get("final_score") or audio_latest.get("evaluation", {}).get("final_score") if audio_latest else None,
            "video": video_latest.get("final_score") or video_latest.get("evaluation", {}).get("final_score") if video_latest else None,
            "coding": coding_score,
            "live_project": lp_score,
            "ai_fluency": ai_score,
            "behaviour": beh_score,
        }
    }


@router.post("/start")
async def start_interview(
    req: StartInterviewRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Begin a new interview session.
    Checks attempt limits, loads candidate vectors + analysis, generates first question.
    """
    user_id = str(current_user["_id"]) if current_user else "anonymous"

    # Enforce recruiter-scheduled configurations
    if user_id != "anonymous":
        schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
        if not schedule or not schedule.get("chat"):
            raise HTTPException(status_code=403, detail="Chat interview has not been scheduled by your recruiter yet.")
        chat_config = schedule["chat"]
        scheduled_time = chat_config.get("scheduled_time")
        if scheduled_time:
            now_str = datetime.utcnow().isoformat()
            if now_str < scheduled_time:
                raise HTTPException(status_code=403, detail=f"Your Chat interview is scheduled for {scheduled_time}. It has not started yet.")
        deadline = chat_config.get("deadline")
        if deadline:
            now_str = datetime.utcnow().isoformat()
            if now_str > deadline:
                raise HTTPException(status_code=403, detail=f"The deadline for this Chat interview has passed ({deadline}). You can no longer start it.")
        req.complexity = chat_config.get("complexity", "medium")
        req.num_questions = int(chat_config.get("num_questions", 8))
        req.max_attempts = int(chat_config.get("max_attempts", 3))

    if req.complexity not in COMPLEXITY_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid complexity. Choose from: {list(COMPLEXITY_CONFIG.keys())}")
    if not (3 <= req.num_questions <= 20):
        raise HTTPException(status_code=400, detail="num_questions must be between 3 and 20")
    if not (1 <= req.max_attempts <= 10):
        raise HTTPException(status_code=400, detail="max_attempts must be between 1 and 10")

    # ── Enforce attempt limit ──
    completed_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "$or": [{"interview_type": "chat"}, {"interview_type": {"$exists": False}}]
    })
    if completed_count >= req.max_attempts:
        raise HTTPException(
            status_code=403,
            detail=f"Attempt limit reached. You have used {completed_count}/{req.max_attempts} allowed attempts. Contact your recruiter to reset.",
        )

    attempt_number = completed_count + 1

    # ── Load candidate context ──
    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    analysis_doc = await db["resume_analyses"].find_one(
        {"user_id": {"$in": [user_id, "anonymous"]}},
        sort=[("created_at", -1)],
    )
    analysis = analysis_doc.get("analysis", {}) if analysis_doc else {}

    personal = (profile or {}).get("personal", {})
    experience = (profile or {}).get("experience", {})
    skills_obj = (profile or {}).get("skills", {})

    candidate_data = {
        "name": personal.get("full_name") or analysis.get("candidate_name", "Candidate"),
        "role": personal.get("applied_role") or analysis.get("target_role", "Software Engineer"),
        "experience": experience.get("years_of_experience") or "",
        "company": experience.get("current_company") or "",
        "skills": skills_obj.get("core_skills") or ", ".join(analysis.get("extracted_skills", [])),
        "strengths": analysis.get("strengths", []),
        "areas_to_explore": analysis.get("areas_to_explore", []),
        "project_names": [p.get("project_name", "") for p in analysis.get("projects", [])],
    }

    # ── RAG: retrieve most relevant resume context ──
    query = f"technical skills {candidate_data['role']} {candidate_data['skills']} projects"
    context_chunks = await retrieve_relevant_chunks(user_id, query, top_k=6)

    # ── Build system prompt ──
    system_prompt = build_interview_system_prompt(
        candidate_data, context_chunks, req.complexity, req.num_questions, MAX_FOLLOWUPS
    )

    # ── Generate opening question ──
    opening_messages = [{
        "role": "user",
        "content": (
            "Start the interview now. "
            "Introduce yourself as Kaira in exactly one sentence, then immediately ask the first technical question. "
            "The question must be directly based on the candidate's specific projects or skills. "
            "Set question_score to null for the opening. Do not ask generic questions."
        ),
    }]

    ai_response = await run_interview_turn(system_prompt, opening_messages)
    first_question = ai_response.get(
        "reply",
        "Hello! I'm Kaira, your interviewer today. Let's begin — can you walk me through your most technically challenging project and the key architectural decisions you made?"
    )

    # ── Create session ──
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    signals = [s for s in [ai_response.get("signal", "")] if s]

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "attempt_number": attempt_number,
        "max_attempts": req.max_attempts,
        "complexity": req.complexity,
        "num_questions": req.num_questions,
        "max_followups": MAX_FOLLOWUPS,
        "candidate_data": candidate_data,
        "system_prompt": system_prompt,
        "messages": [{"role": "assistant", "content": first_question}],
        "signals_captured": signals,
        "question_count": 1,
        "current_followup_count": 0,   # follow-ups on current question
        "question_scores": [],          # per-question scores (0.0 / 0.5 / 1.0)
        "max_depth": ai_response.get("depth_level", 3),
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    await db["interview_sessions"].insert_one(session_doc)

    return {
        "status": "success",
        "session_id": session_id,
        "question": first_question,
        "candidate_name": candidate_data["name"],
        "applied_role": candidate_data["role"],
        "question_count": 1,
        "num_questions": req.num_questions,
        "complexity": req.complexity,
        "attempt_number": attempt_number,
        "attempts_remaining": req.max_attempts - attempt_number,
        "signals_captured": signals,
        "question_scores": [],
    }


@router.post("/message")
async def send_message(
    req: MessageRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Process a candidate message, run RAG retrieval, call Cerebras, return AI reply.
    Enforces max follow-up limit per question and tracks per-question scores.
    """
    user_id = str(current_user["_id"]) if current_user else "anonymous"

    session = await db["interview_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session["status"] == "completed":
        raise HTTPException(status_code=400, detail="This interview session has already been completed")

    current_followup_count = session.get("current_followup_count", 0)
    max_followups = session.get("max_followups", MAX_FOLLOWUPS)
    question_scores = list(session.get("question_scores", []))

    # ── RAG: retrieve context relevant to this answer ──
    context_chunks = await retrieve_relevant_chunks(user_id, req.message, top_k=3)
    rag_text = "\n".join([c.get("content", "")[:250] for c in context_chunks])

    # ── Build message history ──
    history = list(session["messages"])
    augmented_user_msg = req.message
    if rag_text:
        augmented_user_msg = (
            f"{req.message}\n\n"
            f"[Relevant context from candidate's resume: {rag_text[:700]}]"
        )

    # ── Enforce follow-up cap ──
    # If already at the limit, inject an instruction to score & move on
    force_next = current_followup_count >= max_followups
    if force_next:
        augmented_user_msg += (
            f"\n\n[SYSTEM NOTE: Maximum follow-ups ({max_followups}) reached for this question. "
            f"You MUST score this question now (question_score) and move to the next topic. "
            f"Set is_followup=false.]"
        )

    history.append({"role": "user", "content": augmented_user_msg})

    # ── Call Cerebras ──
    ai_response = await run_interview_turn(session["system_prompt"], history)
    reply = ai_response.get("reply", "Let's continue — please elaborate.")
    signal = ai_response.get("signal", "")
    depth = int(ai_response.get("depth_level", 3))
    is_followup = bool(ai_response.get("is_followup", False))

    # Force is_followup=False if cap was hit
    if force_next:
        is_followup = False

    # ── Per-question scoring ──
    raw_score = ai_response.get("question_score")
    question_score = None
    if not is_followup and raw_score is not None:
        try:
            question_score = float(raw_score)
            question_score = max(0.0, min(1.0, question_score))
            # Round to nearest 0.5
            question_score = round(question_score * 2) / 2
        except (TypeError, ValueError):
            question_score = 0.5  # default to partial if AI returned garbage

    if question_score is not None:
        question_scores.append(question_score)

    # ── Update counters ──
    if is_followup:
        new_followup_count = current_followup_count + 1
        new_question_count = session["question_count"]
    else:
        new_followup_count = 0   # reset for next question
        new_question_count = session["question_count"] + 1

    signals = list(session["signals_captured"])
    if signal:
        signals.append(signal)
    if req.tab_switches_count is not None and req.tab_switches_count > 0:
        tab_sig = f"Candidate switched tabs {req.tab_switches_count} times during interview"
        if tab_sig not in signals:
            signals.append(tab_sig)

    updated_messages = list(session["messages"]) + [
        {"role": "user", "content": req.message},   # store original, not augmented
        {"role": "assistant", "content": reply},
    ]

    now = datetime.utcnow().isoformat()
    await db["interview_sessions"].update_one(
        {"session_id": req.session_id},
        {"$set": {
            "messages": updated_messages,
            "signals_captured": signals,
            "question_count": new_question_count,
            "current_followup_count": new_followup_count,
            "question_scores": question_scores,
            "max_depth": max(depth, session.get("max_depth", 3)),
            "tab_switches_count": req.tab_switches_count,
            "updated_at": now,
        }},
    )

    interview_complete = new_question_count >= session["num_questions"]

    return {
        "status": "success",
        "reply": reply,
        "is_followup": is_followup,
        "depth_level": depth,
        "signal": signal,
        "question_score": question_score,
        "question_scores": question_scores,
        "signals_captured": signals,
        "question_count": new_question_count,
        "num_questions": session["num_questions"],
        "current_followup_count": new_followup_count,
        "interview_complete": interview_complete,
    }


@router.post("/end")
async def end_interview(
    req: EndInterviewRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    End the interview, compute score from per-question scores,
    generate AI qualitative evaluation, store in interview_evaluations + vectorize.
    """
    user_id = str(current_user["_id"]) if current_user else "anonymous"

    session = await db["interview_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    question_scores = session.get("question_scores", [])
    num_questions = session.get("num_questions", 8)
    question_count = session.get("question_count", len(question_scores))

    # ── Compute score from per-question data ──
    computed_score = _calc_score_from_questions(question_scores, num_questions)

    # ── Build transcript ──
    transcript_lines = []
    for msg in session["messages"]:
        label = "Interviewer (Kaira)" if msg["role"] == "assistant" else "Candidate"
        transcript_lines.append(f"{label}: {msg['content']}")
    transcript = "\n\n".join(transcript_lines)

    candidate_data = session.get("candidate_data", {})

    # ── Generate qualitative evaluation via Cerebras ──
    evaluation = await generate_interview_evaluation(
        transcript=transcript,
        candidate_name=candidate_data.get("name", "Candidate"),
        applied_role=candidate_data.get("role", "Software Engineer"),
        num_questions=question_count,
        computed_score=computed_score,
        question_scores=question_scores,
    )

    # Override AI-generated score with our computed score (based on actual per-question scoring)
    evaluation["final_score"] = computed_score
    evaluation["question_scores"] = question_scores
    evaluation["questions_attempted"] = question_count
    evaluation["questions_total"] = num_questions
    evaluation["score_breakdown"] = {
        "earned_points": sum(question_scores),
        "total_possible": num_questions,
        "percentage": round((sum(question_scores) / num_questions) * 100) if num_questions else 0,
    }

    now = datetime.utcnow().isoformat()

    # ── Store evaluation in dedicated collection ──
    eval_doc = {
        "session_id": req.session_id,
        "user_id": user_id,
        "attempt_number": session.get("attempt_number", 1),
        "candidate_data": candidate_data,
        "complexity": session.get("complexity", "medium"),
        "num_questions": num_questions,
        "question_count": question_count,
        "question_scores": question_scores,
        "signals_captured": session.get("signals_captured", []),
        "evaluation": evaluation,
        "transcript": transcript,
        "created_at": now,
    }
    await db["interview_evaluations"].insert_one(eval_doc)

    # ── Vectorize evaluation for future interviews ──
    vector_count = await store_interview_vectors(user_id, req.session_id, evaluation)

    # ── Mark session as completed ──
    await db["interview_sessions"].update_one(
        {"session_id": req.session_id},
        {"$set": {"status": "completed", "updated_at": now}},
    )

    try:
        from app.services.email_service import notify_stage_completed
        import asyncio
        if current_user and current_user.get("email"):
            asyncio.create_task(notify_stage_completed(
                to_email=current_user["email"],
                candidate_name=current_user.get("full_name") or "Candidate",
                stage_name="Chat",
                attempt_num=session.get("attempt_number", 1),
                score=computed_score
            ))
    except Exception as email_err:
        print(f"Error sending chat completion email: {email_err}")

    return {
        "status": "success",
        "evaluation": evaluation,
        "signals_captured": session.get("signals_captured", []),
        "question_count": question_count,
        "question_scores": question_scores,
        "computed_score": computed_score,
        "vector_count": vector_count,
    }


@router.get("/attempts")
async def list_attempts(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    List all previous completed interview attempts for the current user.
    Used to display previous results on the chat interview page.
    """
    user_id = str(current_user["_id"]) if current_user else "anonymous"

    cursor = db["interview_evaluations"].find(
        {"user_id": user_id},
        sort=[("created_at", -1)],
    ).limit(10)

    docs = await cursor.to_list(10)
    attempts = []
    for doc in docs:
        eval_data = doc.get("evaluation", {})
        attempts.append({
            "session_id": doc.get("session_id", ""),
            "attempt_number": doc.get("attempt_number", 1),
            "complexity": doc.get("complexity", "medium"),
            "num_questions": doc.get("num_questions", 0),
            "question_count": doc.get("question_count", 0),
            "question_scores": doc.get("question_scores", []),
            "final_score": eval_data.get("final_score", 0),
            "recommendation": eval_data.get("recommendation", "Hold"),
            "score_breakdown": eval_data.get("score_breakdown", {}),
            "created_at": doc.get("created_at", ""),
        })

    return {"status": "success", "attempts": attempts, "total": len(attempts)}


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Retrieve interview session history (system prompt excluded)."""
    session = await db["interview_sessions"].find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["id"] = str(session["_id"])
    del session["_id"]
    session.pop("system_prompt", None)
    return {"status": "success", "data": session}


@router.get("/evaluation/{session_id}")
async def get_evaluation(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Retrieve the final evaluation report for a completed interview session."""
    eval_doc = await db["interview_evaluations"].find_one({"session_id": session_id})
    if not eval_doc:
        raise HTTPException(status_code=404, detail="Evaluation not found for this session")

    eval_doc["id"] = str(eval_doc["_id"])
    del eval_doc["_id"]
    return {"status": "success", "data": eval_doc}
