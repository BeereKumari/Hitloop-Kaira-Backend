"""
Video Interview API — LiveKit Cloud token generation + Beyond Presence avatar + evaluation storage.

Endpoints:
  POST /api/interview/video/token    — generate LiveKit room token, dispatch video agent
  POST /api/interview/video/end      — store transcript + evaluation after session
  GET  /api/interview/video/attempts — list previous video interview attempts
"""

import uuid
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from livekit.api import AccessToken, VideoGrants, LiveKitAPI

from app.database.database import db
from app.middleware.auth import get_current_user_optional
from app.services.vector_service import store_interview_vectors, retrieve_relevant_chunks
from app.services.cerebras_service import generate_interview_evaluation

router = APIRouter(prefix="/api/interview/video", tags=["Video Interview"])

LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_URL        = os.getenv("LIVEKIT_URL", "")

# ─── Request Models ───────────────────────────────────────────────────────────

class VideoTokenRequest(BaseModel):
    complexity:    str = "medium"
    num_questions: int = 8
    max_attempts:  int = 3


class VideoEndSessionRequest(BaseModel):
    session_id:      str
    transcript:      list             # [{speaker, text, timestamp}]
    question_scores: list = []        # [0.0, 0.5, 1.0, ...]
    video_url:       Optional[str] = None
    audio_url:       Optional[str] = None


# ─── Token Endpoint ───────────────────────────────────────────────────────────

@router.post("/token")
async def get_video_token(
    req: VideoTokenRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Generate a LiveKit room token for the video interview candidate.
    Dispatches the kaira-video agent (with Beyond Presence avatar) to the room.
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    user_id = str(current_user["_id"]) if current_user else "anonymous"

    # Enforce recruiter-scheduled configurations and step progression
    if user_id != "anonymous":
        schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
        if not schedule or not schedule.get("video"):
            raise HTTPException(status_code=403, detail="Video interview has not been scheduled by your recruiter yet.")
        
        audio_config = schedule.get("audio")
        if not audio_config or audio_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Prerequisite not met: You must complete and pass the Audio interview before starting the Video interview."
            )

        video_config = schedule["video"]
        scheduled_time = video_config.get("scheduled_time")
        if scheduled_time:
            now_str = datetime.utcnow().isoformat()
            if now_str < scheduled_time:
                raise HTTPException(status_code=403, detail=f"Your Video interview is scheduled for {scheduled_time}. It has not started yet.")
        deadline = video_config.get("deadline")
        if deadline:
            now_str = datetime.utcnow().isoformat()
            if now_str > deadline:
                raise HTTPException(status_code=403, detail=f"The deadline for this Video interview has passed ({deadline}). You can no longer start it.")
        req.complexity = video_config.get("complexity", "medium")
        req.num_questions = int(video_config.get("num_questions", 8))
        req.max_attempts = int(video_config.get("max_attempts", 3))

    if req.complexity not in ("easy", "medium", "hard", "expert"):
        raise HTTPException(status_code=400, detail="Invalid complexity level")
    if not (3 <= req.num_questions <= 20):
        raise HTTPException(status_code=400, detail="num_questions must be 3–20")

    # ── Enforce attempt limit ──
    completed = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "video",
    })
    if completed >= req.max_attempts:
        raise HTTPException(
            status_code=403,
            detail=f"Video interview attempt limit reached ({completed}/{req.max_attempts}). Contact your recruiter.",
        )

    attempt_number = completed + 1
    session_id     = str(uuid.uuid4())
    room_name      = f"kaira-video-{session_id[:12]}"

    # ── Load candidate context ──
    profile      = await db["candidate_profiles"].find_one({"user_id": user_id})
    analysis_doc = await db["resume_analyses"].find_one(
        {"user_id": {"$in": [user_id, "anonymous"]}},
        sort=[("created_at", -1)],
    )
    analysis = analysis_doc.get("analysis", {}) if analysis_doc else {}
    personal   = (profile or {}).get("personal", {})
    experience = (profile or {}).get("experience", {})
    skills_obj = (profile or {}).get("skills", {})

    # Fetch previous chat interview evaluation
    prev_chat_doc = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "chat"},
        sort=[("created_at", -1)],
    )
    prev_chat = prev_chat_doc.get("evaluation", {}) if prev_chat_doc else {}

    # Fetch previous audio interview evaluation
    prev_audio_doc = await db["interview_evaluations"].find_one(
        {"user_id": user_id, "interview_type": "audio"},
        sort=[("created_at", -1)],
    )
    prev_audio = prev_audio_doc.get("evaluation", {}) if prev_audio_doc else {}

    candidate_data = {
        "name":       personal.get("full_name")      or analysis.get("candidate_name", "Candidate"),
        "role":       personal.get("applied_role")   or analysis.get("target_role", "Software Engineer"),
        "experience": experience.get("years_of_experience") or "",
        "company":    experience.get("current_company") or "",
        "skills":     skills_obj.get("core_skills")  or ", ".join(analysis.get("extracted_skills", [])),
        "strengths":          analysis.get("strengths", []),
        "areas_to_explore":   analysis.get("areas_to_explore", []),
        "project_names": [p.get("project_name", "") for p in analysis.get("projects", [])],
        # Previous interview context to probe weak areas
        "prev_chat_strengths":    prev_chat.get("demonstrated_strengths", []),
        "prev_chat_improvements": prev_chat.get("areas_for_improvement", []),
        "prev_audio_strengths":    prev_audio.get("demonstrated_strengths", []),
        "prev_audio_improvements": prev_audio.get("areas_for_improvement", []),
    }

    # RAG: top resume context chunks
    rag_query = f"technical skills {candidate_data['role']} {candidate_data['skills']} projects"
    context_chunks = await retrieve_relevant_chunks(user_id, rag_query, top_k=5)
    rag_context = "\n---\n".join([c.get("content", "")[:400] for c in context_chunks])

    now = datetime.utcnow().isoformat()

    # ── Create pending session record ──
    session_doc = {
        "session_id":      session_id,
        "room_name":       room_name,
        "user_id":         user_id,
        "interview_type":  "video",
        "attempt_number":  attempt_number,
        "max_attempts":    req.max_attempts,
        "complexity":      req.complexity,
        "num_questions":   req.num_questions,
        "candidate_data":  candidate_data,
        "rag_context":     rag_context,
        "status":          "pending",
        "created_at":      now,
        "updated_at":      now,
    }
    await db["interview_sessions"].insert_one(session_doc)

    # ── Generate LiveKit token ──
    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(f"candidate-{user_id[:8]}")
        .with_name(candidate_data["name"])
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )

    # ── Dispatch the Kaira video agent to the room ──
    from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
    lk_http_url = LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")
    async with LiveKitAPI(url=lk_http_url, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET) as lk:
        await lk.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(agent_name="kaira-video", room=room_name)
        )

    return {
        "status":             "success",
        "token":              token,
        "room_name":          room_name,
        "session_id":         session_id,
        "livekit_url":        LIVEKIT_URL,
        "candidate_name":     candidate_data["name"],
        "applied_role":       candidate_data["role"],
        "attempt_number":     attempt_number,
        "attempts_remaining": req.max_attempts - attempt_number,
        "complexity":         req.complexity,
        "num_questions":      req.num_questions,
    }


# ─── End Session Endpoint ─────────────────────────────────────────────────────

@router.post("/end")
async def end_video_session(
    req: VideoEndSessionRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Called when video interview ends.
    Receives transcript + scores, generates evaluation, stores in MongoDB.
    """
    user_id = str(current_user["_id"]) if current_user else "anonymous"

    session = await db["interview_sessions"].find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    question_scores = req.question_scores or []
    num_questions   = session.get("num_questions", 8)

    # ── Compute score ──
    if question_scores:
        earned      = sum(question_scores)
        final_score = round((earned / num_questions) * 10, 1)
    else:
        final_score = 0.0

    # ── Build transcript string ──
    transcript_str = "\n\n".join([
        f"{t.get('speaker', 'Unknown')}: {t.get('text', '')}"
        for t in req.transcript
    ])

    candidate_data = session.get("candidate_data", {})

    # ── Generate qualitative evaluation via Cerebras ──
    evaluation = await _generate_video_evaluation(
        transcript=transcript_str,
        candidate_name=candidate_data.get("name", "Candidate"),
        applied_role=candidate_data.get("role", "Software Engineer"),
        num_questions=num_questions,
        computed_score=final_score,
        question_scores=question_scores,
    )

    evaluation["final_score"]         = final_score
    evaluation["question_scores"]     = question_scores
    evaluation["questions_attempted"] = len(question_scores)
    evaluation["questions_total"]     = num_questions
    evaluation["score_breakdown"] = {
        "earned_points":  sum(question_scores),
        "total_possible": num_questions,
        "percentage":     round((sum(question_scores) / num_questions) * 100) if num_questions else 0,
    }

    now = datetime.utcnow().isoformat()

    # ── Store evaluation ──
    eval_doc = {
        "session_id":       req.session_id,
        "user_id":          user_id,
        "interview_type":   "video",
        "attempt_number":   session.get("attempt_number", 1),
        "candidate_data":   candidate_data,
        "complexity":       session.get("complexity", "medium"),
        "num_questions":    num_questions,
        "question_scores":  question_scores,
        "evaluation":       evaluation,
        "transcript":       transcript_str,
        "transcript_turns": req.transcript,
        "video_url":        req.video_url,
        "audio_url":        req.audio_url,
        "created_at":       now,
    }
    await db["interview_evaluations"].insert_one(eval_doc)

    # ── Vectorize evaluation ──
    await store_interview_vectors(user_id, req.session_id, evaluation)

    # ── Mark session completed ──
    await db["interview_sessions"].update_one(
        {"session_id": req.session_id},
        {"$set": {
            "status":    "completed",
            "video_url": req.video_url,
            "audio_url": req.audio_url,
            "updated_at": now,
        }},
    )

    try:
        from app.services.email_service import notify_stage_completed
        import asyncio
        if current_user and current_user.get("email"):
            asyncio.create_task(notify_stage_completed(
                to_email=current_user["email"],
                candidate_name=current_user.get("full_name") or "Candidate",
                stage_name="Video Interview",
                attempt_num=session.get("attempt_number", 1),
                score=final_score
            ))
    except Exception as email_err:
        print(f"Error sending video completion email: {email_err}")

    return {
        "status":      "success",
        "evaluation":  evaluation,
        "final_score": final_score,
        "video_url":   req.video_url,
        "audio_url":   req.audio_url,
    }


# ─── Attempts Endpoint ────────────────────────────────────────────────────────

@router.get("/attempts")
async def list_video_attempts(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """List all previous video interview attempts for the current user."""
    user_id = str(current_user["_id"]) if current_user else "anonymous"
    docs = await db["interview_evaluations"].find(
        {"user_id": user_id, "interview_type": "video"},
        sort=[("created_at", -1)],
    ).limit(10).to_list(10)

    attempts = []
    for doc in docs:
        ev = doc.get("evaluation", {})
        attempts.append({
            "session_id":      doc.get("session_id", ""),
            "attempt_number":  doc.get("attempt_number", 1),
            "complexity":      doc.get("complexity", "medium"),
            "num_questions":   doc.get("num_questions", 0),
            "question_scores": doc.get("question_scores", []),
            "final_score":     ev.get("final_score", 0),
            "recommendation":  ev.get("recommendation", "Hold"),
            "score_breakdown": ev.get("score_breakdown", {}),
            "video_url":       doc.get("video_url", ""),
            "audio_url":       doc.get("audio_url", ""),
            "created_at":      doc.get("created_at", ""),
        })

    return {"status": "success", "attempts": attempts, "total": len(attempts)}


# ─── Video Evaluation Prompt ──────────────────────────────────────────────────

VIDEO_EVAL_PROMPT = """You are a senior technical recruiter evaluating a video interview.
Analyze the transcript and return a valid JSON object with this exact schema:

{
  "final_score": 7.5,
  "recommendation": "Hire",
  "overall_summary": "3-4 sentence evaluation of technical depth, communication quality, structured thinking, and answer quality based solely on the transcript content.",
  "demonstrated_strengths": ["Specific strength with transcript evidence"],
  "areas_for_improvement": ["Specific area for development with transcript evidence"],
  "topic_evaluations": [
    {"topic": "System Design", "score": 8, "feedback": "Detailed assessment based on answers given"}
  ],
  "communication_score": 8,
  "technical_depth_score": 7,
  "structured_thinking_score": 8,
  "answer_quality_score": 7,
  "problem_solving_score": 8,
  "response_completeness_score": 7,
  "hiring_notes": "Private panel notes: key risks or standout moments from the transcript."
}

IMPORTANT RULES:
- Evaluate ONLY based on answer content from the transcript
- Do NOT make any inferences about appearance, posture, body language, expressions, or visual behaviour
- Do NOT make psychological or personality claims based on appearance
- Focus solely on: technical accuracy, depth, structured thinking, communication clarity
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- All scores are out of 10. Return ONLY valid JSON, no markdown."""


async def _generate_video_evaluation(
    transcript: str,
    candidate_name: str,
    applied_role: str,
    num_questions: int,
    computed_score: float,
    question_scores: list,
) -> dict:
    import asyncio, json
    from app.services.cerebras_service import get_cerebras_client, CEREBRAS_MODEL

    client = get_cerebras_client()
    scores_str = ", ".join([str(s) for s in question_scores]) or "none recorded"

    user_content = (
        f"Candidate: {candidate_name}\n"
        f"Applied Role: {applied_role}\n"
        f"Questions Answered: {num_questions}\n"
        f"Computed Score (use as final_score): {computed_score}/10\n"
        f"Per-Question Scores [{scores_str}]: 1.0=thorough, 0.5=partial, 0.0=poor\n\n"
        f"Video Interview Transcript:\n{transcript[:12000]}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": VIDEO_EVAL_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw  = await loop.run_in_executor(None, _call)
    raw  = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "final_score":                computed_score,
            "recommendation":             "Hold",
            "overall_summary":            raw[:500] or "Evaluation parse error.",
            "demonstrated_strengths":     [],
            "areas_for_improvement":      [],
            "topic_evaluations":          [],
            "communication_score":        0,
            "technical_depth_score":      0,
            "structured_thinking_score":  0,
            "answer_quality_score":       0,
            "problem_solving_score":      0,
            "response_completeness_score": 0,
            "hiring_notes": "Auto-evaluation failed — review transcript.",
        }
