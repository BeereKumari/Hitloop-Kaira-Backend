from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

from app.database.database import db
from app.middleware.auth import get_current_user_optional
from app.services.ai_fluency_service import generate_fluency_questions, generate_fluency_evaluation

router = APIRouter(prefix="/api/ai-fluency", tags=["AI Tool Fluency Assessment"])

class SubmitFluencyRequest(BaseModel):
    answers: Dict[str, Any]

async def _load_candidate_context(user_id: str) -> dict:
    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    user = await db["users"].find_one({"_id": db["users"].codec_options.uuid_representation.to_uuid(user_id) if hasattr(db["users"].codec_options.uuid_representation, "to_uuid") else user_id})
    
    if not user:
        try:
            from bson import ObjectId
            user = await db["users"].find_one({"_id": ObjectId(user_id)})
        except Exception:
            pass

    skills = "Python, JavaScript, React, System Design"
    role = "Software Engineer"
    name = "Candidate"
    
    if profile:
        skills = ", ".join(profile.get("skills", [])) or skills
        personal = profile.get("personal", {})
        role = personal.get("target_role", role)
        name = personal.get("full_name", name)
    elif user:
        name = user.get("full_name", name)
        
    return {
        "candidate_skills": skills,
        "applied_role": role,
        "candidate_name": name,
    }

@router.get("/questions")
async def get_fluency_questions(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    
    user_id = str(current_user["_id"])
    
    # 1. Enforce lock check
    schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
    if not schedule or not schedule.get("live_project"):
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency is locked. You must complete the Live Project assessment and be shortlisted first."
        )
    
    lp_conf = schedule["live_project"]
    if not lp_conf.get("analysis_completed") or lp_conf.get("decision") != "shortlist":
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency is locked. You must complete the Live Project assessment and be shortlisted first."
        )
        
    # 2. Check schedule config
    if not schedule.get("ai_fluency"):
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency has not been scheduled by your recruiter yet."
        )
        
    fluency_conf = schedule["ai_fluency"]
    
    # Enforce scheduled start time
    scheduled_time = fluency_conf.get("scheduled_time")
    if scheduled_time:
        now_str = datetime.utcnow().isoformat()
        if now_str < scheduled_time:
            raise HTTPException(
                status_code=403,
                detail=f"Your AI Tool Fluency assessment is scheduled for {scheduled_time}. It has not started yet."
            )
            
    # Enforce deadline
    deadline = fluency_conf.get("deadline")
    if deadline:
        now_str = datetime.utcnow().isoformat()
        if now_str > deadline:
            raise HTTPException(
                status_code=403,
                detail=f"The deadline for this AI Tool Fluency assessment has passed ({deadline})."
            )

    # 3. Load or generate detailed questions
    cached = await db["ai_fluency_questions"].find_one({"user_id": user_id})
    if cached:
        return cached["questions"]

    ctx = await _load_candidate_context(user_id)
    complexity = fluency_conf.get("complexity", "medium")
    num_questions = int(fluency_conf.get("num_questions") or 5)
    
    questions = await generate_fluency_questions(
        candidate_skills=ctx["candidate_skills"],
        applied_role=ctx["applied_role"],
        complexity=complexity,
        num_questions=num_questions
    )
    
    await db["ai_fluency_questions"].insert_one({
        "user_id": user_id,
        "questions": questions,
        "created_at": datetime.utcnow().isoformat()
    })
    
    return questions

@router.post("/submit")
async def submit_fluency(
    req: SubmitFluencyRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
        
    user_id = str(current_user["_id"])
    ctx = await _load_candidate_context(user_id)
    
    # 1. Enforce lock check
    schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
    if not schedule or not schedule.get("live_project"):
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency is locked. Complete Live Project and get shortlisted first."
        )
    
    lp_conf = schedule["live_project"]
    if not lp_conf.get("analysis_completed") or lp_conf.get("decision") != "shortlist":
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency is locked. Complete Live Project and get shortlisted first."
        )

    # 2. Enforce schedule settings
    if not schedule.get("ai_fluency"):
        raise HTTPException(
            status_code=403,
            detail="AI Tool Fluency round has not been scheduled."
        )
        
    fluency_conf = schedule["ai_fluency"]
    
    # Scheduled time check
    scheduled_time = fluency_conf.get("scheduled_time")
    if scheduled_time:
        now_str = datetime.utcnow().isoformat()
        if now_str < scheduled_time:
            raise HTTPException(
                status_code=403,
                detail=f"This assessment starts at {scheduled_time}."
            )
            
    # Deadline check
    deadline = fluency_conf.get("deadline")
    if deadline:
        now_str = datetime.utcnow().isoformat()
        if now_str > deadline:
            raise HTTPException(
                status_code=403,
                detail="Assessment deadline has passed."
            )
            
    # Attempts check
    completed_count = await db["interview_evaluations"].count_documents({
        "user_id": user_id,
        "interview_type": "ai_fluency"
    })
    max_attempts = int(fluency_conf.get("max_attempts", 3))
    if completed_count >= max_attempts:
        raise HTTPException(
            status_code=403,
            detail=f"Maximum attempts ({max_attempts}) reached."
        )

    # 3. Fetch questions
    questions_doc = await db["ai_fluency_questions"].find_one({"user_id": user_id})
    if not questions_doc:
        raise HTTPException(
            status_code=400,
            detail="Fluency questions not initialized. Fetch questions first."
        )
    questions = questions_doc["questions"]

    # 4. Validate that all questions are answered properly based on type
    for q in questions:
        q_id = q.get("id")
        q_type = q.get("type", "text")
        ans = req.answers.get(q_id)
        
        if q_type == "text":
            if not ans or not isinstance(ans, str) or len(ans.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Please provide a complete answer (min 10 characters) for: {q.get('title')}"
                )
        elif q_type == "mcq":
            if not ans or not isinstance(ans, str) or len(ans.strip()) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Please select an option for: {q.get('title')}"
                )
        elif q_type == "msq":
            if not ans or not isinstance(ans, list) or len(ans) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Please select at least one option for: {q.get('title')}"
                )

    # 5. Call AI evaluation
    evaluation_result = await generate_fluency_evaluation(
        questions=questions,
        answers=req.answers,
        candidate_name=ctx["candidate_name"],
        applied_role=ctx["applied_role"]
    )

    # 6. Save attempt to interview_evaluations
    now = datetime.utcnow().isoformat()
    eval_doc = {
        "user_id": user_id,
        "candidate_name": ctx["candidate_name"],
        "applied_role": ctx["applied_role"],
        "interview_type": "ai_fluency",
        "final_score": evaluation_result.get("overall_score", 0),
        "evaluation": evaluation_result,
        "answers": req.answers,
        "created_at": now
    }
    await db["interview_evaluations"].insert_one(eval_doc)

    try:
        from app.services.email_service import notify_stage_completed
        import asyncio
        if current_user and current_user.get("email"):
            completed_count = await db["interview_evaluations"].count_documents({
                "user_id": user_id,
                "interview_type": "ai_fluency"
            })
            asyncio.create_task(notify_stage_completed(
                to_email=current_user["email"],
                candidate_name=current_user.get("full_name") or "Candidate",
                stage_name="AI Fluency",
                attempt_num=completed_count,
                score=evaluation_result.get("overall_score", 0)
            ))
    except Exception as email_err:
        print(f"Error sending fluency completion email: {email_err}")

    return {
        "status": "success",
        "evaluation": evaluation_result
    }
