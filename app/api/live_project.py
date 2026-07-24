from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

from app.database.database import db
from app.middleware.auth import get_current_user_optional
from app.services.live_project_service import generate_project_description, generate_project_evaluation

router = APIRouter(prefix="/api/live-project", tags=["Live Project Assessment"])

class DeliverableItem(BaseModel):
    url: str
    public_id: Optional[str] = None
    original_filename: Optional[str] = None

class SubmitProjectRequest(BaseModel):
    explanation: str
    deliverables: Dict[str, Optional[DeliverableItem]]

async def _load_candidate_context(user_id: str) -> dict:
    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    user = await db["users"].find_one({"_id": db["users"].codec_options.uuid_representation.to_uuid(user_id) if hasattr(db["users"].codec_options.uuid_representation, "to_uuid") else user_id})
    
    # Fallback to general lookup if UUID representation mismatch
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

@router.get("/prompt")
async def get_project_prompt(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Get the detailed Live Project prompt. Generates a new one if not cached.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    
    user_id = str(current_user["_id"])
    
    # 1. Enforce lock check
    schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
    if not schedule or not schedule.get("coding"):
        raise HTTPException(
            status_code=403,
            detail="Live Project is locked. You must complete the Coding assessment and be shortlisted first."
        )
    
    coding_conf = schedule["coding"]
    if not coding_conf.get("analysis_completed") or coding_conf.get("decision") != "shortlist":
        raise HTTPException(
            status_code=403,
            detail="Live Project is locked. You must complete the Coding assessment and be shortlisted first."
        )
        
    # 2. Check live project schedule config
    if not schedule.get("live_project"):
        raise HTTPException(
            status_code=403,
            detail="Live Project has not been scheduled by your recruiter yet."
        )
        
    lp_conf = schedule["live_project"]
    
    # Enforce scheduled start time
    scheduled_time = lp_conf.get("scheduled_time")
    if scheduled_time:
        now_str = datetime.utcnow().isoformat()
        if now_str < scheduled_time:
            raise HTTPException(
                status_code=403,
                detail=f"Your Live Project assessment is scheduled for {scheduled_time}. It has not started yet."
            )
            
    # Enforce deadline
    deadline = lp_conf.get("deadline")
    if deadline:
        now_str = datetime.utcnow().isoformat()
        if now_str > deadline:
            raise HTTPException(
                status_code=403,
                detail=f"The deadline for this Live Project assessment has passed ({deadline})."
            )

    # 3. Load or generate detailed prompt
    cached = await db["live_project_prompts"].find_one({"user_id": user_id})
    if cached:
        return cached["prompt"]

    ctx = await _load_candidate_context(user_id)
    topic = lp_conf.get("topic", "E-commerce platform backend")
    complexity = lp_conf.get("complexity", "medium")
    
    prompt_data = await generate_project_description(
        topic=topic,
        complexity=complexity,
        candidate_skills=ctx["candidate_skills"],
        applied_role=ctx["applied_role"]
    )
    
    await db["live_project_prompts"].insert_one({
        "user_id": user_id,
        "prompt": prompt_data,
        "created_at": datetime.utcnow().isoformat()
    })
    
    return prompt_data

@router.post("/submit")
async def submit_project(
    req: SubmitProjectRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
        
    user_id = str(current_user["_id"])
    ctx = await _load_candidate_context(user_id)
    
    # 1. Enforce lock check
    schedule = await db["interview_schedules"].find_one({"candidate_id": user_id})
    if not schedule or not schedule.get("coding"):
        raise HTTPException(
            status_code=403,
            detail="Live Project is locked. Complete Coding and get shortlisted first."
        )
    
    coding_conf = schedule["coding"]
    if not coding_conf.get("analysis_completed") or coding_conf.get("decision") != "shortlist":
        raise HTTPException(
            status_code=403,
            detail="Live Project is locked. Complete Coding and get shortlisted first."
        )

    # 2. Enforce schedule settings
    if not schedule.get("live_project"):
        raise HTTPException(
            status_code=403,
            detail="Live Project round has not been scheduled."
        )
        
    lp_conf = schedule["live_project"]
    
    # Scheduled time check
    scheduled_time = lp_conf.get("scheduled_time")
    if scheduled_time:
        now_str = datetime.utcnow().isoformat()
        if now_str < scheduled_time:
            raise HTTPException(
                status_code=403,
                detail=f"This assessment starts at {scheduled_time}."
            )
            
    # Deadline check
    deadline = lp_conf.get("deadline")
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
        "interview_type": "live_project"
    })
    max_attempts = int(lp_conf.get("max_attempts", 3))
    if completed_count >= max_attempts:
        raise HTTPException(
            status_code=403,
            detail=f"Maximum attempts ({max_attempts}) reached."
        )

    # 3. Validate required deliverables chosen by recruiter
    required = lp_conf.get("required_deliverables") or ["source_code", "documentation", "architecture", "deployment_url", "demo_video"]
    
    for key in required:
        item = req.deliverables.get(key)
        if not item or not item.url:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required deliverable: {key.replace('_', ' ').capitalize()}"
            )

    # 4. Fetch prompt
    prompt_doc = await db["live_project_prompts"].find_one({"user_id": user_id})
    if not prompt_doc:
        raise HTTPException(
            status_code=400,
            detail="Project prompt not initialized. Get the prompt first."
        )
    prompt_data = prompt_doc["prompt"]

    # 5. Call AI evaluation
    deliverables_dict = {k: v.model_dump() if v else None for k, v in req.deliverables.items()}
    
    evaluation_result = await generate_project_evaluation(
        prompt=prompt_data,
        explanation=req.explanation,
        deliverables=deliverables_dict,
        candidate_name=ctx["candidate_name"],
        applied_role=ctx["applied_role"]
    )

    # 6. Save attempt to interview_evaluations
    now = datetime.utcnow().isoformat()
    eval_doc = {
        "user_id": user_id,
        "candidate_name": ctx["candidate_name"],
        "applied_role": ctx["applied_role"],
        "interview_type": "live_project",
        "final_score": evaluation_result.get("overall_score", 0),
        "evaluation": evaluation_result,
        "deliverables": deliverables_dict,
        "explanation": req.explanation,
        "created_at": now
    }
    await db["interview_evaluations"].insert_one(eval_doc)

    try:
        from app.services.email_service import notify_stage_completed
        import asyncio
        if current_user and current_user.get("email"):
            completed_count = await db["interview_evaluations"].count_documents({
                "user_id": user_id,
                "interview_type": "live_project"
            })
            asyncio.create_task(notify_stage_completed(
                to_email=current_user["email"],
                candidate_name=current_user.get("full_name") or "Candidate",
                stage_name="Live Project",
                attempt_num=completed_count,
                score=evaluation_result.get("overall_score", 0)
            ))
    except Exception as email_err:
        print(f"Error sending live project completion email: {email_err}")

    return {
        "status": "success",
        "evaluation": evaluation_result
    }
