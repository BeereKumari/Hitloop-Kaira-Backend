from datetime import datetime, timedelta
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from app.database.database import db
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/api/recruiter", tags=["Recruiter"])


def require_recruiter(user: dict):
    if user.get("role") != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiter access required.")


# ─── Recruiter Profile ───────────────────────────────────────────────────────

@router.get("/profile")
async def get_recruiter_profile(current_user: dict = Depends(get_current_user)):
    require_recruiter(current_user)

    profile = await db["recruiter_profiles"].find_one({"user_id": str(current_user["_id"])})
    if not profile:
        profile = {
            "user_id": str(current_user["_id"]),
            "full_name": current_user.get("full_name", ""),
            "email": current_user.get("email", ""),
            "company": None,
            "job_title": None,
            "department": None,
            "company_size": None,
            "industry": None,
            "linkedin_url": None,
            "hiring_for": None,
        }

    return {
        "status": "success",
        "profile": {
            "id": str(profile["_id"]) if "_id" in profile else None,
            "user_id": profile.get("user_id"),
            "full_name": profile.get("full_name", current_user.get("full_name", "")),
            "email": profile.get("email", current_user.get("email", "")),
            "company": profile.get("company"),
            "job_title": profile.get("job_title"),
            "department": profile.get("department"),
            "company_size": profile.get("company_size"),
            "industry": profile.get("industry"),
            "linkedin_url": profile.get("linkedin_url"),
            "hiring_for": profile.get("hiring_for"),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
        },
    }


@router.put("/profile")
async def update_recruiter_profile(
    current_user: dict = Depends(get_current_user),
    full_name: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    job_title: Optional[str] = None,
    department: Optional[str] = None,
    company_size: Optional[str] = None,
    industry: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    hiring_for: Optional[str] = None,
):
    require_recruiter(current_user)

    now = datetime.utcnow().isoformat()
    updates = {"updated_at": now}

    if full_name is not None:
        updates["full_name"] = full_name.strip()
        await db["users"].update_one(
            {"_id": current_user["_id"]},
            {"$set": {"full_name": full_name.strip(), "updated_at": now}}
        )
    if phone is not None:
        await db["users"].update_one(
            {"_id": current_user["_id"]},
            {"$set": {"phone": phone.strip() if phone else None, "updated_at": now}}
        )
    if company is not None:
        updates["company"] = company.strip()
    if job_title is not None:
        updates["job_title"] = job_title.strip()
    if department is not None:
        updates["department"] = department.strip() if department else None
    if company_size is not None:
        updates["company_size"] = company_size
    if industry is not None:
        updates["industry"] = industry
    if linkedin_url is not None:
        updates["linkedin_url"] = linkedin_url.strip() if linkedin_url else None
    if hiring_for is not None:
        updates["hiring_for"] = hiring_for.strip() if hiring_for else None

    await db["recruiter_profiles"].update_one(
        {"user_id": str(current_user["_id"])},
        {"$set": updates},
        upsert=True,
    )

    return {"status": "success", "message": "Profile updated successfully."}


# ─── Dashboard Analytics ─────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    require_recruiter(current_user)

    total_candidates = await db["users"].count_documents({"role": "candidate"})
    total_profiles = await db["candidate_profiles"].count_documents({})
    completed_assessments = await db["coding_evaluations"].count_documents({})
    resume_analyses = await db["resume_analyses"].count_documents({})
    interview_sessions = await db["interview_sessions"].count_documents({})
    project_analyses = await db["project_analyses"].count_documents({})

    recent_candidates = []
    cursor = db["candidate_profiles"].find().sort("created_at", -1).limit(5)
    async for p in cursor:
        user = await db["users"].find_one({"_id": ObjectId(p["user_id"])})
        if user:
            resume = await db["resume_analyses"].find_one({"user_id": p["user_id"]})
            coding = await db["coding_evaluations"].find_one({"user_id": p["user_id"]})
            interviews = await db["interview_sessions"].count_documents({"user_id": p["user_id"]})

            stage = "registered"
            if resume:
                stage = "resume_analyzed"
            if coding:
                stage = "assessment_complete"
            if interviews > 0:
                stage = "interviewed"

            recent_candidates.append({
                "id": p["user_id"],
                "full_name": user.get("full_name", ""),
                "email": user.get("email", ""),
                "created_at": p.get("created_at", ""),
                "applied_role": (p.get("personal") or {}).get("applied_role"),
                "resume_score": resume.get("overall_score") if resume else None,
                "coding_score": coding.get("overall_score") if coding else None,
                "stage": stage,
                "profile_completion": p.get("profile_completion", 0),
            })

    notifications = []
    cursor_n = db["notifications"].find(
        {"user_id": str(current_user["_id"])}
    ).sort("created_at", -1).limit(5)
    async for n in cursor_n:
        notifications.append({
            "id": str(n["_id"]),
            "type": n.get("type", ""),
            "message": n.get("message", ""),
            "read": n.get("read", False),
            "created_at": n.get("created_at", ""),
        })

    pipeline = {
        "registered": total_profiles,
        "resume_analyzed": resume_analyses,
        "assessment_complete": completed_assessments,
        "interviewed": interview_sessions,
    }

    return {
        "status": "success",
        "stats": {
            "total_candidates": total_candidates,
            "total_profiles": total_profiles,
            "completed_assessments": completed_assessments,
            "resume_analyses": resume_analyses,
            "interview_sessions": interview_sessions,
            "project_analyses": project_analyses,
        },
        "recent_candidates": recent_candidates,
        "notifications": notifications,
        "pipeline": pipeline,
    }


# ─── Candidate List ──────────────────────────────────────────────────────────

@router.get("/candidates")
async def list_candidates(
    search: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    sort_by: Optional[str] = Query("created_at"),
    sort_order: Optional[str] = Query("desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    require_recruiter(current_user)

    query = {"role": "candidate"}
    if search:
        query["$or"] = [
            {"full_name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
        ]

    total = await db["users"].count_documents(query)
    skip = (page - 1) * limit
    sort_dir = -1 if sort_order == "desc" else 1

    cursor = db["users"].find(query).sort(sort_by, sort_dir).skip(skip).limit(limit)
    candidates = []
    async for u in cursor:
        profile = await db["candidate_profiles"].find_one({"user_id": str(u["_id"])})
        resume = await db["resume_analyses"].find_one({"user_id": str(u["_id"])})
        coding = await db["coding_evaluations"].find_one({"user_id": str(u["_id"])})

        stage_val = "registered"
        if resume:
            stage_val = "resume_analyzed"
        if coding:
            stage_val = "assessment_complete"

        candidates.append({
            "id": str(u["_id"]),
            "full_name": u.get("full_name", ""),
            "email": u.get("email", ""),
            "phone": u.get("phone"),
            "created_at": u.get("created_at", ""),
            "last_login": u.get("last_login"),
            "stage": stage_val,
            "applied_role": (profile or {}).get("personal", {}).get("applied_role") if profile else None,
            "resume_score": resume.get("overall_score") if resume else None,
            "coding_score": coding.get("overall_score") if coding else None,
            "profile_completion": (profile or {}).get("profile_completion", 0) if profile else 0,
        })

    return {
        "status": "success",
        "candidates": candidates,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit if total > 0 else 1,
    }


# ─── Candidate Detail ────────────────────────────────────────────────────────

@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str, current_user: dict = Depends(get_current_user)):
    require_recruiter(current_user)

    try:
        user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid candidate ID.")
    if not user or user.get("role") != "candidate":
        raise HTTPException(status_code=404, detail="Candidate not found.")

    profile = await db["candidate_profiles"].find_one({"user_id": candidate_id})
    resume = await db["resume_analyses"].find_one({"user_id": candidate_id})
    project = await db["project_analyses"].find_one({"user_id": candidate_id})
    coding = await db["coding_evaluations"].find_one({"user_id": candidate_id})

    interviews = []
    cursor = db["interview_sessions"].find({"user_id": candidate_id}).sort("created_at", -1)
    async for s in cursor:
        interviews.append({
            "id": str(s["_id"]),
            "type": s.get("type", ""),
            "status": s.get("status", ""),
            "created_at": s.get("created_at", ""),
            "score": s.get("overall_score"),
        })

    return {
        "status": "success",
        "candidate": {
            "id": candidate_id,
            "full_name": user.get("full_name", ""),
            "email": user.get("email", ""),
            "phone": user.get("phone"),
            "created_at": user.get("created_at", ""),
            "last_login": user.get("last_login"),
            "profile": {
                "personal": (profile or {}).get("personal"),
                "education": (profile or {}).get("education"),
                "experience": (profile or {}).get("experience"),
                "skills": (profile or {}).get("skills"),
                "uploads": (profile or {}).get("uploads"),
                "profile_completion": (profile or {}).get("profile_completion", 0),
            } if profile else None,
            "resume_analysis": {
                "overall_score": resume.get("overall_score"),
                "skills": resume.get("skills"),
                "strengths": resume.get("strengths"),
                "weaknesses": resume.get("weaknesses"),
                "recommendation": resume.get("recommendation"),
                "created_at": resume.get("created_at"),
            } if resume else None,
            "project_analysis": {
                "overall_score": project.get("overall_score"),
                "complexity_score": project.get("complexity_score"),
                "architecture_score": project.get("architecture_score"),
                "innovation_score": project.get("innovation_score"),
                "created_at": project.get("created_at"),
            } if project else None,
            "coding_assessment": {
                "overall_score": coding.get("overall_score"),
                "total_tests": coding.get("total_tests"),
                "passed_tests": coding.get("passed_tests"),
                "skills_demonstrated": coding.get("skills_demonstrated"),
                "recommended_topics": coding.get("recommended_topics"),
                "recruiter_insight": coding.get("recruiter_insight"),
                "created_at": coding.get("created_at"),
            } if coding else None,
            "interviews": interviews,
        },
    }


# ─── Notifications ───────────────────────────────────────────────────────────

@router.get("/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    require_recruiter(current_user)

    notifications = []
    cursor = db["notifications"].find({"user_id": str(current_user["_id"])}).sort("created_at", -1).limit(20)
    async for n in cursor:
        notifications.append({
            "id": str(n["_id"]),
            "type": n.get("type", ""),
            "message": n.get("message", ""),
            "read": n.get("read", False),
            "created_at": n.get("created_at", ""),
        })

    return {"status": "success", "notifications": notifications}


@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, current_user: dict = Depends(get_current_user)):
    require_recruiter(current_user)

    try:
        await db["notifications"].update_one(
            {"_id": ObjectId(notification_id), "user_id": str(current_user["_id"])},
            {"$set": {"read": True}}
        )
    except Exception:
        pass

    return {"status": "success"}


# ─── Candidate Full Analysis (for recruiter view) ────────────────────────────

@router.get("/candidates/{candidate_id}/analysis")
async def get_candidate_full_analysis(
    candidate_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the full resume + project analysis data for a candidate."""
    require_recruiter(current_user)

    try:
        user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid candidate ID.")
    if not user or user.get("role") != "candidate":
        raise HTTPException(status_code=404, detail="Candidate not found.")

    resume_doc  = await db["resume_analyses"].find_one({"user_id": candidate_id})
    if not resume_doc:
        # Fallback: look up candidate profile and link anonymous resume analyses
        profile = await db["candidate_profiles"].find_one({"user_id": candidate_id})
        if profile and profile.get("uploads", {}).get("resume", {}).get("public_id"):
            public_id = profile["uploads"]["resume"]["public_id"]
            resume_doc = await db["resume_analyses"].find_one({
                "cloudinary_public_id": public_id,
                "user_id": "anonymous"
            })
            if resume_doc:
                await db["resume_analyses"].update_one(
                    {"_id": resume_doc["_id"]},
                    {"$set": {"user_id": candidate_id}}
                )
                resume_doc["user_id"] = candidate_id

    project_doc = await db["project_analyses"].find_one({"user_id": candidate_id})
    decision    = await db["recruiter_decisions"].find_one({"candidate_id": candidate_id})

    def _clean(doc):
        if not doc:
            return None
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        return doc

    return {
        "status": "success",
        "candidate_id": candidate_id,
        "full_name": user.get("full_name", ""),
        "email": user.get("email", ""),
        "resume_analysis":  _clean(resume_doc),
        "project_analysis": _clean(project_doc),
        "recruiter_decision": {
            "action": decision.get("action"),
            "analysis_completed": decision.get("analysis_completed", False),
            "notes": decision.get("notes"),
            "updated_at": decision.get("updated_at"),
        } if decision else None,
    }


# ─── Recruiter Decision (Complete Analysis / Shortlist / Reject) ─────────────

from pydantic import BaseModel

class DecisionPayload(BaseModel):
    action: str            # "complete" | "shortlist" | "reject"
    notes: Optional[str] = None


@router.post("/candidates/{candidate_id}/decision")
async def set_candidate_decision(
    candidate_id: str,
    payload: DecisionPayload,
    current_user: dict = Depends(get_current_user),
):
    """Store or update the recruiter's hiring decision for a candidate."""
    require_recruiter(current_user)

    allowed = {"complete", "shortlist", "reject"}
    if payload.action not in allowed:
        raise HTTPException(status_code=400, detail=f"action must be one of {allowed}")

    now = datetime.utcnow().isoformat()
    update = {
        "candidate_id": candidate_id,
        "recruiter_id": str(current_user["_id"]),
        "action": payload.action,
        "analysis_completed": payload.action == "complete" or (
            # preserve existing completed flag if switching to shortlist/reject
            True if payload.action in {"shortlist", "reject"} else False
        ),
        "notes": payload.notes,
        "updated_at": now,
    }

    existing = await db["recruiter_decisions"].find_one({"candidate_id": candidate_id})
    if existing:
        # Preserve analysis_completed=True once set
        if existing.get("analysis_completed"):
            update["analysis_completed"] = True
        await db["recruiter_decisions"].update_one(
            {"candidate_id": candidate_id},
            {"$set": update},
        )
    else:
        update["created_at"] = now
        await db["recruiter_decisions"].insert_one(update)

    return {"status": "success", "action": payload.action}


# ─── Recruiter Interview Scheduling & Decisions ──────────────────────────────

class SchedulePayload(BaseModel):
    interview_type: str    # "chat" | "audio" | "video" | "coding" | "live_project" | "ai_fluency" | "behaviour"
    complexity: str        # "easy" | "medium" | "hard" | "expert" | "low" | "high"
    num_questions: int
    max_attempts: int
    scheduled_time: Optional[str] = None
    deadline: Optional[str] = None
    complexities: Optional[List[str]] = None
    topic: Optional[str] = None
    required_deliverables: Optional[List[str]] = None


class InterviewDecisionPayload(BaseModel):
    interview_type: str
    action: str            # "complete" | "shortlist" | "reject"
    notes: Optional[str] = None


@router.get("/candidates/{candidate_id}/interviews")
async def get_candidate_interviews(
    candidate_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return candidates interview & assessment schedules and all completed evaluations."""
    require_recruiter(current_user)

    schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not schedule:
        schedule = {
            "candidate_id": candidate_id,
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
        # Ensure all fields are initialized
        for field in ["chat", "audio", "video", "coding", "live_project", "ai_fluency", "behaviour"]:
            if field not in schedule:
                schedule[field] = None

    evaluations = []
    # 1. Interview evaluations
    cursor = db["interview_evaluations"].find({"user_id": candidate_id})
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        if "interview_type" not in doc:
            doc["interview_type"] = "chat"
        evaluations.append(doc)

    # 2. Coding evaluations
    coding_cursor = db["coding_evaluations"].find({"user_id": candidate_id})
    async for doc in coding_cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        doc["interview_type"] = "coding"
        # Map coding evaluation fields for visual compatibility
        if "evaluation" in doc:
            eval_data = doc["evaluation"]
            doc["final_score"] = eval_data.get("overall_score")
            doc["recommendation"] = eval_data.get("recommendation")
            doc["feedback"] = eval_data.get("overall_summary")
        evaluations.append(doc)

    return {
        "status": "success",
        "schedule": schedule,
        "evaluations": evaluations,
    }


@router.post("/candidates/{candidate_id}/interviews/schedule")
async def schedule_candidate_interview(
    candidate_id: str,
    payload: SchedulePayload,
    current_user: dict = Depends(get_current_user),
):
    """Save or update configuration and scheduled time for an interview or assessment type."""
    require_recruiter(current_user)

    allowed_types = {"chat", "audio", "video", "coding", "live_project", "ai_fluency", "behaviour"}
    if payload.interview_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"interview_type must be in {allowed_types}")

    # Look up candidate resume decision first
    resume_decision = await db["recruiter_decisions"].find_one({"candidate_id": candidate_id})
    resume_shortlisted = resume_decision and resume_decision.get("action") == "shortlist"

    # Look up existing interview schedules
    existing_schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})

    # Rule 3: Chat schedule requires resume to be shortlisted
    if payload.interview_type == "chat":
        if not resume_shortlisted:
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Chat interview: Candidate must first be shortlisted in their Resume evaluation."
            )

    # Rule 4: Audio schedule requires Chat analysis to be completed and shortlisted
    elif payload.interview_type == "audio":
        if not existing_schedule or not existing_schedule.get("chat"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Audio interview: Chat interview must be scheduled and completed first."
            )
        chat_config = existing_schedule["chat"]
        if not chat_config.get("analysis_completed") or chat_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Audio interview: Chat interview analysis must be completed and candidate shortlisted."
            )

    # Rule 5: Video schedule requires Audio analysis to be completed and shortlisted
    elif payload.interview_type == "video":
        if not existing_schedule or not existing_schedule.get("audio"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Video interview: Audio interview must be scheduled and completed first."
            )
        audio_config = existing_schedule["audio"]
        if not audio_config.get("analysis_completed") or audio_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Video interview: Audio interview analysis must be completed and candidate shortlisted."
            )

    # Rule 6: Coding Assessment requires Video analysis to be completed and shortlisted
    elif payload.interview_type == "coding":
        if not existing_schedule or not existing_schedule.get("video"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Coding assessment: Video interview must be scheduled and completed first."
            )
        video_config = existing_schedule["video"]
        if not video_config.get("analysis_completed") or video_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Coding assessment: Video interview analysis must be completed and candidate shortlisted."
            )

    # Rule 7: Live Project requires Coding analysis to be completed and shortlisted
    elif payload.interview_type == "live_project":
        if not existing_schedule or not existing_schedule.get("coding"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Live Project: Coding assessment must be scheduled and completed first."
            )
        coding_config = existing_schedule["coding"]
        if not coding_config.get("analysis_completed") or coding_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Live Project: Coding assessment analysis must be completed and candidate shortlisted."
            )

    # Rule 8: AI Tool Fluency requires Live Project to be completed and shortlisted
    elif payload.interview_type == "ai_fluency":
        if not existing_schedule or not existing_schedule.get("live_project"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule AI Tool Fluency: Live Project must be scheduled and completed first."
            )
        lp_config = existing_schedule["live_project"]
        if not lp_config.get("analysis_completed") or lp_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule AI Tool Fluency: Live Project analysis must be completed and candidate shortlisted."
            )

    # Rule 9: Behaviour requires AI Tool Fluency to be completed and shortlisted
    elif payload.interview_type == "behaviour":
        if not existing_schedule or not existing_schedule.get("ai_fluency"):
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Behaviour assessment: AI Tool Fluency must be scheduled and completed first."
            )
        ai_config = existing_schedule["ai_fluency"]
        if not ai_config.get("analysis_completed") or ai_config.get("decision") != "shortlist":
            raise HTTPException(
                status_code=403,
                detail="Cannot schedule Behaviour assessment: AI Tool Fluency analysis must be completed and candidate shortlisted."
            )

    # Initialize the stage sub-object if it's null or missing
    if not existing_schedule or existing_schedule.get(payload.interview_type) is None:
        await db["interview_schedules"].update_one(
            {"candidate_id": candidate_id},
            {"$set": {payload.interview_type: {}}},
            upsert=True
        )

    update_fields = {
        f"{payload.interview_type}.complexity": payload.complexity,
        f"{payload.interview_type}.num_questions": payload.num_questions,
        f"{payload.interview_type}.max_attempts": payload.max_attempts,
        f"{payload.interview_type}.scheduled_time": payload.scheduled_time,
        f"{payload.interview_type}.deadline": payload.deadline,
    }
    if payload.complexities is not None:
        update_fields[f"{payload.interview_type}.complexities"] = payload.complexities
    if payload.topic is not None:
        update_fields[f"{payload.interview_type}.topic"] = payload.topic
    if payload.required_deliverables is not None:
        update_fields[f"{payload.interview_type}.required_deliverables"] = payload.required_deliverables

    await db["interview_schedules"].update_one(
        {"candidate_id": candidate_id},
        {"$set": update_fields},
        upsert=True
    )

    try:
        from bson import ObjectId
        from app.services.email_service import notify_stage_scheduled
        import asyncio
        candidate_user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
        if candidate_user:
            candidate_email = candidate_user.get("email")
            candidate_name = candidate_user.get("full_name") or "Candidate"
            if candidate_email:
                details = {
                    "complexity": payload.complexity,
                    "num_questions": payload.num_questions,
                    "scheduled_time": payload.scheduled_time,
                    "deadline": payload.deadline,
                }
                asyncio.create_task(notify_stage_scheduled(
                    to_email=candidate_email,
                    candidate_name=candidate_name,
                    stage_name=payload.interview_type.replace("_", " ").title(),
                    details=details
                ))
    except Exception as email_err:
        print(f"Error sending scheduling email: {email_err}")

    return {"status": "success", "message": f"Successfully configured/scheduled {payload.interview_type}."}


@router.post("/candidates/{candidate_id}/interviews/decision")
async def set_interview_decision(
    candidate_id: str,
    payload: InterviewDecisionPayload,
    current_user: dict = Depends(get_current_user),
):
    """Save evaluation decision (Complete Analysis / Shortlist / Reject) for a specific interview or assessment type."""
    require_recruiter(current_user)

    allowed_types = {"chat", "audio", "video", "coding", "live_project", "ai_fluency", "behaviour"}
    allowed_actions = {"complete", "shortlist", "reject"}
    if payload.interview_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"interview_type must be in {allowed_types}")
    if payload.action not in allowed_actions:
        raise HTTPException(status_code=400, detail=f"action must be in {allowed_actions}")

    now = datetime.utcnow().isoformat()
    
    update_fields = {
        f"{payload.interview_type}.decision": payload.action,
        f"{payload.interview_type}.analysis_completed": payload.action == "complete" or (
            True if payload.action in {"shortlist", "reject"} else False
        ),
        f"{payload.interview_type}.decision_notes": payload.notes,
        f"{payload.interview_type}.decision_updated_at": now,
    }

    existing = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not existing or existing.get(payload.interview_type) is None:
        await db["interview_schedules"].update_one(
            {"candidate_id": candidate_id},
            {"$set": {payload.interview_type: {}}},
            upsert=True
        )
        existing = await db["interview_schedules"].find_one({"candidate_id": candidate_id})

    if existing and existing.get(payload.interview_type):
        type_data = existing[payload.interview_type]
        if type_data.get("analysis_completed"):
            update_fields[f"{payload.interview_type}.analysis_completed"] = True

    await db["interview_schedules"].update_one(
        {"candidate_id": candidate_id},
        {"$set": update_fields},
        upsert=True
    )

    try:
        from bson import ObjectId
        from app.services.email_service import notify_stage_decision
        import asyncio
        candidate_user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
        if candidate_user:
            candidate_email = candidate_user.get("email")
            candidate_name = candidate_user.get("full_name") or "Candidate"
            if candidate_email:
                asyncio.create_task(notify_stage_decision(
                    to_email=candidate_email,
                    candidate_name=candidate_name,
                    stage_name=payload.interview_type.replace("_", " ").title(),
                    decision=payload.action,
                    notes=payload.notes or ""
                ))
    except Exception as email_err:
        print(f"Error sending stage decision email: {email_err}")

    return {"status": "success", "action": payload.action}


