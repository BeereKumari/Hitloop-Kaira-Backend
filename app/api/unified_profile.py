from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from bson import ObjectId

from app.database.database import db
from app.middleware.auth import get_current_user_optional, get_current_user
from app.services.unified_profile_service import generate_unified_candidate_profile

router = APIRouter(tags=["Unified Candidate Profile"])

class OfferPayload(BaseModel):
    company_name: str
    founder_name: str
    salary: str

def require_recruiter(user: dict):
    if not user or user.get("role") != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiter access required.")

async def _get_or_generate_profile(candidate_id: str) -> dict:
    # 1. Check Cache
    cached = await db["unified_candidate_profiles"].find_one({"user_id": candidate_id})
    if cached:
        cached["id"] = str(cached["_id"])
        del cached["_id"]
        return cached

    # 2. Load context
    user = await db["users"].find_one({"_id": db["users"].codec_options.uuid_representation.to_uuid(candidate_id) if hasattr(db["users"].codec_options.uuid_representation, "to_uuid") else candidate_id})
    if not user:
        try:
            user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
        except Exception:
            pass

    candidate_name = "Candidate"
    target_role = "Software Engineer"
    
    profile = await db["candidate_profiles"].find_one({"user_id": candidate_id})
    if profile:
        personal = profile.get("personal", {})
        candidate_name = personal.get("full_name", candidate_name)
        target_role = personal.get("target_role", target_role)
    elif user:
        candidate_name = user.get("full_name", candidate_name)

    resume_analysis = await db["resume_analyses"].find_one({"user_id": candidate_id}) or {}
    project_analysis = await db["project_analyses"].find_one({"user_id": candidate_id}) or {}

    evaluations = []
    cursor = db["interview_evaluations"].find({"user_id": candidate_id})
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        evaluations.append(doc)

    coding_cursor = db["coding_evaluations"].find({"user_id": candidate_id})
    async for doc in coding_cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        doc["interview_type"] = "coding"
        if "evaluation" in doc:
            eval_data = doc["evaluation"]
            doc["final_score"] = eval_data.get("overall_score")
            doc["feedback"] = eval_data.get("overall_summary")
        evaluations.append(doc)

    # 3. Generate
    generated = await generate_unified_candidate_profile(
        candidate_name=candidate_name,
        target_role=target_role,
        resume_analysis=resume_analysis,
        project_analysis=project_analysis,
        evaluations=evaluations
    )
    
    generated["user_id"] = candidate_id
    generated["created_at"] = datetime.utcnow().isoformat()
    
    await db["unified_candidate_profiles"].insert_one(generated)
    
    generated["id"] = str(generated["_id"])
    del generated["_id"]
    return generated

def _check_schedule_completed(schedule: dict) -> bool:
    if not schedule:
        return False
    # Check that all scheduled rounds (non-null in DB) are completed
    active_any = False
    for field in ["audio", "video", "coding", "live_project", "ai_fluency", "behaviour"]:
        field_conf = schedule.get(field)
        if field_conf and isinstance(field_conf, dict) and field_conf.get("complexity"):
            active_any = True
            if not field_conf.get("analysis_completed"):
                return False
    return active_any

@router.get("/api/recruiter/candidates/{candidate_id}/unified-profile")
async def get_recruiter_candidate_profile(
    candidate_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    require_recruiter(current_user)

    schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not schedule or not _check_schedule_completed(schedule):
        return {
            "status": "pending_assessments",
            "message": "Please complete all assessments for final report."
        }

    profile = await _get_or_generate_profile(candidate_id)

    # Fetch recruiter profile company and name
    recruiter_company = ""
    recruiter_founder = ""
    recruiter_prof = await db["recruiter_profiles"].find_one({"user_id": str(current_user["_id"])})
    if recruiter_prof:
        recruiter_company = (recruiter_prof.get("company_name") or recruiter_prof.get("company") or "").strip()
        recruiter_founder = f"{recruiter_prof.get('first_name', '')} {recruiter_prof.get('last_name', '')}".strip()
        if not recruiter_founder:
            recruiter_founder = recruiter_prof.get("full_name", "").strip()
    if not recruiter_founder:
        recruiter_founder = current_user.get("full_name", "").strip()

    return {
        "status": "success",
        "profile": profile,
        "hiring_status": schedule.get("hiring_status"),
        "offer_details": schedule.get("offer_details"),
        "recruiter_company": recruiter_company,
        "recruiter_founder": recruiter_founder
    }

@router.post("/api/recruiter/candidates/{candidate_id}/hiring-decision/offer")
async def send_candidate_offer(
    candidate_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    require_recruiter(current_user)

    schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not schedule:
        raise HTTPException(status_code=404, detail="Candidate schedule not found.")

    # Retrieve recruiter profile company and name
    company_name = "Kaira Client"
    founder_name = "Hiring Panel"
    
    recruiter_prof = await db["recruiter_profiles"].find_one({"user_id": str(current_user["_id"])})
    if recruiter_prof:
        company_name = (recruiter_prof.get("company_name") or recruiter_prof.get("company") or "Kaira Client").strip()
        founder_name = f"{recruiter_prof.get('first_name', '')} {recruiter_prof.get('last_name', '')}".strip()
        if not founder_name:
            founder_name = recruiter_prof.get("full_name", "").strip()
    if not founder_name:
        founder_name = current_user.get("full_name", "Hiring Panel").strip()

    # Get target role (applied_role first, then target_role)
    target_role = "Software Engineer"
    profile = await db["candidate_profiles"].find_one({"user_id": candidate_id})
    if profile:
        target_role = profile.get("personal", {}).get("applied_role") or profile.get("personal", {}).get("target_role") or target_role

    now = datetime.utcnow().isoformat()
    offer_details = {
        "company_name": company_name,
        "founder_name": founder_name,
        "role": target_role,
        "decided_at": now
    }

    await db["interview_schedules"].update_one(
        {"candidate_id": candidate_id},
        {
            "$set": {
                "hiring_status": "offer_sent",
                "offer_details": offer_details
            }
        }
    )

    try:
        from bson import ObjectId
        from app.services.email_service import notify_hiring_decision
        import asyncio
        candidate_user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
        if candidate_user:
            candidate_email = candidate_user.get("email")
            candidate_name = candidate_user.get("full_name") or "Candidate"
            if candidate_email:
                asyncio.create_task(notify_hiring_decision(
                    to_email=candidate_email,
                    candidate_name=candidate_name,
                    decision="offer_sent",
                    details=offer_details
                ))
    except Exception as email_err:
        print(f"Error sending offer email: {email_err}")

    return {"status": "success", "offer_details": offer_details}

@router.post("/api/recruiter/candidates/{candidate_id}/hiring-decision/reject")
async def reject_candidate(
    candidate_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    require_recruiter(current_user)

    schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not schedule:
        raise HTTPException(status_code=404, detail="Candidate schedule not found.")

    await db["interview_schedules"].update_one(
        {"candidate_id": candidate_id},
        {
            "$set": {
                "hiring_status": "rejected",
                "rejected_at": datetime.utcnow().isoformat()
            }
        }
    )

    try:
        from bson import ObjectId
        from app.services.email_service import notify_hiring_decision
        import asyncio
        candidate_user = await db["users"].find_one({"_id": ObjectId(candidate_id)})
        if candidate_user:
            candidate_email = candidate_user.get("email")
            candidate_name = candidate_user.get("full_name") or "Candidate"
            if candidate_email:
                asyncio.create_task(notify_hiring_decision(
                    to_email=candidate_email,
                    candidate_name=candidate_name,
                    decision="rejected",
                    details={"company_name": "Kaira Partner"}
                ))
    except Exception as email_err:
        print(f"Error sending reject email: {email_err}")

    return {"status": "success"}

@router.get("/api/candidate/final-report")
async def get_candidate_final_report(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    candidate_id = str(current_user["_id"])
    schedule = await db["interview_schedules"].find_one({"candidate_id": candidate_id})
    if not schedule or not _check_schedule_completed(schedule):
        return {
            "status": "pending_assessments",
            "message": "Please complete all assessments for final report."
        }

    hiring_status = schedule.get("hiring_status")
    
    # If decided:
    if hiring_status == "offer_sent":
        profile = await _get_or_generate_profile(candidate_id)
        return {
            "status": "hired",
            "profile": profile,
            "offer_details": schedule.get("offer_details")
        }
    elif hiring_status == "rejected":
        return {
            "status": "rejected",
            "message": "Thank you for participating in our interview process. After careful consideration of your evaluation performance across all rounds, we regret to inform you that we are proceeding with other candidates whose profiles align more closely with our current team needs. We wish you the best of luck in your career search."
        }
    else:
        profile = await _get_or_generate_profile(candidate_id)
        return {
            "status": "under_review",
            "profile": profile,
            "message": "Your unified profile is currently under review by the recruiter."
        }
