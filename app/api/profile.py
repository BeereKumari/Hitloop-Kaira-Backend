from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from app.database.database import db
from app.models.profile import CandidateProfileUpdate
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/api/profile", tags=["Candidate Profile"])


def calc_completion(profile: dict) -> int:
    required_fields = []
    personal = profile.get("personal") or {}
    education = profile.get("education") or {}
    experience = profile.get("experience") or {}
    skills = profile.get("skills") or {}
    uploads = profile.get("uploads") or {}

    required_fields.extend([
        personal.get("full_name"),
        personal.get("email"),
        personal.get("phone"),
        personal.get("location"),
        personal.get("applied_role"),
        education.get("highest_degree"),
        education.get("university"),
        experience.get("years_of_experience"),
        skills.get("core_skills"),
        uploads.get("resume"),
    ])

    filled = sum(1 for f in required_fields if f and (str(f).strip() if isinstance(f, str) else True))
    return round((filled / len(required_fields)) * 100) if required_fields else 0


def serialize_profile(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "user_id": doc.get("user_id", ""),
        "personal": doc.get("personal", {}),
        "education": doc.get("education", {}),
        "experience": doc.get("experience", {}),
        "skills": doc.get("skills", {}),
        "uploads": doc.get("uploads", {}),
        "profile_completion": doc.get("profile_completion", 0),
        "autosave_status": doc.get("autosave_status", "saved"),
        "created_at": doc.get("created_at", ""),
        "updated_at": doc.get("updated_at", ""),
    }


@router.get("")
async def get_profile(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    if not profile:
        now = datetime.utcnow().isoformat()
        profile_doc = {
            "user_id": user_id,
            "personal": {
                "full_name": current_user.get("full_name", ""),
                "email": current_user.get("email", ""),
                "phone": current_user.get("phone"),
                "location": None,
                "linkedin": None,
                "github": None,
                "portfolio_url": None,
                "applied_role": None,
            },
            "education": {"highest_degree": None, "university": None},
            "experience": {"years_of_experience": None, "current_company": None},
            "skills": {"core_skills": None},
            "profile_completion": 0,
            "autosave_status": "saved",
            "created_at": now,
            "updated_at": now,
        }
        result = await db["candidate_profiles"].insert_one(profile_doc)
        profile_doc["_id"] = result.inserted_id
        return {"status": "success", "data": serialize_profile(profile_doc)}

    return {"status": "success", "data": serialize_profile(profile)}


@router.put("")
async def update_profile(
    data: CandidateProfileUpdate,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    now = datetime.utcnow().isoformat()
    update_fields = {"updated_at": now}

    if data.personal is not None:
        personal_dict = data.personal.model_dump(exclude_none=True)
        for key, value in personal_dict.items():
            update_fields[f"personal.{key}"] = value

    if data.education is not None:
        edu_dict = data.education.model_dump(exclude_none=True)
        for key, value in edu_dict.items():
            update_fields[f"education.{key}"] = value

    if data.experience is not None:
        exp_dict = data.experience.model_dump(exclude_none=True)
        for key, value in exp_dict.items():
            update_fields[f"experience.{key}"] = value

    if data.skills is not None:
        skills_dict = data.skills.model_dump(exclude_none=True)
        for key, value in skills_dict.items():
            update_fields[f"skills.{key}"] = value

    if data.uploads is not None:
        uploads_dict = data.uploads.model_dump(exclude_none=True)
        for key, value in uploads_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    update_fields[f"uploads.{key}.{sub_key}"] = sub_value
            else:
                update_fields[f"uploads.{key}"] = value

    await db["candidate_profiles"].update_one(
        {"user_id": user_id},
        {"$set": update_fields},
        upsert=True,
    )

    profile = await db["candidate_profiles"].find_one({"user_id": user_id})
    completion = calc_completion(profile)
    await db["candidate_profiles"].update_one(
        {"user_id": user_id},
        {"$set": {"profile_completion": completion, "autosave_status": "saved", "updated_at": now}}
    )
    profile["profile_completion"] = completion
    profile["autosave_status"] = "saved"

    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "data": serialize_profile(profile),
    }


@router.patch("")
async def patch_profile(
    data: CandidateProfileUpdate,
    current_user: dict = Depends(get_current_user),
):
    return await update_profile(data, current_user)
