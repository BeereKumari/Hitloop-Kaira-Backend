from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio_url: Optional[str] = None
    applied_role: Optional[str] = None


class EducationInfo(BaseModel):
    highest_degree: Optional[str] = None
    university: Optional[str] = None


class ExperienceInfo(BaseModel):
    years_of_experience: Optional[str] = None
    current_company: Optional[str] = None


class SkillsInfo(BaseModel):
    core_skills: Optional[str] = None


class UploadInfo(BaseModel):
    url: str
    public_id: Optional[str] = None
    filename: Optional[str] = None
    repository_url: Optional[str] = None
    deployment_url: Optional[str] = None


class UploadsInfo(BaseModel):
    resume: Optional[UploadInfo] = None
    projects: Optional[List[UploadInfo]] = None


class CandidateProfileCreate(BaseModel):
    user_id: str
    personal: Optional[PersonalInfo] = None
    education: Optional[EducationInfo] = None
    experience: Optional[ExperienceInfo] = None
    skills: Optional[SkillsInfo] = None
    uploads: Optional[UploadsInfo] = None


class CandidateProfileUpdate(BaseModel):
    personal: Optional[PersonalInfo] = None
    education: Optional[EducationInfo] = None
    experience: Optional[ExperienceInfo] = None
    skills: Optional[SkillsInfo] = None
    uploads: Optional[UploadsInfo] = None


class CandidateProfileResponse(BaseModel):
    id: str
    user_id: str
    personal: PersonalInfo = PersonalInfo()
    education: EducationInfo = EducationInfo()
    experience: ExperienceInfo = ExperienceInfo()
    skills: SkillsInfo = SkillsInfo()
    uploads: UploadsInfo = UploadsInfo()
    profile_completion: int = 0
    autosave_status: str = "saved"
    created_at: str
    updated_at: str
