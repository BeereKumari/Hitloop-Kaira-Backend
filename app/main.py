from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.test import router as test_router
from app.api.resume import router as resume_router
from app.api.auth import router as auth_router
from app.api.profile import router as profile_router
from app.api.interview import router as interview_router
from app.api.coding import router as coding_router
from app.api.audio_interview import router as audio_interview_router
from app.api.video_interview import router as video_interview_router
from app.api.recruiter import router as recruiter_router
from app.api.live_project import router as live_project_router
from app.api.ai_fluency import router as ai_fluency_router
from app.api.behaviour import router as behaviour_router
from app.api.unified_profile import router as unified_profile_router

app = FastAPI(title="Hitloop Hiring Platform API", version="1.0.0")

ALLOWED_ORIGINS = [
    "http://localhost:8085",
    "http://localhost:8086",
    "http://localhost:8002",
    "http://127.0.0.1:8085",
    "http://127.0.0.1:8086",
    "http://127.0.0.1:8002",
    "http://173.249.55.150:8085",
    "http://173.249.55.150:8086",
    "http://173.249.55.150:8002",
    # ── Production domains ──────────────────────────────────
    # Add your actual deployed frontend URL below after deployment
    # e.g. "https://hitloop.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=(
        r"http://.*:(8085|8086|8002|3000|5173)"                 # local dev
        r"|https://.*\.vercel\.app"                              # Vercel previews & prod
        r"|https://.*\.onrender\.com"                           # Render
        r"|https://.*\.railway\.app"                            # Railway
        r"|https://.*\.up\.railway\.app"                        # Railway alt
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.include_router(test_router)
app.include_router(resume_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(interview_router)
app.include_router(coding_router)
app.include_router(audio_interview_router)
app.include_router(video_interview_router)
app.include_router(recruiter_router)
app.include_router(live_project_router)
app.include_router(ai_fluency_router)
app.include_router(behaviour_router)
app.include_router(unified_profile_router)