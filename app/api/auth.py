import bcrypt
from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from app.database.database import db
from app.models.user import UserCreate, RecruiterCreate, UserLogin, UserResponse, TokenResponse
from app.middleware.auth import create_access_token, get_current_user, serialize_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=dict)
async def register(data: UserCreate):
    existing = await db["users"].find_one({"email": data.email.lower().strip()})
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    password_bytes = data.password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt).decode("utf-8")

    now = datetime.utcnow().isoformat()
    user_doc = {
        "full_name": data.full_name.strip(),
        "email": data.email.lower().strip(),
        "password": hashed_password,
        "phone": data.phone.strip() if data.phone else None,
        "role": "candidate",
        "account_status": "active",
        "email_verified": False,
        "last_login": now,
        "created_at": now,
        "updated_at": now,
    }

    result = await db["users"].insert_one(user_doc)
    user_id = str(result.inserted_id)

    profile_doc = {
        "user_id": user_id,
        "personal": {
            "full_name": data.full_name.strip(),
            "email": data.email.lower().strip(),
            "phone": data.phone.strip() if data.phone else None,
            "location": None,
            "linkedin": None,
            "github": None,
            "portfolio_url": None,
            "applied_role": None,
        },
        "education": {
            "highest_degree": None,
            "university": None,
        },
        "experience": {
            "years_of_experience": None,
            "current_company": None,
        },
        "skills": {
            "core_skills": None,
        },
        "uploads": {},
        "profile_completion": 0,
        "autosave_status": "saved",
        "created_at": now,
        "updated_at": now,
    }
    await db["candidate_profiles"].insert_one(profile_doc)

    access_token = create_access_token(user_id, remember_me=False)
    user_doc["_id"] = result.inserted_id

    return {
        "status": "success",
        "message": "Account created successfully.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_user(user_doc),
    }


@router.post("/register-recruiter", response_model=dict)
async def register_recruiter(data: RecruiterCreate):
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    existing = await db["users"].find_one({"email": data.email.lower().strip()})
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    password_bytes = data.password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt).decode("utf-8")

    full_name = f"{data.first_name.strip()} {data.last_name.strip()}"
    now = datetime.utcnow().isoformat()
    user_doc = {
        "full_name": full_name,
        "email": data.email.lower().strip(),
        "password": hashed_password,
        "phone": None,
        "role": "recruiter",
        "account_status": "active",
        "email_verified": False,
        "last_login": now,
        "created_at": now,
        "updated_at": now,
    }

    result = await db["users"].insert_one(user_doc)
    user_id = str(result.inserted_id)

    recruiter_doc = {
        "user_id": user_id,
        "first_name": data.first_name.strip(),
        "last_name": data.last_name.strip(),
        "company_name": data.company_name.strip(),
        "email": data.email.lower().strip(),
        "company_website": data.company_website.strip() if data.company_website else None,
        "created_at": now,
        "updated_at": now,
    }
    await db["recruiter_profiles"].insert_one(recruiter_doc)

    access_token = create_access_token(user_id, remember_me=False)
    user_doc["_id"] = result.inserted_id

    return {
        "status": "success",
        "message": "Recruiter account created successfully.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_user(user_doc),
    }


@router.post("/login", response_model=dict)
async def login(data: UserLogin):
    user = await db["users"].find_one({"email": data.email.lower().strip()})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if user.get("account_status") == "disabled":
        raise HTTPException(status_code=403, detail="Account has been disabled. Contact support.")

    password_bytes = data.password.encode("utf-8")
    stored_hash = user["password"].encode("utf-8")
    if not bcrypt.checkpw(password_bytes, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    now = datetime.utcnow().isoformat()
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": now, "updated_at": now}}
    )

    access_token = create_access_token(str(user["_id"]), remember_me=data.remember_me or False)
    user["last_login"] = now

    return {
        "status": "success",
        "message": "Logged in successfully.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_user(user),
    }


@router.post("/logout")
async def logout():
    return {"status": "success", "message": "Logged out successfully."}


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "status": "success",
        "user": serialize_user(current_user),
    }
