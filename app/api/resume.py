import os
import shutil
import tempfile
import urllib.request
from datetime import datetime
from typing import Optional
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.database.database import db
from app.services.pdf_service import extract_text_from_file
from app.services.cerebras_service import analyze_resume_with_cerebras
from app.services.vector_service import store_resume_vectors
from app.middleware.auth import get_current_user_optional

router = APIRouter(prefix="/api/resume", tags=["Resume Analysis"])

# Use /tmp for temporary file storage — writable on all cloud platforms (Render, Railway, etc.)
# Files are only used momentarily for text extraction, then deleted.
UPLOADS_DIR = os.path.join(tempfile.gettempdir(), "hitloop_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)




def serialize_mongo_doc(doc: dict) -> dict:
    """Helper to convert Mongo _id ObjectId to string."""
    if not doc:
        return doc
    doc["id"] = str(doc["_id"])
    if "_id" in doc:
        del doc["_id"]
    return doc


@router.post("/upload")
async def upload_and_analyze_resume(
    file: Optional[UploadFile] = File(None),
    cloudinary_url: Optional[str] = Form(None),
    cloudinary_public_id: Optional[str] = Form(None),
    target_role: str = Form("Senior AI Engineer"),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    temp_file_path = None
    try:
        user_id = str(current_user["_id"]) if current_user else "anonymous"
        filename = file.filename if file else "uploaded_resume.pdf"
        temp_file_path = os.path.join(UPLOADS_DIR, f"{user_id}_{filename}")

        if file:
            # Save uploaded file temporarily for text extraction
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif cloudinary_url:
            # Download file from Cloudinary URL for text extraction
            try:
                req = urllib.request.Request(
                    cloudinary_url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req) as response, open(temp_file_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as dl_err:
                raise HTTPException(status_code=400, detail=f"Failed to fetch document from Cloudinary URL: {str(dl_err)}")
        else:
            raise HTTPException(status_code=400, detail="No resume file or Cloudinary URL provided.")

        # Extract text from PDF / Document
        extracted_text = extract_text_from_file(temp_file_path)
        if not extracted_text or len(extracted_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Unable to extract sufficient text content from uploaded file.")

        # Call Cerebras LLM Analysis
        analysis_data = await analyze_resume_with_cerebras(extracted_text, target_role=target_role)

        # Generate & Store Vector Embeddings for future interview RAG
        vector_count = await store_resume_vectors(user_id=user_id, resume_text=extracted_text, analysis=analysis_data)

        # Build analysis record (no local file_path — use cloudinary_url for access)
        now_str = datetime.utcnow().isoformat()
        record = {
            "user_id": user_id,
            "filename": filename,
            "cloudinary_url": cloudinary_url,
            "cloudinary_public_id": cloudinary_public_id,
            "target_role": target_role,
            "vector_count": vector_count,
            "created_at": now_str,
            "analysis": analysis_data
        }

        # Insert / Update in MongoDB resume_analyses
        if user_id != "anonymous":
            await db["resume_analyses"].delete_many({"user_id": user_id})
        result = await db["resume_analyses"].insert_one(record)
        record["id"] = str(result.inserted_id)
        if "_id" in record:
            del record["_id"]

        # Update candidate profile if user is logged in
        if user_id != "anonymous":
            upload_info = {
                "url": cloudinary_url or "",
                "public_id": cloudinary_public_id or "",
                "filename": filename,
            }
            profile_update = {
                "uploads.resume": upload_info,
                "personal.applied_role": target_role,
                "updated_at": now_str,
            }
            await db["candidate_profiles"].update_one(
                {"user_id": user_id},
                {"$set": profile_update},
                upsert=True
            )

        return {
            "status": "success",
            "message": "Resume analyzed and vectors stored successfully",
            "analysis_id": record["id"],
            "vector_count": vector_count,
            "data": record
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume processing failed: {str(e)}")
    finally:
        # Always clean up temp file to avoid filling /tmp
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass




@router.get("/latest")
async def get_latest_resume_analysis(current_user: Optional[dict] = Depends(get_current_user_optional)):
    try:
        user_id = str(current_user["_id"]) if current_user else None

        doc = None
        if user_id:
            doc = await db["resume_analyses"].find_one({"user_id": user_id}, sort=[("created_at", -1)])

        if not doc:
            doc = await db["resume_analyses"].find_one(sort=[("created_at", -1)])

        if not doc:
            return {"status": "empty", "data": None, "recruiter_decision": None}

        # Also fetch the recruiter's decision so the candidate UI can show status
        recruiter_decision = None
        if user_id:
            rd = await db["recruiter_decisions"].find_one({"candidate_id": user_id})
            if rd:
                recruiter_decision = {
                    "action": rd.get("action"),
                    "analysis_completed": rd.get("analysis_completed", False),
                    "notes": rd.get("notes"),
                    "updated_at": rd.get("updated_at"),
                }

        serialized = serialize_mongo_doc(doc)
        return {
            "status": "success",
            "data": serialized,
            "recruiter_decision": recruiter_decision,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest analysis: {str(e)}")




@router.get("/{analysis_id}")
async def get_resume_analysis_by_id(analysis_id: str):
    try:
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(status_code=400, detail="Invalid Analysis ID format")

        doc = await db["resume_analyses"].find_one({"_id": ObjectId(analysis_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Resume analysis not found")

        return {"status": "success", "data": serialize_mongo_doc(doc)}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")
