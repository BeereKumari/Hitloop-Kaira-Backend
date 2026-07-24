from fastapi import APIRouter
from app.database.database import db

router = APIRouter()

@router.get("/test-db")
async def test_db():
    try:
        collections = await db.list_collection_names()
        return {
            "status": "✅ Database Connected Successfully",
            "database": db.name,
            "collections": collections
        }
    except Exception as e:
        return {
            "status": "❌ Database Connection Failed",
            "error": str(e)
        }