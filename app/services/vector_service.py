import re
import math
from datetime import datetime
from typing import List, Dict, Any
from app.database.database import db

COLLECTION_NAME = "resume_vectors"


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def generate_simple_embedding(text: str, vector_dim: int = 128) -> List[float]:
    """
    Generates a deterministic 128-dimensional dense vector embedding from text using hashing.
    Used for vector indexing and cosine similarity RAG search in MongoDB.
    """
    tokens = _tokenize(text)
    vector = [0.0] * vector_dim
    if not tokens:
        return vector

    for idx, token in enumerate(tokens):
        hash_val = hash(token)
        dim = abs(hash_val) % vector_dim
        weight = 1.0 / (1.0 + math.log(idx + 1))
        sign = 1.0 if hash_val > 0 else -1.0
        vector[dim] += sign * weight

    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude > 0:
        vector = [round(v / magnitude, 6) for v in vector]

    return vector


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Splits text into overlapping text chunks for retrieval."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{p}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    final_chunks = []
    for c in chunks:
        if len(c) <= chunk_size:
            final_chunks.append(c)
        else:
            start = 0
            while start < len(c):
                end = start + chunk_size
                final_chunks.append(c[start:end])
                start += chunk_size - overlap

    return final_chunks


async def retrieve_relevant_chunks(
    user_id: str, query: str, top_k: int = 6, collection: str = COLLECTION_NAME
) -> List[Dict]:
    """
    Cosine similarity RAG retrieval over stored resume/interview vectors.
    Falls back to 'anonymous' vectors when no user-specific ones exist.
    """
    query_embedding = generate_simple_embedding(query)

    # Try user-specific first, then fall back to anonymous
    user_vectors = await db[collection].find(
        {"user_id": user_id}
    ).to_list(None)

    if not user_vectors:
        user_vectors = await db[collection].find(
            {"user_id": "anonymous"}
        ).to_list(None)

    if not user_vectors:
        return []

    scored = [
        (_cosine_similarity(query_embedding, v.get("embedding", [])), v)
        for v in user_vectors
        if v.get("embedding")
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [v for _, v in scored[:top_k]]


async def store_resume_vectors(user_id: str, resume_text: str, analysis: Dict[str, Any]) -> int:
    """
    Chunks the candidate's resume text and AI analysis, generates vector embeddings,
    and stores them in MongoDB's `resume_vectors` collection for RAG retrieval during interviews.
    Always stores under the provided user_id (not 'anonymous').
    """
    now = datetime.utcnow().isoformat()

    # Clear previous vectors for this user
    await db[COLLECTION_NAME].delete_many({"user_id": user_id})

    vector_docs = []

    # 1. Chunk & vectorify raw resume text
    resume_chunks = chunk_text(resume_text, chunk_size=400, overlap=80)
    for idx, chunk in enumerate(resume_chunks):
        embedding = generate_simple_embedding(chunk)
        vector_docs.append({
            "user_id": user_id,
            "chunk_id": f"resume_raw_{idx}",
            "category": "raw_resume",
            "content": chunk,
            "embedding": embedding,
            "created_at": now,
        })

    # 2. Vectorify candidate summary & strengths
    summary_text = (
        f"Candidate Summary: {analysis.get('candidate_summary', '')} "
        f"Strengths: {' '.join(analysis.get('strengths', []))} "
        f"Skills: {' '.join(analysis.get('extracted_skills', []))}"
    )
    if summary_text.strip():
        vector_docs.append({
            "user_id": user_id,
            "chunk_id": "analysis_summary",
            "category": "ai_summary",
            "content": summary_text,
            "embedding": generate_simple_embedding(summary_text),
            "created_at": now,
        })

    # 3. Vectorify project audits
    projects = analysis.get("projects") or []
    for idx, proj in enumerate(projects):
        proj_str = (
            f"Project: {proj.get('project_name', '')}. "
            f"Problem Solved: {proj.get('problem_solved', '')}. "
            f"Technologies: {', '.join(proj.get('technologies_used', []))}. "
            f"Architecture: {proj.get('architecture_used', '')}. "
            f"Complexity Rationale: {proj.get('system_complexity', {}).get('rationale', '')}. "
            f"Code Evidence: {proj.get('code_quality', {}).get('evidence_snippet', '')}"
        )
        vector_docs.append({
            "user_id": user_id,
            "chunk_id": f"project_{idx}",
            "category": "project_audit",
            "project_name": proj.get("project_name", ""),
            "content": proj_str,
            "embedding": generate_simple_embedding(proj_str),
            "created_at": now,
        })

    # 4. Vectorify areas_to_explore for targeted interview coverage
    areas = analysis.get("areas_to_explore", [])
    if areas:
        areas_text = f"Areas to Explore in Interview: {'. '.join(areas)}"
        vector_docs.append({
            "user_id": user_id,
            "chunk_id": "areas_to_explore",
            "category": "interview_targets",
            "content": areas_text,
            "embedding": generate_simple_embedding(areas_text),
            "created_at": now,
        })

    if vector_docs:
        await db[COLLECTION_NAME].insert_many(vector_docs)

    return len(vector_docs)


async def store_interview_vectors(
    user_id: str, session_id: str, evaluation: Dict[str, Any]
) -> int:
    """
    Stores the interview evaluation as vectors in `interview_vectors` collection.
    Used for future interviews and recruiter RAG queries.
    """
    now = datetime.utcnow().isoformat()
    vector_docs = []

    # Overall evaluation vector
    eval_text = (
        f"Interview Evaluation. "
        f"Final Score: {evaluation.get('final_score', 0)}/10. "
        f"Recommendation: {evaluation.get('recommendation', '')}. "
        f"Summary: {evaluation.get('overall_summary', '')}. "
        f"Strengths: {', '.join(evaluation.get('demonstrated_strengths', []))}. "
        f"Areas for Improvement: {', '.join(evaluation.get('areas_for_improvement', []))}."
    )
    vector_docs.append({
        "user_id": user_id,
        "session_id": session_id,
        "chunk_id": f"interview_eval_{session_id[:8]}",
        "category": "interview_evaluation",
        "content": eval_text,
        "embedding": generate_simple_embedding(eval_text),
        "created_at": now,
    })

    # Per-topic evaluation vectors
    for topic_eval in evaluation.get("topic_evaluations", []):
        topic = topic_eval.get("topic", "")
        topic_text = (
            f"Interview Topic: {topic}. "
            f"Score: {topic_eval.get('score', 0)}/10. "
            f"Feedback: {topic_eval.get('feedback', '')}"
        )
        vector_docs.append({
            "user_id": user_id,
            "session_id": session_id,
            "chunk_id": f"interview_topic_{session_id[:8]}_{topic[:20]}",
            "category": "interview_topic",
            "content": topic_text,
            "embedding": generate_simple_embedding(topic_text),
            "created_at": now,
        })

    if vector_docs:
        await db["interview_vectors"].insert_many(vector_docs)

    return len(vector_docs)
