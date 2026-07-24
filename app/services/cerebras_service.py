import os
import json
import asyncio
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")


def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)


# ─── Resume Analysis ────────────────────────────────────────────────────────

RESUME_ANALYSIS_PROMPT = """You are an expert Principal AI System Architect and Technical Recruiter.
Analyze the provided resume and project details for a candidate.

You must perform a rigorous technical analysis and return a valid JSON object matching this exact schema:

{
  "candidate_name": "Extracted Candidate Name or Unknown Candidate",
  "target_role": "Target Role",
  "candidate_summary": "Comprehensive 3-4 sentence synthesis of technical depth, production discipline, research/engineering balance, and overall fit.",
  "role_match_percentage": 92,
  "overall_score": 90,
  "extracted_skills": ["Python", "FastAPI", "MongoDB", "React", "TypeScript"],
  "experience_timeline": [
    {
      "title": "Role Title",
      "company": "Company Name",
      "period": "2023 - Present",
      "highlights": "Key achievement or responsibility"
    }
  ],
  "education": [
    {
      "degree": "Degree Name",
      "institution": "University Name",
      "period": "2018 - 2022",
      "gpa": "9.1/10 or N/A"
    }
  ],
  "strengths": [
    "Specific technical strength backed by evidence from resume"
  ],
  "areas_to_explore": [
    "Potential gap or area requiring clarification in interview"
  ],
  "radar_metrics": [
    {"label": "LLMs", "value": 90},
    {"label": "Systems", "value": 85},
    {"label": "Backend", "value": 88},
    {"label": "Frontend", "value": 75},
    {"label": "Code Quality", "value": 88},
    {"label": "Ops", "value": 80}
  ],
  "projects": [
    {
      "project_name": "Name of Project",
      "problem_solved": "What problem does this project solve? Who is the target user?",
      "technologies_used": ["Python", "FastAPI", "MongoDB", "Docker"],
      "architecture_used": "System design pattern e.g. Microservices, Monolithic API, Async Pipeline.",
      "system_complexity": {
        "score": 8,
        "rationale": "Explanation of system complexity, concurrency, or algorithmic scale."
      },
      "role_relevance": {
        "score": 9,
        "alignment_notes": "How relevant is this project to the applied target role?"
      },
      "code_quality": {
        "score": 8,
        "patterns": "Clean code structure, modularity, type hints, error handling practices.",
        "evidence_snippet": "Relevant quote or code reference demonstrating quality."
      },
      "maintainability": {
        "score": 8,
        "technical_debt": "Evaluation of modularity, test coverage signals, and ease of refactoring."
      },
      "production_readiness": {
        "status": "Production-Ready",
        "checklist": ["Logging and Monitoring", "Dockerized Container", "CI/CD Pipeline", "Error Handling"],
        "verdict": "Detailed verdict on deployment readiness."
      },
      "security_concerns": {
        "risk_level": "Low",
        "findings": ["Finding 1: JWT token validation present", "Finding 2: CORS configuration needs narrowing"],
        "recommendations": ["Recommendation 1 for security hardening"]
      },
      "documentation_quality": {
        "score": 8,
        "summary": "Evaluation of README clarity, API documentation, and setup instructions."
      }
    }
  ],
  "internal_evidence_log": [
    "Evidence Item 1: Verbatim quote or exact metric extracted from resume"
  ]
}

Ensure the output is STRICTLY valid JSON with no markdown code fences.
"""


async def analyze_resume_with_cerebras(resume_text: str, target_role: str = "Senior Software Engineer") -> dict:
    client = get_cerebras_client()
    user_content = f"Target Role: {target_role}\n\nResume Document Content:\n{resume_text[:15000]}"

    def _call_cerebras():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": RESUME_ANALYSIS_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw_json = await loop.run_in_executor(None, _call_cerebras)

    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Cerebras response as JSON: {e}\nRaw: {raw_json[:500]}")


# ─── Interview Turn ──────────────────────────────────────────────────────────

async def run_interview_turn(system_prompt: str, messages: list) -> dict:
    """
    Execute one interview conversation turn via Cerebras.
    Returns structured JSON: { reply, is_followup, depth_level, signal }
    """
    client = get_cerebras_client()

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        # Extract question_score safely (null / 0.0 / 0.5 / 1.0)
        raw_score = parsed.get("question_score")
        question_score = None
        if raw_score is not None:
            try:
                question_score = float(raw_score)
            except (TypeError, ValueError):
                question_score = None
        return {
            "reply": parsed.get("reply", raw),
            "is_followup": bool(parsed.get("is_followup", False)),
            "depth_level": int(parsed.get("depth_level", 3)),
            "signal": str(parsed.get("signal", "")),
            "question_score": question_score,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "reply": raw,
            "is_followup": False,
            "depth_level": 3,
            "signal": "",
            "question_score": None,
        }


# ─── Final Interview Evaluation ──────────────────────────────────────────────

EVALUATION_PROMPT = """You are a senior technical recruiter generating a final interview evaluation report.
Analyze the interview transcript carefully and return a valid JSON object matching this exact schema:

{
  "final_score": 7.5,
  "recommendation": "Strong Hire",
  "overall_summary": "3-4 sentence comprehensive technical and communication evaluation of the candidate.",
  "demonstrated_strengths": [
    "Specific strength with evidence from the interview"
  ],
  "areas_for_improvement": [
    "Specific area that needs development"
  ],
  "topic_evaluations": [
    {
      "topic": "System Design",
      "score": 8,
      "feedback": "Detailed assessment of the candidate's performance on this topic"
    }
  ],
  "communication_score": 8,
  "technical_depth_score": 7,
  "problem_solving_score": 8,
  "hiring_notes": "Private notes for the hiring panel — specific observations, risk flags, or standout moments."
}

recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
All scores are out of 10.
Return ONLY valid JSON with no markdown fences."""


async def generate_interview_evaluation(
    transcript: str,
    candidate_name: str,
    applied_role: str,
    num_questions: int,
    computed_score: float = 0.0,
    question_scores: list = None,
) -> dict:
    """
    Generate a structured final evaluation report from the interview transcript.
    Stored in MongoDB interview_evaluations collection and vectorized.
    """
    client = get_cerebras_client()

    scores_str = ", ".join([str(s) for s in (question_scores or [])]) or "none recorded"
    user_content = (
        f"Candidate: {candidate_name}\n"
        f"Applied Role: {applied_role}\n"
        f"Questions Answered: {num_questions}\n"
        f"Computed Score (DO NOT change this in your response): {computed_score}/10\n"
        f"Per-Question Scores: [{scores_str}] (1.0=complete, 0.5=partial, 0.0=poor)\n\n"
        f"Interview Transcript:\n{transcript[:12000]}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "final_score": 0,
            "recommendation": "Hold",
            "overall_summary": raw[:500] if raw else "Evaluation could not be parsed.",
            "demonstrated_strengths": [],
            "areas_for_improvement": [],
            "topic_evaluations": [],
            "communication_score": 0,
            "technical_depth_score": 0,
            "problem_solving_score": 0,
            "hiring_notes": "Auto-generated evaluation failed. Review transcript manually.",
        }
