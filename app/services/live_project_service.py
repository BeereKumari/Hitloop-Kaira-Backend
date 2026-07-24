import os
import json
import asyncio
from typing import List, Dict, Any
from cerebras.cloud.sdk import Cerebras

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

__all__ = [
    "generate_project_description",
    "generate_project_evaluation",
]

def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)

PROJECT_GENERATION_PROMPT = """You are a principal software architect generating a detailed, structured technical project assignment for a candidate.
Based on the candidate's skills, applied role, configured topic, and complexity, generate a comprehensive project scenario.
Return a valid JSON object matching this exact schema:

{
  "title": "Project Title",
  "description": "Detailed project scenario, background context, and implementation requirements.",
  "complexity_description": "Explanation of expected scope based on complexity.",
  "guidelines": [
    "Instruction or architectural guideline candidates must follow"
  ],
  "architectural_constraints": [
    "Constraint candidate must respect (e.g. state management, database schema choice, etc.)"
  ],
  "expected_deliverables": {
    "source_code": "Instructions on what to submit for source code.",
    "documentation": "Instructions on what should be documented.",
    "architecture": "Architecture design expectation.",
    "deployment_url": "Guidelines for deploying the project.",
    "demo_video": "Guidelines for demo video recording."
  }
}

Rules:
- Keep guidelines and constraints specific, actionable, and realistic.
- Return ONLY valid JSON with no markdown fences."""

PROJECT_EVALUATION_PROMPT = """You are a principal software engineer conducting a detailed assessment of a candidate's Live Project submission.
Analyze the project prompt, deliverables metadata, and candidate explanation write-up.
Rate performance across the following core criteria: Functionality, Architecture, System design, Code quality, Deployment, Documentation, Security, and Product thinking.
Return a valid JSON object matching this exact schema:

{
  "overall_score": 85,
  "recommendation": "Hire",
  "overall_summary": "4-5 sentence comprehensive evaluation of the candidate's implementation quality, choice of architecture, trade-offs, and readiness.",
  "score_breakdown": {
    "functionality": {
      "score": 8,
      "assessment": "Detailed assessment of implementation features against requirements."
    },
    "architecture": {
      "score": 9,
      "assessment": "Evaluation of architecture choice, modularity, and correctness."
    },
    "system_design": {
      "score": 8,
      "assessment": "Analysis of system design decisions, data flow, and components scalability."
    },
    "code_quality": {
      "score": 7,
      "assessment": "Review of readability, formatting, code structure, and best practices."
    },
    "deployment": {
      "score": 9,
      "assessment": "Review of the deployment structure, live URL availability, and hosting choices."
    },
    "documentation": {
      "score": 8,
      "assessment": "Review of explanation write-up, setup guidelines, and README clarity."
    },
    "security": {
      "score": 8,
      "assessment": "Assessment of security choices (CORS, data sanitization, auth, secrets handling)."
    },
    "product_thinking": {
      "score": 9,
      "assessment": "Evaluation of UX considerations, UI layout focus, and solving user pain points."
    }
  },
  "strengths": [
    "Demonstrated strength with concrete explanation"
  ],
  "weaknesses": [
    "Weakness or area for improvements with details"
  ],
  "hiring_notes": "Private notes for the hiring panel.",
  "recruiter_insight": "2-3 sentence plain English summary for non-technical recruiters outlining fit for the role."
}

Rules:
- overall_score must be on a 0-100 scale.
- Individual criteria scores inside score_breakdown must be on a 0-10 scale.
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- Return ONLY valid JSON with no markdown fences."""


async def generate_project_description(
    topic: str,
    complexity: str,
    candidate_skills: str,
    applied_role: str,
) -> dict:
    client = get_cerebras_client()
    user_content = (
        f"Topic: {topic}\n"
        f"Complexity: {complexity}\n"
        f"Candidate Skills: {candidate_skills[:200]}\n"
        f"Applied Role: {applied_role}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": PROJECT_GENERATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=3000,
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
    except Exception:
        # Fallback project description
        return {
            "title": f"{topic} Challenge",
            "description": f"Build a prototype system for: {topic} at {complexity} complexity. Ensure clean architecture, complete documentation, and secure endpoints.",
            "complexity_description": f"Designed for a candidate with {complexity} level expertise.",
            "guidelines": [
                "Implement modular components with clear separation of concerns.",
                "Ensure clean state management and clean layout styling.",
                "Provide detailed README instructions for setup."
            ],
            "architectural_constraints": [
                "Choose standard database systems or mock data store.",
                "Implement basic validation and cross-origin resource sharing constraints."
            ],
            "expected_deliverables": {
                "source_code": "Zip code file containing the backend/frontend codebase.",
                "documentation": "System setup manual or README document.",
                "architecture": "Architecture block diagram.",
                "deployment_url": "Active deployment link (e.g. Netlify/Vercel/Render).",
                "demo_video": "Brief video demonstration of working features."
            }
        }


async def generate_project_evaluation(
    prompt: dict,
    explanation: str,
    deliverables: dict,
    candidate_name: str,
    applied_role: str,
) -> dict:
    client = get_cerebras_client()
    
    deliverables_text = ""
    for k, v in deliverables.items():
        if v:
            deliverables_text += f"- {k.replace('_', ' ').capitalize()}: {v.get('url', '')} (filename: {v.get('original_filename', '')})\n"

    user_content = (
        f"Candidate: {candidate_name}\n"
        f"Applied Role: {applied_role}\n\n"
        f"=== Project Assignment ===\n"
        f"Title: {prompt.get('title', 'N/A')}\n"
        f"Description:\n{prompt.get('description', 'N/A')[:1000]}\n\n"
        f"=== Candidate Deliverables ===\n{deliverables_text}\n"
        f"=== Candidate Explanation ===\n{explanation[:4000]}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": PROJECT_EVALUATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=3000,
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
    except Exception:
        return {
            "overall_score": 60,
            "recommendation": "Hold",
            "overall_summary": "AI evaluation failed to parse. Review candidate explanation and deliverables manually.",
            "score_breakdown": {
                "functionality": {"score": 6, "assessment": "Evaluation failed to generate."},
                "architecture": {"score": 6, "assessment": "Evaluation failed to generate."},
                "system_design": {"score": 6, "assessment": "Evaluation failed to generate."},
                "code_quality": {"score": 6, "assessment": "Evaluation failed to generate."},
                "deployment": {"score": 6, "assessment": "Evaluation failed to generate."},
                "documentation": {"score": 6, "assessment": "Evaluation failed to generate."},
                "security": {"score": 6, "assessment": "Evaluation failed to generate."},
                "product_thinking": {"score": 6, "assessment": "Evaluation failed to generate."}
            },
            "strengths": ["Candidate uploaded all required files."],
            "weaknesses": ["Unable to dynamically parse assessment details."],
            "hiring_notes": "Please manually score this candidate attempt.",
            "recruiter_insight": "Manual review of candidate deliverables is required."
        }
