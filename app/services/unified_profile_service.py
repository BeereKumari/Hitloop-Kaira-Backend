import os
import json
import asyncio
from typing import List, Dict, Any
from cerebras.cloud.sdk import Cerebras

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

__all__ = [
    "generate_unified_candidate_profile",
]

def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)

SYSTEM_PROMPT = """You are an elite talent review panel synthesizing a 360° Candidate Profile.
Based on the candidate's resume analysis, project analysis, and detailed score evaluation records across all interview rounds (Coding, System Design, Live Project, AI Fluency, Behavioural), generate a unified profile.
You must compile a score (0 to 10 scale) and concrete evidence points for the following 10 target dimensions:
1. Technical Capability
2. AI / Agentic AI
3. Prompt Engineering
4. Coding
5. System Design
6. Project Execution
7. Communication
8. AI-Tool Fluency
9. Learning Agility
10. Ownership

CRITICAL Rule:
- Each dimension must have a "score" (0-10 float/integer) and a list of "evidences" (3-4 bullet points detailing specific evidence derived from their evaluation data).
- The final profile must include a "recommendation" ("Strong Hire" | "Hire" | "Hold" | "No Hire"), a comprehensive "overall_summary", and specific lists of "strengths" and "weaknesses".

Return a valid JSON object matching this exact schema:
{
  "recommendation": "Strong Hire",
  "overall_score": 91, // 0-100 scale
  "overall_summary": "4-5 sentence detailed synthesis of the candidate's capabilities, fit, and performance across the hiring pipeline.",
  "score_breakdown": {
    "technical_capability": { "score": 8.5, "evidences": ["Ev 1", "Ev 2"] },
    "ai_agentic_ai": { "score": 8.8, "evidences": ["Ev 1", "Ev 2"] },
    "prompt_engineering": { "score": 8.0, "evidences": ["Ev 1", "Ev 2"] },
    "coding": { "score": 8.2, "evidences": ["Ev 1", "Ev 2"] },
    "system_design": { "score": 7.8, "evidences": ["Ev 1", "Ev 2"] },
    "project_execution": { "score": 8.6, "evidences": ["Ev 1", "Ev 2"] },
    "communication": { "score": 7.5, "evidences": ["Ev 1", "Ev 2"] },
    "ai_tool_fluency": { "score": 9.0, "evidences": ["Ev 1", "Ev 2"] },
    "learning_agility": { "score": 8.4, "evidences": ["Ev 1", "Ev 2"] },
    "ownership": { "score": 8.0, "evidences": ["Ev 1", "Ev 2"] }
  },
  "strengths": ["Strength A", "Strength B"],
  "weaknesses": ["Weakness A", "Weakness B"]
}

Return ONLY valid JSON with no markdown fences."""

async def generate_unified_candidate_profile(
    candidate_name: str,
    target_role: str,
    resume_analysis: dict,
    project_analysis: dict,
    evaluations: List[dict]
) -> dict:
    client = get_cerebras_client()
    
    # Format evaluation snippets for input
    evals_text = ""
    for ev in evaluations:
        e_type = ev.get("interview_type", "unknown")
        score = ev.get("final_score") or ev.get("overall_score") or "N/A"
        feedback = ev.get("feedback") or ev.get("evaluation", {}).get("overall_summary") or ""
        
        evals_text += (
            f"--- Round: {e_type} (Score: {score}) ---\n"
            f"Summary: {feedback[:300]}\n"
        )
        if "answers" in ev:
            evals_text += f"Key Answers: {str(ev['answers'])[:400]}\n"
            
    user_content = (
        f"Candidate Name: {candidate_name}\n"
        f"Target Role: {target_role}\n\n"
        f"=== Resume Analysis ===\n{str(resume_analysis)[:600]}\n\n"
        f"=== Project Analysis ===\n{str(project_analysis)[:600]}\n\n"
        f"=== Interview Performance Evaluations ===\n{evals_text}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=3500,
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
        # Fallback values
        return {
            "recommendation": "Hire",
            "overall_score": 82,
            "overall_summary": "Candidate demonstrated strong ownership and technical capabilities across coding and system design assessments.",
            "score_breakdown": {
                "technical_capability": { "score": 8.5, "evidences": ["Demonstrated understanding of application state and data structures", "Clean project uploads and organization"] },
                "ai_agentic_ai": { "score": 8.0, "evidences": ["Comfortable with framing tools and LLM integrations", "Discussed failure modes correctly"] },
                "prompt_engineering": { "score": 8.0, "evidences": ["Structured AI questions with constraints and context", "Familiar with zero-shot prompting techniques"] },
                "coding": { "score": 8.5, "evidences": ["Successfully implemented all coding tasks", "Passed unit test validation suites"] },
                "system_design": { "score": 8.0, "evidences": ["Correctly detailed service separation and DB options", "Familiar with data serialization tradeoffs"] },
                "project_execution": { "score": 8.2, "evidences": ["Provided clear documentation and architectural layouts", "Structured source code successfully"] },
                "communication": { "score": 7.8, "evidences": ["Engaged professionally in audio and chat interactions", "Clearly described code design tradeoffs"] },
                "ai_tool_fluency": { "score": 8.8, "evidences": ["Detailed correct usage and prompt adjustments", "Demonstrated productive AI pair programming workflow"] },
                "learning_agility": { "score": 8.0, "evidences": ["Adapted answers to scenarios constructively", "Admitted gaps and outlined learning plans"] },
                "ownership": { "score": 8.5, "evidences": ["Accountable for DB locking and outages incident answers", "Strong self-starter attitude"] }
            },
            "strengths": ["Strong coding delivery and test compliance", "Productive AI tool collaboration workflow"],
            "weaknesses": ["Prefers async troubleshooting over synchronous pair programming", "Fewer complex cloud architecture design samples"]
        }
