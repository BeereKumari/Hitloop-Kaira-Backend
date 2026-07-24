import os
import json
import asyncio
from typing import List, Dict, Any
from cerebras.cloud.sdk import Cerebras

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

__all__ = [
    "generate_behaviour_questions",
    "generate_behaviour_evaluation",
]

def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)

QUESTIONS_GENERATION_PROMPT = """You are a senior technical interviewer framing questions to evaluate a candidate's Behavioural qualities.
The assessment must measure the following target traits:
1. Honesty
2. Humility
3. Ownership
4. Work ethic
5. Curiosity
6. Learning mindset
7. Attention to detail
8. Adaptability

Based on the candidate's skills, applied role, configured complexity, and desired number of questions, generate exactly the requested number of questions.
The questions must be a mix of three types:
- "text": Free text response area where they explain their scenario response in detail.
- "mcq": Multiple choice question (single select option).
- "msq": Multiple select question (multiple select options, checklists).

Ensure a balanced mix among "text", "mcq", and "msq" types.
Each question should focus on one of the 8 target traits. Mention which trait is tested by this question in a "category" field.
Frame these questions around realistic software engineering scenarios.

Return a valid JSON array of objects matching this exact schema:
[
  {
    "id": "unique_question_id",
    "type": "text" | "mcq" | "msq",
    "title": "Question Title",
    "question": "The actual question text...",
    "category": "Ownership", // One of: "Honesty" | "Humility" | "Ownership" | "Work ethic" | "Curiosity" | "Learning mindset" | "Attention to detail" | "Adaptability"
    "placeholder": "Guideline or placeholder text for candidate...", // e.g. "Take 4-6 minutes. Structure your answer around what, why, and next steps."
    "options": ["Option A", "Option B", "Option C", "Option D"] // Empty array [] for type "text"
  }
]

Return ONLY valid JSON with no markdown fences."""

EVALUATION_PROMPT = """You are a hiring manager conducting a technical evaluation of a candidate's Behavioural Assessment answers.
Evaluate the responses (including multiple choice/multiple select selections and free text explanations) across these 8 specific traits:
1. Honesty
2. Humility
3. Ownership
4. Work ethic
5. Curiosity
6. Learning mindset
7. Attention to detail
8. Adaptability

IMPORTANT Rules:
- Rate candidate strictly based on evidence from their responses.
- Avoid claiming that this system can scientifically diagnose personality or psychological traits.
- The evaluation must include a diagnostic disclaimer in the "disclaimer" field.

Return a valid JSON object matching this exact schema:
{
  "overall_score": 85,
  "recommendation": "Hire",
  "overall_summary": "4-5 sentence comprehensive evaluation of the candidate's behavioral posture based on evidence from their answers.",
  "disclaimer": "This evaluation is based solely on candidate scenario responses as evidence of behavioral patterns. It is not a psychological or personality diagnostic test.",
  "score_breakdown": {
    "honesty": { "score": 8, "assessment": "Evidence of honesty in admitting bugs or gaps..." },
    "humility": { "score": 9, "assessment": "Evidence of humility in collaboration..." },
    "ownership": { "score": 8, "assessment": "Taking responsibility for project status..." },
    "work_ethic": { "score": 8, "assessment": "Diligence and focus on quality delivery..." },
    "curiosity": { "score": 7, "assessment": "Desire to explore root causes and systems..." },
    "learning_mindset": { "score": 8, "assessment": "Post-mortem analysis and growing from mistakes..." },
    "attention_to_detail": { "score": 9, "assessment": "Careful validation of logic and dependencies..." },
    "adaptability": { "score": 8, "assessment": "Responding to scope shifts and PM deadlines..." }
  },
  "strengths": [
    "Demonstrated strength with explanation"
  ],
  "weaknesses": [
    "Weakness or area for improvement with details"
  ],
  "hiring_notes": "Private notes for the hiring panel.",
  "recruiter_insight": "2-3 sentence summary explaining fit for the team based on behavioral indicators."
}

Rules:
- overall_score must be on a 0-100 scale.
- Individual criteria scores inside score_breakdown must be on a 0-10 scale.
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- Return ONLY valid JSON with no markdown fences."""


async def generate_behaviour_questions(
    candidate_skills: str,
    applied_role: str,
    complexity: str,
    num_questions: int,
) -> list:
    client = get_cerebras_client()
    user_content = (
        f"Candidate Skills: {candidate_skills[:200]}\n"
        f"Applied Role: {applied_role}\n"
        f"Complexity Level: {complexity}\n"
        f"Generate exactly: {num_questions} questions"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": QUESTIONS_GENERATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=2500,
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
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            return data[:num_questions]
    except Exception:
        pass

    # Predefined Fallback Behavioural questions with categories/placeholders matching the image
    fallbacks = [
        {
            "id": "prod_incident",
            "type": "text",
            "category": "Adaptability",
            "title": "Prod incident at 2AM — you're on-call",
            "question": "You are on-call at 2 AM and receive an alert that the main candidate dashboard service is throwing 500 errors. You look at logs and notice a database schema lock. Walk us through your exact steps to handle this under stress.",
            "placeholder": "Structure your explanation around quick containment, root cause identification, and communication.",
            "options": []
        },
        {
            "id": "architecture_conflict",
            "type": "mcq",
            "category": "Humility",
            "title": "Conflict with a senior engineer over architecture",
            "question": "You disagree with a senior engineer on whether to use SQL or NoSQL for a core feature. The debate is delaying development. How do you resolve it?",
            "placeholder": "",
            "options": [
                "Propose a quick proof-of-concept (PoC) with concrete performance indicators to resolve it objectively",
                "Escalate immediately to the engineering manager to make the decision for the team",
                "Concede to the senior colleague's opinion since they have more tenure",
                "Insist on your choice and wait until they compromise"
            ]
        },
        {
            "id": "deadline_slip",
            "type": "text",
            "category": "Ownership",
            "title": "Deadline slip — telling your PM",
            "question": "You committed to shipping a redesign by Friday. On Wednesday, you realize the DB migration needs an extra week. Your PM has already told stakeholders the Friday date. Walk us through what you do next.",
            "placeholder": "Take 4–6 minutes. Structure your answer around what, why, and next steps.",
            "options": []
        },
        {
            "id": "security_bug",
            "type": "mcq",
            "category": "Honesty",
            "title": "Discovering you shipped a security bug",
            "question": "You discover you accidentally pushed test code containing a hardcoded DB credential to the main git branch yesterday. What is your immediate action?",
            "placeholder": "",
            "options": [
                "Force rotate the credential immediately, clear the git history, and report the mistake to security",
                "Delete the credential from the codebase in a new commit and hope nobody checked logs",
                "Wait to see if security scanners flag it before taking action",
                "Deny that you committed it if asked by teammates"
            ]
        },
        {
            "id": "mentorship",
            "type": "msq",
            "category": "Learning mindset",
            "title": "New hire ramp — you're the buddy",
            "question": "Select all options that you believe represent effective Buddy/Mentorship behaviors when onboarding a junior engineer:",
            "placeholder": "",
            "options": [
                "Scheduling regular daily syncs to answer questions and pairing on early commits",
                "Providing a checklist of documentation links and setup steps first",
                "Directing them to read existing codebases without sync sessions",
                "Giving them a small, low-risk bug ticket to gain confidence on day one"
            ]
        },
        {
            "id": "scope_creep",
            "type": "msq",
            "category": "Work ethic",
            "title": "Managing Sprint Scope Creep",
            "question": "Select all ways to handle sudden PM request additions at the end of a sprint without burning out:",
            "placeholder": "",
            "options": [
                "Explain the timeline impact clearly and ask to prioritize for the next sprint",
                "Offer to implement a simplified version of the addition if feasible",
                "Work overnight to cram it in without raising warnings",
                "Suggest swapping out an existing, lower priority task of similar weight"
            ]
        }
    ]
    return fallbacks[:num_questions]


async def generate_behaviour_evaluation(
    questions: list,
    answers: dict,
    candidate_name: str,
    applied_role: str,
) -> dict:
    client = get_cerebras_client()
    
    responses_text = ""
    for q in questions:
        q_id = q.get("id")
        q_title = q.get("title", "Question")
        q_text = q.get("question", "")
        q_type = q.get("type", "text")
        
        raw_ans = answers.get(q_id, "Not answered.")
        if isinstance(raw_ans, list):
            answer_str = ", ".join(raw_ans)
        else:
            answer_str = str(raw_ans)
            
        responses_text += (
            f"=== {q_title} ({q_type}) ===\n"
            f"Question: {q_text}\n"
            f"Candidate Response: {answer_str}\n\n"
        )

    user_content = (
        f"Candidate Name: {candidate_name}\n"
        f"Applied Role: {applied_role}\n\n"
        f"{responses_text}"
    )

    def _call():
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT},
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
            "overall_summary": "AI evaluation failed to parse. Review answers manually.",
            "disclaimer": "This evaluation is based solely on candidate scenario responses as evidence of behavioral patterns. It is not a psychological or personality diagnostic test.",
            "score_breakdown": {
                "honesty": { "score": 6, "assessment": "Failed to parse." },
                "humility": { "score": 6, "assessment": "Failed to parse." },
                "ownership": { "score": 6, "assessment": "Failed to parse." },
                "work_ethic": { "score": 6, "assessment": "Failed to parse." },
                "curiosity": { "score": 6, "assessment": "Failed to parse." },
                "learning_mindset": { "score": 6, "assessment": "Failed to parse." },
                "attention_to_detail": { "score": 6, "assessment": "Failed to parse." },
                "adaptability": { "score": 6, "assessment": "Failed to parse." }
            },
            "strengths": ["Completed all behavioural questions."],
            "weaknesses": ["Unable to dynamically parse assessment details."],
            "hiring_notes": "Please manually score this candidate attempt.",
            "recruiter_insight": "Manual review of candidate answers is required."
        }
