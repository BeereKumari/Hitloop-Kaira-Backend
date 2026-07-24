import os
import json
import asyncio
from typing import List, Dict, Any
from cerebras.cloud.sdk import Cerebras

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

__all__ = [
    "generate_fluency_questions",
    "generate_fluency_evaluation",
]

def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from environment variables.")
    return Cerebras(api_key=api_key)

QUESTIONS_GENERATION_PROMPT = """You are a senior technical interviewer framing questions to evaluate a candidate's AI Tool Fluency.
The assessment must measure how effectively they leverage AI tools to improve productivity, and how rigorously they validate AI output.
Based on the candidate's skills, applied role, configured complexity, and desired number of questions, generate exactly the requested number of questions.
The questions must be a mix of three types:
- "text": Free text response area where they explain their workflow or caught mistakes in detail.
- "mcq": Multiple choice question (single select option).
- "msq": Multiple select question (multiple select options, checklists).

Ensure roughly equal distribution among "text", "mcq", and "msq" types.
Frame these questions around:
1. AI tools used & Selection Rationale
2. Implementation pattern / How they were used
3. Validation strategy / How output was verified
4. Error detection / Caught mistakes & self-correction
5. Acceleration value / Time saved & productivity improvements

Return a valid JSON array of objects matching this exact schema:
[
  {
    "id": "unique_question_id",
    "type": "text", // "text" | "mcq" | "msq"
    "title": "Question Title",
    "question": "The actual question text...",
    "options": ["Option A", "Option B", "Option C", "Option D"] // Empty array [] for type "text"
  }
]

Return ONLY valid JSON with no markdown fences."""

EVALUATION_PROMPT = """You are a principal software engineer conducting a technical evaluation of a candidate's AI Tool Fluency answers.
Analyze their responses (including multiple choice/multiple select selections and free text explanations) to see if they understand the risks of AI tools (hallucinations, security, edge cases) and if they validate/inspect outputs rigorously.
Return a valid JSON object matching this exact schema:

{
  "overall_score": 82,
  "recommendation": "Hire",
  "overall_summary": "4-5 sentence comprehensive evaluation of the candidate's fluency with AI tools, prompting patterns, validation mechanisms, and engineering rigor.",
  "score_breakdown": {
    "prompting_effectiveness": {
      "score": 8,
      "assessment": "Evaluation of how well they instruct/query AI tools to get optimal results."
    },
    "validation_rigor": {
      "score": 9,
      "assessment": "Detailed review of their strategies to inspect, compile, test, or manually review AI output."
    },
    "error_detection": {
      "score": 8,
      "assessment": "How well they identify, trace, and self-correct AI-generated logical bugs or security vulnerabilities."
    },
    "acceleration_leverage": {
      "score": 8,
      "assessment": "Assessment of how effectively they leverage tools to automate boilerplate and save development hours."
    }
  },
  "strengths": [
    "Demonstrated strength with explanation"
  ],
  "weaknesses": [
    "Weakness or area for improvement with details"
  ],
  "hiring_notes": "Private notes for the hiring panel.",
  "recruiter_insight": "2-3 sentence summary explaining to non-technical recruiters how effectively this candidate uses AI to accelerate shipping."
}

Rules:
- overall_score must be on a 0-100 scale.
- Individual criteria scores inside score_breakdown must be on a 0-10 scale.
- recommendation must be one of: "Strong Hire", "Hire", "Hold", "No Hire"
- Return ONLY valid JSON with no markdown fences."""


async def generate_fluency_questions(
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

    # Dynamic Fallback questions based on required count
    fallbacks = [
        {
            "id": "tools_used",
            "type": "mcq",
            "title": "Weekly AI Tool Selection",
            "question": "Which of the following AI tools do you rely on most heavily to automate code construction or scaffolding during your weekly routine?",
            "options": [
                "Cursor (IDE) and Claude 3.5 Sonnet",
                "GitHub Copilot (Auto-complete) and ChatGPT",
                "Command-line helpers (Aider/supermaven)",
                "I do not use AI tools for daily coding scaffolding"
            ]
        },
        {
            "id": "how_used",
            "type": "msq",
            "title": "Application Purposes",
            "question": "Select all options that accurately represent how you leverage AI code generators in your daily work tasks:",
            "options": [
                "Scaffolding boilerplates & initial component layouts",
                "Explaining foreign libraries or complex legacy logic",
                "Generating automated test suites & test mock files",
                "Refactoring, sorting imports, or sorting styles"
            ]
        },
        {
            "id": "validation",
            "type": "text",
            "title": "Verification & Validation Method",
            "question": "Explain in detail your verification methodology. How do you validate that AI generated code is secure, scalable, and does not contain hidden logical errors?",
            "options": []
        },
        {
            "id": "error_detection",
            "type": "text",
            "title": "Catching Halucinations & Mistakes",
            "question": "Describe a real situation where an AI tool generated a bug or hallucinated a library method. How did you catch the error, and what actions did you take to self-correct the code?",
            "options": []
        },
        {
            "id": "acceleration",
            "type": "mcq",
            "title": "Productivity Acceleration Value",
            "question": "What is the primary way AI acceleration manifests in your engineering speed and final output quality?",
            "options": [
                "Reduces cognitive load of writing repetitive boilerplate, freeing time for architecture design",
                "Speeds up typing speed but requires equal time for debugging and validation cycles",
                "Mostly helps with syntax references, acting as an interactive documentation tool",
                "Does not significantly affect my daily delivery speeds"
            ]
        },
        {
            "id": "ethics_security",
            "type": "msq",
            "title": "Data Privacy & Security Precautions",
            "question": "Which actions do you systematically take to prevent leaking internal database schemas or secrets to public AI servers?",
            "options": [
                "Sanitizing files to strip API keys, secrets, and private candidate database strings",
                "Relying on enterprise agreements that explicitly restrict training on code uploads",
                "Running local self-hosted models (Ollama/llama-3) for proprietary workflows",
                "I do not take specific precautions regarding data sharing"
            ]
        }
    ]
    return fallbacks[:num_questions]


async def generate_fluency_evaluation(
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
            "score_breakdown": {
                "prompting_effectiveness": {"score": 6, "assessment": "Evaluation failed to generate."},
                "validation_rigor": {"score": 6, "assessment": "Evaluation failed to generate."},
                "error_detection": {"score": 6, "assessment": "Evaluation failed to generate."},
                "acceleration_leverage": {"score": 6, "assessment": "Evaluation failed to generate."}
              },
            "strengths": ["Completed all fluency questions."],
            "weaknesses": ["Unable to dynamically parse assessment details."],
            "hiring_notes": "Please manually score this candidate attempt.",
            "recruiter_insight": "Manual review of candidate answers is required."
        }
