"""
Kaira Audio Interview Agent — LiveKit Agents v1.6.6
Strict technical interviewer with real-time question progress, answer scoring, and auto-disconnection.

Run alongside FastAPI:
    cd /var/www/hitloop/backend
    ./venv/bin/python -m app.services.livekit_agent dev
"""

import os
import json
import logging
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    inference,
    room_io,
    TurnHandlingOptions,
)
from livekit.plugins import ai_coustics

load_dotenv()

logger = logging.getLogger("kaira-agent")

# ─── Complexity Instructions ──────────────────────────────────────────────────

COMPLEXITY_LABELS = {
    "easy":   "Easy (L1-L2) — foundational concepts",
    "medium": "Medium (L3-L4) — applied knowledge and trade-offs",
    "hard":   "Hard (L5-L6) — advanced system design and optimization",
    "expert": "Expert (L7-L8) — research-level architecture",
}


def build_kaira_instructions(candidate_data: dict, rag_context: str, complexity: str, num_questions: int) -> str:
    """Build Kaira's strict interviewer system prompt."""
    complexity_label = COMPLEXITY_LABELS.get(complexity, COMPLEXITY_LABELS["medium"])
    name     = candidate_data.get("name", "Candidate")
    role     = candidate_data.get("role", "Software Engineer")
    exp      = candidate_data.get("experience", "unspecified")
    company  = candidate_data.get("company", "N/A")
    skills   = candidate_data.get("skills", "general engineering")
    projects = candidate_data.get("project_names", [])
    strengths = candidate_data.get("strengths", [])
    areas    = candidate_data.get("areas_to_explore", [])
    prev_str = candidate_data.get("prev_chat_strengths", [])
    prev_imp = candidate_data.get("prev_chat_improvements", [])

    prev_ctx = ""
    if prev_str or prev_imp:
        prev_ctx = f"""
Previous chat interview showed:
- Strengths: {", ".join(prev_str[:3])}
- Needs improvement: {", ".join(prev_imp[:3])}
Probe the weak areas more deeply.
"""

    return f"""You are Kaira, a strict senior technical interviewer.
You are conducting a {complexity_label} voice interview.

CANDIDATE:
- Name: {name}
- Role: {role}
- Experience: {exp} years at {company}
- Skills: {skills}
- Projects: {", ".join(projects[:4]) if projects else "see resume below"}

RESUME CONTEXT:
{rag_context[:2500] if rag_context else "No resume loaded."}

ANALYSIS:
- Strengths: {", ".join(strengths[:3]) if strengths else "assess from context"}
- Areas to probe: {", ".join(areas[:3]) if areas else "general technical depth"}
{prev_ctx}
CRITICAL QUESTION NUMBERING RULE:
- You MUST start EVERY question with "Question [N] of {num_questions}." — for example: "Question 1 of {num_questions}. Tell me about..."
- This is MANDATORY. Every single question MUST begin with "Question N of {num_questions}."
- Do NOT say "My first question" or "Next question" — always use the exact format.
- Count your questions carefully. If you said "Question 3 of {num_questions}" last, next MUST be "Question 4 of {num_questions}."

INTERVIEW RULES (voice — must be concise and spoken naturally):
1. Ask exactly {num_questions} questions total based ONLY on this candidate's actual experience
2. Responses must be 1-3 SHORT sentences — this is spoken audio not text
3. NEVER explain, hint, or help — you evaluate, never teach
4. If asked to explain: say only "I am here to evaluate, not guide you."
5. If off topic: say only "Let us stay focused." then repeat the question briefly
6. After each answer acknowledge in 1 sentence max before asking the next question
7. NO markdown, NO bullets, NO asterisks, NO emojis — spoken language only
8. Ground every question in their specific projects and tech stack
9. Maximum 2 follow-ups per question, only when genuinely needed
10. Be warm but firm — professional interviewer tone"""


def score_answer_turn(text: str) -> float:
    """Evaluate candidate answer turn quality (1.0 = thorough/correct, 0.5 = partial, 0.0 = poor/unanswered)."""
    text_clean = text.strip().lower()
    words = len(text_clean.split())
    if words < 4 or any(p in text_clean for p in ["dont know", "don't know", "no idea", "not sure", "pass"]):
        return 0.0

    tech_keywords = [
        "system", "architecture", "design", "database", "sql", "nosql", "api", "rest", "graphql",
        "service", "microservice", "cache", "redis", "latency", "scale", "throughput", "async",
        "thread", "process", "memory", "cpu", "performance", "test", "security", "auth", "token",
        "model", "pipeline", "function", "class", "data", "algorithm", "complexity", "index",
        "query", "node", "server", "client", "request", "response", "error", "event", "queue", "react"
    ]
    has_tech_kw = any(kw in text_clean for kw in tech_keywords)

    if words >= 16 and has_tech_kw:
        return 1.0
    elif words >= 8:
        return 0.5
    else:
        return 0.5 if has_tech_kw else 0.0


# ─── Agent Class ──────────────────────────────────────────────────────────────

class KairaInterviewAgent(Agent):
    """Strict technical interviewer with real-time question progress & auto-disconnect."""

    def __init__(
        self,
        instructions: str,
        candidate_data: dict,
        session_id: str,
        user_id: str,
        num_questions: int,
    ):
        super().__init__(instructions=instructions)
        self.candidate_data      = candidate_data
        self.session_id          = session_id
        self.user_id             = user_id
        self.num_questions       = num_questions
        self.questions_answered  = 0    # valid candidate answers counted
        self.questions_asked     = 0    # updated via conversation_item_added (Q N of M)
        self.question_scores     = []
        self._transcript         = []
        self._is_ending          = False
        self._interview_started  = False  # True after Kaira's first message
        self._room               = None   # set after session.start() via set_room()

    def set_room(self, room):
        """Bind room reference for reliable data publishing."""
        self._room = room

    async def _publish_data(self, data_dict: dict):
        """Publish JSON payload over LiveKit data channel."""
        try:
            payload = json.dumps(data_dict).encode("utf-8")
            # Prefer explicitly bound room; fall back to session.room
            room = self._room
            if room is None and hasattr(self, "session") and self.session:
                room = getattr(self.session, "room", None)
            if room:
                await room.local_participant.publish_data(payload)
            else:
                logger.warning(f"_publish_data: no room available — dropping {data_dict.get('type')}")
        except Exception as e:
            logger.warning(f"publish_data warning: {e}")

    def add_transcript_turn(self, speaker: str, text: str):
        """Append to transcript if it's not a duplicate of the last turn."""
        text_clean = text.strip()
        if not text_clean:
            return
        
        # Don't add duplicate Kaira turns
        if speaker == "Kaira" and self._transcript:
            last_turn = self._transcript[-1]
            if last_turn["speaker"] == "Kaira" and last_turn["text"] == text_clean:
                return

        turn = {
            "speaker":   speaker,
            "text":      text_clean,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._transcript.append(turn)
        # Schedule publishing
        asyncio.create_task(self._publish_data({"type": "transcript_turn", "turn": turn}))

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Called every time candidate finishes speaking an answer turn."""
        if self._is_ending:
            return

        text = getattr(new_message, "text_content", None) or str(new_message)
        if not text or not text.strip():
            return

        candidate_text = text.strip()

        # Always record and publish the transcript turn for display
        self.add_transcript_turn("Candidate", candidate_text)

        # ── Answer counting guards ────────────────────────────────────────────
        # Guard: Only count after Kaira has spoken her first message.
        if not self._interview_started:
            logger.debug("Skipping count — Kaira hasn't spoken yet")
            return

        # Clean and lowercase text for check
        clean_text = "".join(c for c in candidate_text.lower() if c.isalnum() or c.isspace()).strip()
        words = clean_text.split()

        # Check if the user is explicitly saying they don't know or passing.
        # These are VALID completed answers — they score 0.0 but still advance the question counter.
        knows_nothing = any(p in clean_text for p in [
            "dont know", "don't know", "no idea", "not sure", "pass",
            "i don't", "i dont", "have no idea", "no clue", "skip",
        ])

        # Common pure-filler/greeting/system test phrases that indicate NO answer was given
        filler_words = {
            "sorry", "hmm", "hm", "ok", "okay", "yes", "no", "hello", "hi", "hey",
            "pardon", "repeat", "what", "test", "testing", "audio", "hear",
            "you", "me", "is", "it", "working",
        }

        # Is it filler/irrelevant? (Don't apply to knows_nothing — those must advance counter)
        is_irrelevant = (
            not knows_nothing and (
                (len(words) < 4) or
                (len(words) > 0 and all(w in filler_words for w in words)) or
                any(phrase in clean_text for phrase in [
                    "can you hear me", "is it working", "test test", "testing testing",
                    "could you repeat", "please repeat", "repeat the question",
                    "what was the question", "what did you say",
                    "i didnt hear", "i didn't hear", "i cant hear", "i can't hear",
                ])
            )
        )

        if is_irrelevant:
            logger.info(f"Filler/irrelevant response detected: {candidate_text!r}. Staying on same question.")
            current_q = max(self.questions_asked, self.questions_answered)
            # Inject system note so LLM re-asks the same question
            new_message.content = [
                f"[System Note: The candidate's response was completely irrelevant filler/noise. "
                f"Do NOT ask a new question. You must stay on the same question. "
                f"Politely ask them to answer Question {current_q if current_q > 0 else 1} of {self.num_questions} again.]"
            ]
            await self._publish_data({
                "type":            "question_progress",
                "question_number": current_q,
                "num_questions":   self.num_questions,
                "score":           None,
                "is_irrelevant":   True,
            })
            return
        # ─────────────────────────────────────────────────────────────────────

        # Valid answer — score it and count it
        score = score_answer_turn(candidate_text)
        self.question_scores.append(score)
        self.questions_answered += 1

        current_q = max(self.questions_asked, self.questions_answered)

        await self._publish_data({
            "type":            "question_progress",
            "question_number": current_q,
            "num_questions":   self.num_questions,
            "score":           score,
            "candidate_text":  candidate_text,
        })

        logger.info(f"Q{current_q}/{self.num_questions} answered — score={score} words={len(words)} knows_nothing={knows_nothing}")

        # Auto-disconnect when all questions answered
        if self.questions_answered >= self.num_questions:
            self._is_ending = True
            name = self.candidate_data.get("name", "there")
            closing_speech = (
                f"Thank you {name} for completing all {self.num_questions} questions of this technical interview. "
                f"I am now concluding our interview session and generating your evaluation analysis. Have a wonderful day!"
            )
            logger.info(f"Interview complete ({self.num_questions}/{self.num_questions}). Concluding call.")

            await self._publish_data({
                "type":            "interview_complete",
                "question_scores": self.question_scores,
                "transcript":      self._transcript,
            })

            try:
                await self.session.say(closing_speech, allow_interruptions=False)
            except Exception:
                pass

            await asyncio.sleep(3)
            try:
                await self.session.aclose()
            except Exception:
                pass

    async def on_agent_speech_committed(self, message) -> None:
        """Called whenever Kaira finishes a speech turn."""
        text = getattr(message, "text", None) or str(message)
        if text and text.strip():
            self.add_transcript_turn("Kaira", text)

    async def on_exit(self) -> None:
        """Save final transcript and scores when session ends."""
        try:
            from app.database.database import db
            await db["interview_sessions"].update_one(
                {"session_id": self.session_id},
                {"$set": {
                    "transcript":      self._transcript,
                    "question_scores": self.question_scores,
                    "status":          "agent_done",
                    "updated_at":      datetime.utcnow().isoformat(),
                }},
            )
            logger.info(f"Session {self.session_id} exited — Saved {len(self._transcript)} turns, {len(self.question_scores)} scores.")
        except Exception as e:
            logger.error(f"Failed to save exit transcript: {e}")


# ─── Context Loader ───────────────────────────────────────────────────────────

async def _load_session_context(room_name: str) -> dict:
    """Load candidate data from MongoDB by room_name."""
    try:
        from app.database.database import db
        session = await db["interview_sessions"].find_one({"room_name": room_name})
        if session:
            return {
                "session_id":     session["session_id"],
                "user_id":        session.get("user_id", "anonymous"),
                "candidate_data": session.get("candidate_data", {}),
                "rag_context":    session.get("rag_context", ""),
                "complexity":     session.get("complexity", "medium"),
                "num_questions":  session.get("num_questions", 8),
            }
    except Exception as e:
        logger.error(f"Context load failed for room {room_name}: {e}")

    return {
        "session_id":    room_name,
        "user_id":       "anonymous",
        "candidate_data": {
            "name": "Candidate", "role": "Software Engineer", "skills": "",
            "strengths": [], "areas_to_explore": [], "project_names": [],
        },
        "rag_context":   "",
        "complexity":    "medium",
        "num_questions": 8,
    }


# ─── Agent Server ─────────────────────────────────────────────────────────────

server = AgentServer()


@server.rtc_session(agent_name="kaira-audio")
async def kaira_session(ctx: agents.JobContext):
    """Entrypoint for LiveKit Cloud job dispatch."""
    room_name = ctx.room.name
    logger.info(f"Job received for room: {room_name}")

    ctx_data = await _load_session_context(room_name)
    candidate_data = ctx_data["candidate_data"]
    complexity     = ctx_data["complexity"]
    num_questions  = ctx_data["num_questions"]

    instructions = build_kaira_instructions(
        candidate_data=candidate_data,
        rag_context=ctx_data["rag_context"],
        complexity=complexity,
        num_questions=num_questions,
    )

    agent = KairaInterviewAgent(
        instructions=instructions,
        candidate_data=candidate_data,
        session_id=ctx_data["session_id"],
        user_id=ctx_data["user_id"],
        num_questions=num_questions,
    )

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
        ),
    )

    # Bind room explicitly so _publish_data always has a reference
    agent.set_room(ctx.room)

    # ── Session-level handler: reliably captures Kaira's text ─────────────────
    # conversation_item_added fires for every LLM output.
    # Use it to: (a) publish Kaira transcript, (b) update questions_asked,
    # (c) set _interview_started so answer counting can begin.
    _kaira_seen_texts: set = set()

    @session.on("conversation_item_added")
    def _on_conversation_item(event):
        item = event.item
        if getattr(item, "role", None) != "assistant":
            return
        text = (
            getattr(item, "text_content", None)
            or getattr(item, "text", None)
            or ""
        ).strip()
        if not text or text in _kaira_seen_texts:
            return
        _kaira_seen_texts.add(text)

        # Mark interview as started so user answers can be counted
        agent._interview_started = True

        # Publish Kaira's transcript turn to frontend
        agent.add_transcript_turn("Kaira", text)

        # Extract "Question N of M" and sync the frontend counter
        from app.services.livekit_video_agent import extract_question_number
        q_num = extract_question_number(text)
        if q_num and q_num > agent.questions_asked:
            agent.questions_asked = q_num
            logger.info(f"[conv_item] Kaira asked Question {q_num}/{num_questions}")
            asyncio.create_task(agent._publish_data({
                "type":            "question_progress",
                "question_number": q_num,
                "num_questions":   num_questions,
                "score":           None,
            }))
    # ──────────────────────────────────────────────────────────────────────────
    name = candidate_data.get("name", "there")
    role = candidate_data.get("role", "this role")
    complexity_label = {"easy": "foundational", "medium": "applied", "hard": "advanced", "expert": "expert"}.get(complexity, "technical")

    opening_instructions = (
        f"Greet {name} warmly by name. Introduce yourself as Kaira. "
        f"State this is a {complexity_label}-level interview for the {role} role covering {num_questions} questions. "
        f"Then immediately ask Question 1 of {num_questions} — a specific technical question from their resume or primary skill. "
        f"Keep the entire opening to 3 spoken sentences. Be direct and professional."
    )

    await session.generate_reply(instructions=opening_instructions)


if __name__ == "__main__":
    agents.cli.run_app(server)
