"""
Kaira Video Interview Agent — LiveKit Agents v1.6.6 + Beyond Presence Avatar
Strictly follows Avatar-Reference/main.py pattern for bey avatar initialization.

Key patterns from reference:
- Avatar started BEFORE session.start()
- When avatar active: audio_enabled=False (avoids double audio)
- session.start() uses room_input_options/room_output_options (NOT room_options)
- _publish_data uses session.room (set after session.start)
"""

import os
import re
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
    TurnHandlingOptions,
    RoomInputOptions,
    RoomOutputOptions,
)
from livekit.plugins import ai_coustics, bey

load_dotenv()

logger = logging.getLogger("kaira-video-agent")

BEY_API_KEY      = os.getenv("BEY_API_KEY", "")
BEY_AVATAR_ID    = os.getenv("BEY_AVATAR_ID", "")
LIVEKIT_URL      = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY  = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# ─── Complexity Labels ────────────────────────────────────────────────────────

COMPLEXITY_LABELS = {
    "easy":   "Easy (L1-L2) — foundational concepts",
    "medium": "Medium (L3-L4) — applied knowledge and trade-offs",
    "hard":   "Hard (L5-L6) — advanced system design and optimization",
    "expert": "Expert (L7-L8) — research-level architecture",
}

# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_kaira_video_instructions(
    candidate_data: dict,
    rag_context: str,
    complexity: str,
    num_questions: int,
) -> str:
    complexity_label = COMPLEXITY_LABELS.get(complexity, COMPLEXITY_LABELS["medium"])
    name      = candidate_data.get("name", "Candidate")
    role      = candidate_data.get("role", "Software Engineer")
    exp       = candidate_data.get("experience", "unspecified")
    company   = candidate_data.get("company", "N/A")
    skills    = candidate_data.get("skills", "general engineering")
    projects  = candidate_data.get("project_names", [])
    strengths = candidate_data.get("strengths", [])
    areas     = candidate_data.get("areas_to_explore", [])
    prev_str  = candidate_data.get("prev_chat_strengths", [])
    prev_imp  = candidate_data.get("prev_chat_improvements", [])
    audio_str = candidate_data.get("prev_audio_strengths", [])
    audio_imp = candidate_data.get("prev_audio_improvements", [])

    prev_ctx = ""
    if prev_str or prev_imp:
        prev_ctx += f"\nPrevious chat interview — Strengths: {', '.join(prev_str[:3])}; Needs improvement: {', '.join(prev_imp[:3])}\n"
    if audio_str or audio_imp:
        prev_ctx += f"\nPrevious audio interview — Strengths: {', '.join(audio_str[:3])}; Needs improvement: {', '.join(audio_imp[:3])}. Probe these weak areas.\n"

    return f"""You are Kaira, a strict senior technical interviewer conducting a video interview.
You appear as an AI avatar on screen. This is a {complexity_label} video interview.

CANDIDATE:
- Name: {name}
- Role: {role}
- Experience: {exp} years at {company}
- Skills: {skills}
- Projects: {", ".join(projects[:4]) if projects else "see resume below"}

RESUME & PRIOR INTERVIEW CONTEXT:
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

VIDEO INTERVIEW RULES (spoken language only, no markdown):
1. Ask exactly {num_questions} questions total based ONLY on this candidate's actual experience
2. Each response must be 1-3 SHORT spoken sentences — no walls of text
3. NEVER explain, hint, teach, or help — you are an evaluator only
4. If asked to explain: say only "I am here to evaluate, not guide you."
5. If off topic: say only "Let us stay focused." then repeat the question briefly
6. After each answer, give ONE brief acknowledgement sentence maximum, then ask the NEXT question
7. NO markdown, NO bullets, NO asterisks, NO emojis — spoken language only
8. Ground every question in their specific projects and declared tech stack
9. Maximum 1 follow-up per question, only if the answer was too vague to evaluate
10. Be professional and direct — you are a senior panel interviewer, not a coach
11. After question {num_questions} is answered, say your closing statement — "Thank you [name], that concludes our interview. Your evaluation is now being generated. Have a wonderful day." — then stop speaking"""


# ─── Answer Scorer ────────────────────────────────────────────────────────────

def score_answer_turn(text: str) -> float:
    text_clean = text.strip().lower()
    words = len(text_clean.split())
    if words < 4 or any(p in text_clean for p in ["dont know", "don't know", "no idea", "not sure", "pass", "i don't"]):
        return 0.0
    tech_keywords = [
        "system", "architecture", "design", "database", "sql", "nosql", "api", "rest", "graphql",
        "service", "microservice", "cache", "redis", "latency", "scale", "throughput", "async",
        "thread", "process", "memory", "cpu", "performance", "test", "security", "auth", "token",
        "model", "pipeline", "function", "class", "data", "algorithm", "complexity", "index",
        "query", "node", "server", "client", "request", "response", "error", "event", "queue",
        "react", "component", "hook", "state", "deployment", "docker", "kubernetes", "ci", "cd",
        "implement", "use", "build", "create", "handle", "manage", "optimize", "because", "which", "when",
    ]
    has_tech_kw = any(kw in text_clean for kw in tech_keywords)
    if words >= 20 and has_tech_kw:
        return 1.0
    elif words >= 10:
        return 0.5
    else:
        return 0.5 if has_tech_kw else 0.0


# ─── Question Number Detector ─────────────────────────────────────────────────

QUESTION_PATTERN = re.compile(r"question\s+(\d+)\s+of\s+\d+", re.IGNORECASE)

def extract_question_number(text: str):
    m = QUESTION_PATTERN.search(text)
    return int(m.group(1)) if m else None


# ─── Agent Class ──────────────────────────────────────────────────────────────

class KairaVideoAgent(Agent):
    """Strict video technical interviewer with Beyond Presence avatar."""

    def __init__(self, instructions, candidate_data, session_id, user_id, num_questions):
        super().__init__(instructions=instructions)
        self.candidate_data      = candidate_data
        self.session_id          = session_id
        self.user_id             = user_id
        self.num_questions       = num_questions
        self.questions_asked     = 0    # updated via session conversation_item_added
        self.questions_answered  = 0    # updated via on_user_turn_completed
        self.question_scores     = []
        self._transcript         = []
        self._is_ending          = False
        self._room               = None  # set after session.start()
        self._interview_started  = False  # True after Kaira's first message

    def set_room(self, room):
        """Called after session.start() to bind room for data publishing."""
        self._room = room

    async def _publish_data(self, data_dict: dict):
        """Publish JSON to frontend via LiveKit data channel."""
        try:
            payload = json.dumps(data_dict).encode("utf-8")
            # Try room first, then session.room as fallback
            room = self._room
            if room is None and hasattr(self, "session") and self.session:
                room = getattr(self.session, "room", None)
            if room:
                await room.local_participant.publish_data(payload)
            else:
                logger.warning("_publish_data: no room available yet")
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
        """Called every time candidate finishes speaking."""
        if self._is_ending:
            return

        text = getattr(new_message, "text_content", None) or str(new_message)
        if not text or not text.strip():
            return

        candidate_text = text.strip()

        # Always record and publish the transcript turn for display
        self.add_transcript_turn("Candidate", candidate_text)

        # ── Answer counting guards ────────────────────────────────────────────
        # Guard 1: Only count after Kaira has spoken her first message.
        # _interview_started is set True by the session-level
        # conversation_item_added handler when Kaira's first assistant
        # message arrives (reliable — fires even in bey-avatar mode).
        if not self._interview_started:
            logger.debug("Skipping count — Kaira hasn't spoken yet")
            return

        # Clean and lowercase text for check
        clean_text = "".join(c for c in candidate_text.lower() if c.isalnum() or c.isspace()).strip()
        words = clean_text.split()

        # Check if the user is explicitly saying they don't know or passing (which is a valid completed turn)
        knows_nothing = any(p in clean_text for p in ["dont know", "don't know", "no idea", "not sure", "pass", "i don't"])

        # Common filler/greeting/system test words
        fillers = {
            "sorry", "hmm", "hm", "ok", "okay", "yes", "no", "hello", "hi", "hey",
            "pardon", "repeat", "what", "can you hear me", "test", "testing",
            "audio", "hear", "you", "me", "is", "it", "working", "hello?", "hi?",
            "sorry?", "pardon?", "what?"
        }

        # Is it filler/irrelevant?
        is_irrelevant = (
            (len(words) < 4 and not knows_nothing) or
            (len(words) > 0 and all(w in fillers for w in words)) or
            any(phrase in clean_text for phrase in [
                "can you hear me", "is it working", "test test", "testing testing",
                "could you repeat", "please repeat", "repeat the question", "what was the question",
                "what did you say", "i didnt hear", "i didn't hear"
            ])
        )

        if is_irrelevant:
            logger.info(f"Video filler/irrelevant response detected: {candidate_text!r}. Instructing LLM to stay on current question.")
            new_message.content = [
                f"[System Note: The candidate's response was completely irrelevant filler/noise. "
                f"Do NOT ask a new question. You must stay on the same question number. "
                f"Politely ask them to answer Question {self.questions_answered + 1} of {self.num_questions} again.]"
            ]
            current_q = max(self.questions_asked, self.questions_answered)
            await self._publish_data({
                "type":            "question_progress",
                "question_number": current_q,
                "num_questions":   self.num_questions,
                "score":           None,
                "is_irrelevant":   True,
            })
            return
        # ─────────────────────────────────────────────────────────────────────

        # Score and count
        score = score_answer_turn(candidate_text)
        self.question_scores.append(score)
        self.questions_answered += 1

        # Display counter = whichever is higher (Kaira's spoken vs user answers)
        current_q = max(self.questions_asked, self.questions_answered)

        await self._publish_data({
            "type":            "question_progress",
            "question_number": current_q,
            "num_questions":   self.num_questions,
            "score":           score,
        })
        logger.info(
            f"Video Q{current_q}/{self.num_questions} answered — "
            f"score={score} words={len(words)}"
        )

        # Auto-end when all questions answered
        if self.questions_answered >= self.num_questions:
            asyncio.create_task(self._do_close())

    async def on_agent_speech_committed(self, message) -> None:
        """Called when Kaira finishes speaking a turn. Detects 'Question N of M'."""
        text = getattr(message, "text", None) or str(message)
        if not text or not text.strip():
            return

        self.add_transcript_turn("Kaira", text)

        # Detect "Question N of M" in Kaira's speech to update frontend counter
        q_num = extract_question_number(text)
        if q_num and q_num > self.questions_asked:
            self.questions_asked = q_num
            logger.info(f"Kaira spoke Question {q_num}/{self.num_questions}")
            # Position-only update (score=null) to sync the counter
            await self._publish_data({
                "type":            "question_progress",
                "question_number": q_num,
                "num_questions":   self.num_questions,
                "score":           None,
            })

    async def _do_close(self):
        """Speak closing, broadcast completion, disconnect."""
        if self._is_ending:
            return
        self._is_ending = True

        name = self.candidate_data.get("name", "there")
        closing = (
            f"Thank you {name} for completing all {self.num_questions} questions of this video interview. "
            f"Your responses have been recorded and your evaluation is now being generated. "
            f"We will be in touch shortly. Have a wonderful day."
        )
        logger.info(f"Interview complete — closing session {self.session_id}")

        # Tell frontend first
        await self._publish_data({
            "type":            "interview_complete",
            "question_scores": self.question_scores,
            "transcript":      self._transcript,
        })

        # Speak closing words
        try:
            await self.session.say(closing, allow_interruptions=False)
        except Exception as e:
            logger.warning(f"Closing say() failed: {e}")

        await asyncio.sleep(4)

        # Disconnect
        try:
            await self.session.aclose()
        except Exception:
            pass
        if self._room:
            try:
                await self._room.disconnect()
            except Exception:
                pass

    async def on_exit(self) -> None:
        """Save transcript and scores when session ends."""
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
            logger.info(f"Session {self.session_id} saved — {len(self._transcript)} turns, {len(self.question_scores)} scores")
        except Exception as e:
            logger.error(f"Failed to save transcript on exit: {e}")


# ─── Context Loader ───────────────────────────────────────────────────────────

async def _load_video_session_context(room_name: str) -> dict:
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
                "num_questions":  session.get("num_questions", 5),
            }
    except Exception as e:
        logger.error(f"Context load failed for room {room_name}: {e}")
    return {
        "session_id": room_name,
        "user_id": "anonymous",
        "candidate_data": {
            "name": "Candidate", "role": "Software Engineer", "skills": "",
            "strengths": [], "areas_to_explore": [], "project_names": [],
        },
        "rag_context":   "",
        "complexity":    "medium",
        "num_questions": 5,
    }


# ─── Agent Server ─────────────────────────────────────────────────────────────

server = AgentServer()


@server.rtc_session(agent_name="kaira-video")
async def kaira_video_session(ctx: agents.JobContext):
    """Entrypoint for LiveKit Cloud job dispatch — video interview with Beyond Presence avatar."""
    room_name = ctx.room.name
    logger.info(f"Video job received for room: {room_name}")

    ctx_data       = await _load_video_session_context(room_name)
    candidate_data = ctx_data["candidate_data"]
    complexity     = ctx_data["complexity"]
    num_questions  = ctx_data["num_questions"]

    instructions = build_kaira_video_instructions(
        candidate_data=candidate_data,
        rag_context=ctx_data["rag_context"],
        complexity=complexity,
        num_questions=num_questions,
    )

    agent = KairaVideoAgent(
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

    # ── Beyond Presence Avatar — EXACTLY as reference: start BEFORE session.start() ──
    bey_avatar = None
    is_audio_enabled = True  # disable when avatar handles audio

    if BEY_API_KEY and BEY_AVATAR_ID:
        try:
            logger.info("🎭 Creating Bey avatar session...")
            bey_avatar = bey.AvatarSession(
                api_key=BEY_API_KEY,
                avatar_id=BEY_AVATAR_ID,
            )
            logger.info("✅ Bey avatar session created")
        except Exception as e:
            logger.error(f"❌ bey.AvatarSession() failed: {e}")
            bey_avatar = None
    else:
        logger.warning("⚠️ BEY_API_KEY or BEY_AVATAR_ID missing — audio-only mode")

    if bey_avatar:
        try:
            logger.info("🎭 Starting Bey avatar before session.start()...")
            # bey.AvatarSession.start requires LiveKit credentials explicitly
            await bey_avatar.start(
                session,
                room=ctx.room,
                livekit_url=LIVEKIT_URL,
                livekit_api_key=LIVEKIT_API_KEY,
                livekit_api_secret=LIVEKIT_API_SECRET,
            )
            is_audio_enabled = False  # avatar handles audio — disable room TTS output
            logger.info("✅ Bey avatar started — video track published, audio disabled for room")
        except Exception as e:
            logger.error(f"❌ bey_avatar.start() failed: {e}")
            bey_avatar = None
            is_audio_enabled = True

    # ── Start session — following reference pattern exactly ───────────────────
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_S,
            ),
        ),
        room_output_options=RoomOutputOptions(audio_enabled=is_audio_enabled),
    )

    # Bind room to agent for data publishing
    agent.set_room(ctx.room)

    # ── Session-level handler: reliably captures Kaira's text ─────────────────
    # conversation_item_added fires for every LLM output even in bey-avatar mode.
    # Use it to: (a) publish Kaira transcript, (b) update questions_asked,
    # (c) set _interview_started so answer counting can begin.
    _kaira_seen_texts: set = set()  # deduplicate if on_agent_speech_committed also fires

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

    name             = candidate_data.get("name", "there")
    role             = candidate_data.get("role", "this role")
    complexity_label = {
        "easy": "foundational", "medium": "applied",
        "hard": "advanced", "expert": "expert",
    }.get(complexity, "technical")

    opening = (
        f"Greet {name} by name. Introduce yourself as Kaira, an AI video interviewer. "
        f"State this is a {complexity_label}-level interview for the {role} role covering {num_questions} questions. "
        f"Then immediately ask Question 1 of {num_questions} — a specific technical question from their resume or primary skill. "
        f"Keep the entire opening to 3 spoken sentences. Be direct and professional. No markdown."
    )

    await session.generate_reply(instructions=opening)


if __name__ == "__main__":
    agents.cli.run_app(server)
