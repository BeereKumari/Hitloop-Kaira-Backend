# base_agent.py
from __future__ import annotations
import logging
import json
import asyncio
from typing import Optional

from livekit.agents.voice import Agent, RunContext  
from livekit.agents.llm import RealtimeModelError
from livekit.agents import mcp

from helpers import UserData, load_info_from_metadata, get_country_from_metadata

logger = logging.getLogger("livekit.base_agent")

class BaseAgent(Agent):
    def __init__(
        self,
        *,
        agent_type: str,
        room_name: str,
        instructions: str,
        llm,
        stt=None,
        tts=None,
        turn_detection=None,
        vad=None,
        language: Optional[str] = None,
        user_name: Optional[str] = None,
        patient_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        mcp_servers: Optional[list[mcp.MCPServer]] = None,
        **kwargs,
    ):
        self.agent_type = agent_type
        self.room_name = room_name
        self.language = language
        self.user_name = user_name
        self.patient_id = patient_id
        self.organization_id = organization_id

        super().__init__(
            instructions=instructions,
            llm=llm,
            stt=stt,
            tts=tts,
            turn_detection=turn_detection,
            vad=vad,
            mcp_servers=mcp_servers,
            **kwargs,
        )

    async def on_enter(self) -> None:
        """Base hook: error wiring and simple agent-name broadcast."""
        userdata: UserData = self.session.userdata
        agent_name = self.agent_type
        print("===========================")
        print(f"******************{agent_name}******")
        print("===========================")
        
        # Diagnostic: confirm tools are visible at runtime
        from livekit.agents.utils.misc import is_given
        agent_mcp = self.mcp_servers
        session_mcp = self.session.mcp_servers if hasattr(self, 'session') else None
        print(f"🔍 Agent MCP servers: {agent_mcp} (is_given={is_given(agent_mcp)})")
        print(f"🔍 Session MCP servers: {session_mcp}")
        print(f"🔍 Agent tools count: {len(self.tools)}")

        # Generic error handler
        def _on_llm_error_sync(error: RealtimeModelError):
            async def _handle_error_async():
                logger.warning(f"Caught generic session error ({type(error.error).__name__}). Notifying frontend.")
                payload = {
                    "type": "agent_error",
                    "message": "Something went wrong. Please try again later.",
                }
                if userdata.ctx and userdata.ctx.room:
                    await userdata.ctx.room.local_participant.publish_data(json.dumps(payload), reliable=True)
            asyncio.create_task(_handle_error_async())

        self.session.on("error", _on_llm_error_sync)

        # Update display name in metadata
        provider_info = userdata.scenario_data.get("healthcareProvider", {})
        scenario_info = userdata.scenario_data.get("scenario", {})
        country = provider_info.get("countryName", "USA")

        agent_display_name_map = {
            "triage_agent": "Front Desk Agent",
            "support_agent": "Support Agent - Outpatients" ,
            "billing_agent": "Support Agent - Primary Care" if country in ["UK", "United Kingdom"] else "Billing Agent",
            "heather_avatar_agent": "Myasthenia Gravis Companion - Heather(Avatar)",
            "heather_audio_agent": "Myasthenia Gravis Companion - Heather(Audio)",
            "maya_avatar_agent": "Myasthenia Gravis Companion - Maya (Avatar)",
            "narcolepsy_audio_agent": "Narcolepsy Companion - Heather (Audio)",
            "narcolepsy_avatar_agent": "Narcolepsy Companion - Heather (Avatar)",
            "viiv_patient_agent": "HIV Companion - ViiV Healthcare",
            "novo_patient_agent": "Obesity Companion - Novo Nordisk",
            "health_companion_patient_audio_agent": "Heather - Health Companion (Audio)",
            "ehr_intake_agent": "ED intake agent",
            "patient_intake_agent":"Intake Agent - Inpatients",

        }

        display = agent_display_name_map.get(agent_name, scenario_info.get("scenarioName", agent_name))
        print("==============================")
        print("display", display)
        print("=================================")
        userdata.current_agent_running = display

        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes({"agent": display})

    async def _transfer_to_agent(self, name: str, context: RunContext[UserData], reason: str) -> Agent:
        """Transfer logic creating a new UniversalAgent."""
        logger.info(f"Initiating transfer to '{name}' agent for reason: {reason}")
        userdata = context.userdata
        room_name = self.room_name
        

        hcp_gender, language, user_name, patient_id, _resume_cid = load_info_from_metadata(userdata.ctx)
        country_code = get_country_from_metadata(userdata.ctx)
        scenario_data = userdata.scenario_data or {}

        agent_kwargs = {
            "agent_type": name,
            "room_name": room_name,
            "hcp_gender": hcp_gender,
            "language": language,
            "user_name": user_name,
            "patient_id": patient_id,
            "country_code": country_code,
            "scenario_data": scenario_data,
        }

        # Import locally to avoid circular import
        from universal_agent import UniversalAgent

        next_agent = UniversalAgent(**agent_kwargs)


        userdata.personas[name] = next_agent
        userdata.prev_agent = context.session.current_agent
        userdata.is_transfer = True
        userdata.transfer_reason = reason

        return next_agent

    def _truncate_chat_ctx(self, items: list, keep_last_n_messages: int = 6, keep_function_call: bool = True) -> list:
        def _valid(item) -> bool:
            if item.type == "message" and item.role in ["user", "assistant"]: return True
            if keep_function_call and item.type in ["function_call", "function_call_output"]: return True
            return False
        
        new_items = [item for item in reversed(items) if _valid(item)][:keep_last_n_messages]
        new_items.reverse()
        return new_items

    async def on_exit(self) -> None:
        """
        Hook called when agent session ends.
        For patient_intake_agent, this generates the patient summary and enqueues to physician queue.
        """
        print("Calling the on_exit function")
        # Only generate summary for patient_intake_agent
        if self.agent_type not in ["ehr_intake_agent", "xon_patient_intake_agent", "patient_intake_agent"]:
            return

        try:
            logger.info(f"🏥 Patient intake session ending - generating summary for patient_id: {self.patient_id}")
            
            # Import utilities
            from helpers import (
                build_conversation_text,
                enqueue_patient_after_intake,
                get_country_from_metadata,
                fetch_resume_conversation_transcript,
            )
            from models import PreVisitIntakeSummary, Metadata
            from datetime import datetime, timezone
            #from openai import AsyncAzureOpenAI
            from langfuse.openai import AsyncAzureOpenAI
            from config import setting
            import os
            from langfuse import get_client

            # os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "intake_summary"
            # langfuse = get_client()
            from langfuse import Langfuse
            langfuse = Langfuse(
                environment="intake_summary",
                public_key=setting("LANGFUSE_PUBLIC_KEY"),
                secret_key=setting("LANGFUSE_SECRET_KEY"),
                host=setting("LANGFUSE_BASE_URL"),
            )

            # Get userdata from session
            userdata: UserData = self.session.userdata
            
            # Extract current session transcript first.
            history = self.session.history.to_dict()
            print("📜 FULL SESSION HISTORY:")
            print(json.dumps(history, indent=2))
            current_session_transcripts = []
            
            for item in history.get("items", []):
                if item.get("type") != "message":
                    continue

                # Skip interrupted messages
                if item.get("interrupted"):
                    continue

                role = item.get("role")
                content = item.get("content", [])
                text = " ".join(content).strip()
                if not text:
                    continue
                speaker = "Agent" if role == "assistant" else "Patient"
                current_session_transcripts.append((speaker, text))

            # If this is a resumed conversation, prepend previous transcript so
            # summary extraction sees the full conversation context.
            transcripts = current_session_transcripts
            if getattr(userdata, "resume_conversation_id", None):
                previous_transcripts = await fetch_resume_conversation_transcript(userdata.resume_conversation_id)
                if previous_transcripts:
                    logger.info(
                        f"🔗 Merging resume transcript: previous={len(previous_transcripts)}, "
                        f"current={len(current_session_transcripts)}"
                    )
                    transcripts = previous_transcripts + current_session_transcripts
            
            if not transcripts:
                logger.warning("⚠️ No conversation transcript found")
                transcripts = [("System", "Session started and ended without conversation.")]

            # Ensure at least one patient message exists
            patient_msgs = [t for t in transcripts if t[0] == "Patient"]

            if not patient_msgs:
                logger.warning("⚠️ No patient speech detected. Skipping summary generation.")
                return
            
            # Get country and database
            country_code = get_country_from_metadata(userdata.ctx) if userdata.ctx else "UAE"
            logger.info(f"🌍 Detected country code: {country_code}")
            organization_id = getattr(self, "organization_id", None) or country_code
            
            # Build conversation text with metadata
            conversation_text = build_conversation_text(
                transcripts=transcripts,
                patient_id=int(self.patient_id) if (self.patient_id and str(self.patient_id).isdigit()) else 0,
                patient_name=self.user_name or "Unknown",
                organization_id=organization_id
            )

            print("🧠 CONVERSATION SENT TO LLM")
            print(conversation_text)
            
            # Generate summary using Azure OpenAI
            logger.info("📝 Generating patient summary using Azure OpenAI...")
            
            summary_system_prompt = """
            You need to extract structured data from a pre-visit intake conversation.
            Output must strictly follow the PreVisitIntakeSummary schema.
            Use ONLY information explicitly stated in the conversation.
            Output language: English only.

            ====================
            TEMPLATE STRUCTURE - PRIMARY FIELDS
            ====================
            ### Critical Note
            If the below information for mentioned fields is present in the pre-visit intake conversation it should extract those details perfectly without missing them eventhough the fields are optional.

            BASIC INFO
            - patient_name
            - age
            - gender
            - mobile_number
            - specialty_context: The medical specialty context (e.g., Cardiology, Neurology, Orthopedics, General Medicine)

            PRESENT COMPLAINTS (List of ChiefComplaint)
            - present_complaints: List of complaints, each with:
            * complaint: Main symptom/problem in patient's words
            * duration: How long they've had it
            * severity: Severity or progression
            * details: Additional context (radiation, triggers, associated symptoms, aggravating/relieving factors)

            MEDICAL HISTORY (Structured with status and details)
            - medical_history: Any chronic or significant medical condition with:
            * condition_name: Name of condition (e.g., Hypertension, Asthma, CKD)
            * status: true/false
            * duration: How long they've had it
            * severity: Stage or severity if known
            * notes: Additional details

            FAMILY HISTORY
            - family_history: Relevant family medical history with:
            * condition_name: Condition in family
            * relation: Father/Mother/etc
            * details: "..."

            PAST PROCEDURES (Surgeries and other past procedures including cardiac procedures if any)
            - past_procedures: List of past surgeries/procedures with:
            * procedure_name
            * date
            * place
            * outcome
            * notes

            PERSONAL HABITS (Extract for male patients)
            - personal_habits:
            * smoking: {status: true/false, frequency: "X per day", duration: "X years"}
            * alcohol: {status: true/false, frequency: "X times/week", duration: "X years"}
            * tobacco_chewing: {status: true/false, frequency: "...", duration: "..."}
            * recreational_drugs: {status: true/false, frequency: "...", duration: "..."}

            MEDICATIONS
            - known_allergies: List of drug/food allergies
            - current_medications: List of ConditionBasedMedication:
            * condition_name: Condition medication is prescribed for
            * medications: [{name, dose, frequency, route}]
            - recently_stopped_medications: [{medicine, reason, date_stopped}]

            PREVIOUS REPORTS AVAILABLE
            - previous_reports_available: List of reports patient mentioned having
            (e.g., "CAG Report", "ECG", "Echo", "Recent Blood Tests", "PTCA Report", "CABG Report")

            INTERNAL FIELDS (for physician context, always extract)
            - suggestions_to_doctor (list): 3-4 key questions doctor should ask or investigate

            PATIENT-FACING SUGGESTIONS
            - suggestions_to_patient (list): Questions or topics the patient might want to ask the doctor during consultation
            
            IMPORTANT RULE FOR INCOMPLETE CONVERSATIONS:
            * If the conversation was NOT completed fully (missing key information, abruptly ended, or lacks proper transcript details):
                - DO NOT generate respective information for the above fields as it may be inaccurate or fabricated
            * If the conversation IS complete with proper details:
                - Generate 3-5 relevant questions the patient should consider asking the doctor
                - Base suggestions ONLY on information explicitly mentioned in the conversation
                - Keep suggestions patient-friendly and non-technical
                - Examples: "Ask about medication side effects", "Discuss lifestyle modifications for managing [condition]"

            ### Spelling Normalization Rule:
            - You MAY use standard medical knowledge **only to correct incorrectly spelled medical terms, drug names, tests, or clinical words explicitly mentioned in the conversation**  
            (e.g., “paracetamal” → “paracetamol”).
            - Do NOT introduce new medications, diagnoses, tests, or instructions that are not explicitly stated.

            - Do NOT fill fields based on probability or common practice. If a required field is NOT clearly mentioned in the conversation, set it to `null` (or `"Not mentioned"` / `None` / empty list as appropriate).

            ====================
            EMOTIONAL & PSYCHOSOCIAL UNDERSTANDING (IMPORTANT)
            ====================

            These fields capture the human side of the conversation. Extract them whenever the patient expresses emotions or concerns.

            1. emotional_validation (Extract if patient expressed ANY feelings):
            - feelings_expressed: List ANY emotions mentioned (worried, scared, anxious, frustrated, confused, upset, etc.)
                Example: If patient says "I'm really worried about this pain" → ["worried"]
            - validation_provided: How the agent responded to these feelings
                Example: "I understand your concern", "That's completely normal to feel that way"
            - emotional_impact: How symptoms affect patient's daily life or emotional state
                Example: "Can't sleep due to pain", "Affecting work performance"

            2. key_concerns (Extract if patient asked questions or expressed worries):
            - concern: Patient's worry in their own words
                Example: "Is this something serious?", "Will I need surgery?"
            - explanation: How the agent addressed this concern
                Example: "We'll need to examine you first, but many cases are manageable with medication"
            
            IMPORTANT: Create one entry for EACH distinct concern or question the patient raised

            3. next_steps (Extract if agent explained what happens next):
            - step: What will happen next in patient-friendly language
                Example: "You'll meet with the cardiologist", "We need to do some blood tests"
            - purpose: Why this step is important
                Example: "To check how your heart is functioning", "To rule out any serious conditions"

            4. educational_points (Extract key health information shared):
            - List of educational points explained in simple terms
                Example: "High blood pressure can affect your heart over time", "Regular exercise helps control diabetes"

            ====================
            EXTRACTION GUIDELINES FOR EMOTIONAL FIELDS
            ====================

            - If patient says "I'm worried/scared/anxious" → MUST fill emotional_validation
            - If patient asks "Is this serious?" or "What if..." → MUST fill key_concerns
            - If patient says "What will happen next?" or "What do I do now?" → MUST fill next_steps
            - If agent explains health information or gives advice → MUST fill educational_points
            - Even brief emotional expressions should be captured
            - Look for implicit emotions too (e.g., "I can't sleep" suggests anxiety/distress)

            ====================
            DEFAULT VALUES
            ====================

            - Missing string fields → None
            - Missing list fields → [] empty list
            - Missing nested objects → None
            - Missing boolean values → None

            ====================
            STRICT RULES
            ====================
            - Do NOT add or infer information NOT in the conversation
            - Do NOT interpret medically beyond what patient said
            - Do NOT normalize or expand medical terms
            - Fill fields only if mentioned
            - For emotional fields: Be generous - if there's ANY hint of emotion/concern, extract it
            - For key_concerns: Each question or worry = separate entry
            - suggestions_to_doctor is the ONLY field where clinical reasoning is allowed
            - Everything must be derived from the conversation text only
            """

            client = AsyncAzureOpenAI(
                api_key=setting("AZURE_OPENAI_API_KEY"),
                # api_version=setting("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                api_version="2025-04-01-preview",
                azure_endpoint=setting("AZURE_OPENAI_ENDPOINT")
            )
            
            # Create schema for structured output
            deployment = setting("AZURE_OPENAI_DEPLOYMENT_CHAT", "gpt-4o")
            response = await client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": summary_system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Extract patient intake information from this conversation:\n\n{conversation_text}"
                    }
                ],
                response_format=PreVisitIntakeSummary,
                # temperature=0.3,
                metadata={
                    "langfuse_session_id": self.room_name,
                    "langfuse_user_id": str(self.patient_id),
                    "langfuse_tags": [country_code, "session_summary", deployment],
                }
            )
            
            summary_obj = response.choices[0].message.parsed
            
            if not summary_obj:
                logger.error("❌ Failed to parse summary from OpenAI response")
                return
            
            # Create dynamic metadata based on country
            country_metadata = {
                "UAE": {
                    "generated_by": "Arab Healthcare AI",
                    "healthcare_system": "UAE",
                    "facility": "Arab Global Medical Center"
                },
                "UK": {
                    "generated_by": "Royal Healthcare AI", 
                    "healthcare_system": "UK",
                    "facility": "Royal Westminister Center"
                }
            }
            
            metadata_config = country_metadata.get(country_code, country_metadata["UAE"])
            metadata = Metadata(
                generated_by=metadata_config["generated_by"],
                generated_on=datetime.now(timezone.utc),
                healthcare_system=metadata_config["healthcare_system"],
                facility=metadata_config["facility"],
            )
            
            # Convert to dict and inject metadata + organization info
            summary_dict = summary_obj.model_dump(mode="json", exclude_none=True)
            summary_dict["metadata"] = metadata.model_dump(mode="json")
            summary_dict["organization_id"] = organization_id
            summary_dict["roomName"] = self.room_name
            
            # Get patient name from demographics
            patient_name = "Unknown"
            if summary_obj.patient_demographics and summary_obj.patient_demographics.full_name:
                patient_name = summary_obj.patient_demographics.full_name.value or "Unknown"
            
            logger.info(f"✅ Summary generated successfully for patient {patient_name}")
            print(f'✅✅{summary_dict}✅✅')
            # Enqueue to physician queue
            success = enqueue_patient_after_intake(summary_dict)
            
            # Get patient_id for logging
            patient_id = summary_obj.patient_demographics.patient_id if summary_obj.patient_demographics else "Unknown"
            
            if success:
                logger.info(f"🎯 Patient {patient_id} successfully enqueued to physician queue")
            else:
                logger.error(f"❌ Failed to enqueue patient {patient_id} to physician queue")
                
        except Exception as e:
            logger.error(f"❌ Error in patient intake summary generation: {e}", exc_info=True)