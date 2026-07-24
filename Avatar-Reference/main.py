# main.py
import asyncio
import sys
import os
import json
import logging
import signal

from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    JobContext, WorkerOptions, cli, WorkerType, 
    RoomInputOptions, RoomOutputOptions, BackgroundAudioPlayer, 
    AudioConfig, BuiltinAudioClip, UserInputTranscribedEvent, AgentSession
)
from livekit.agents import mcp
from livekit.plugins import noise_cancellation, silero, tavus, bey
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from mem0 import AsyncMemory

# Import config
load_dotenv()
from config import setting
from mem0_config import MEM0_CONFIG
# from medplum_token_manager import MedplumTokenManager, get_medplum_auth_header

# Import centralized Medplum token management
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../backend'))


# Import modules
from helpers import (
    UserData, load_info_from_metadata, get_country_from_metadata, 
    determine_starting_agent, get_mcp_server_url , start_audio_egress,
    upload_json_to_blob,extract_text_from_item, setup_langfuse, fetch_resume_conversation_context,
    AZURE_CONTAINER_NAME
)
from universal_agent import UniversalAgent
# Import MetricsCollectedEvent if not already
from livekit.agents.voice import MetricsCollectedEvent
from livekit.agents.metrics.base import RealtimeModelMetrics


from opentelemetry import trace
from typing import Optional
from livekit.agents.voice import ConversationItemAddedEvent  # Add this
from langfuse_methods import setup_langfuse_with_enrichment, LangfuseVoiceMetricsTracker

# Logging
logger = logging.getLogger("livekit")
logger.setLevel(logging.DEBUG)
sys.stdout.reconfigure(line_buffering=True)

# Configuration
LIVEKIT_API_KEY = setting("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = setting("LIVEKIT_API_SECRET")
LIVEKIT_URL = setting("LIVEKIT_URL")

# BeyondPresence Configuration
BEY_API_KEY = os.getenv("BEY_API_KEY")
BEY_AVATAR_ID = os.getenv("BEY_AVATAR_ID")

# Optional BeyondPresence Configuration (if needed)
BEY_BASE_URL = os.getenv("BEY_BASE_URL")  # Custom API endpoint if needed
BEY_MODEL_ID = os.getenv("BEY_MODEL_ID")  # Specific model/voice if needed


# Environment prefix for room names (healthflow_dev, healthflow_live, etc.)
ENVIRONMENT_PREFIX = os.getenv("ENVIRONMENT_PREFIX", "healthflow_dev")

store_data_in_blob = os.getenv("STORE_DATA_IN_BLOB", "true").lower() == "true"

_active_rag_tasks: set = set()

def prewarm(proc: JobContext):
    proc.userdata["preload"] = "data"
    print("Agent process prewarmed and ready.")

async def _entrypoint_impl(ctx: JobContext):
    await ctx.connect()
    
    # ✅ CRITICAL FIX: Get room info AFTER connecting
    room_sid = await ctx.room.sid
    room_name = ctx.room.name
    
    print(f"\n{'='*70}\n🔥 ENTRYPOINT FIRED for room: {room_name} (SID: {room_sid})\n{'='*70}\n", flush=True)
    
    # Variables for Langfuse spans
    DEBUG_TRANSCRIPTS = "1"  # Set to "1" to enable debug logging
    last_user_text: Optional[str] = None
    last_total_latency_s: Optional[float] = None
    
    if True: 
        # Room filter check
        if not ctx.room.name.startswith(f"{ENVIRONMENT_PREFIX}-"):
            print(f"❌ SKIPPING - Room doesn't match filter: {ctx.room.name}")
            return
        
        # Metadata Logic
        participant = None
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=45.0)
        except Exception as e:
            print(f"⚠️ No participant after 45s: {e}")
        
        meta = {}
        ctx._agent_meta = None
        max_attempts = 40
        
        for attempt in range(max_attempts):
            if participant:
                try:
                    raw_meta = participant.metadata
                    meta = json.loads(raw_meta or "{}")
                except:
                    meta = {}
            
            if meta.get("healthcareProvider") or meta.get("scenario"):
                print(f"✅ Metadata ready after {attempt + 1} attempts")
                ctx._agent_meta = meta
                break
            
            await asyncio.sleep(0.5 if attempt < 2 else (1.0 if attempt < 5 else 2.0))
        
        if not ctx._agent_meta:
            print("❌ CRITICAL: No metadata received.")
            return
        
        # Info Extraction
        hcp_gender, language, user_name, patient_id, resume_conversation_id = load_info_from_metadata(ctx)
        country_code = get_country_from_metadata(ctx)
        scenario_data = ctx._agent_meta
        
        print("***************************************")
        print(f"User Name        : {user_name}")
        print(f"Patient ID       : {patient_id}")
        print(f"Country Code     : {country_code}")
        print(f"Resume Conv ID   : {resume_conversation_id}")
        print(f"Scenario Data : {scenario_data}")
        print("***************************************")
        
        # added the langfuse trace provider for web agent
        trace_provider = setup_langfuse_with_enrichment(
            metadata={
                "langfuse.session.id": ctx.room.name,
                "langfuse.user.id": patient_id,
                "langfuse.tags": [language, country_code],
                "langfuse.environment": "web_agent",
                "langfuse.trace.name": "Web Agent Session"
            }
            
        )
        # Setup MCP inside UniversalAgent itself, so we just log the action here
        starting_agent_type = determine_starting_agent(scenario_data)
        
        # Context Setup
        use_memory = meta.get("use_memory", True)
        userdata = UserData(
            scenario_data=scenario_data,
            timezone="Asia/Kolkata",
            ctx=ctx,
            patient_id=str(patient_id),
            use_memory=use_memory,
            resume_conversation_id=resume_conversation_id,
        )
        
        async def write_transcript_to_blob():
            try:
                transcript = session.history.to_dict()
                
                # Extract metadata for path construction
                meta = ctx._agent_meta or {}
                country = get_country_from_metadata(ctx)
                provider_info = meta.get("healthcareProvider", {})
                org_name = provider_info.get("name", "default_org").replace(" ", "_").lower()
                patient_id = userdata.patient_id or "unknown"
                
                now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                
                # Determine session type
                is_resume = bool(userdata.resume_conversation_id)
                session_suffix = f"_resume_{userdata.resume_conversation_id[:8]}" if is_resume else ""
                
                # Transcripts in separate folder: {country}/{org_name}/intake/{patient_id}/transcripts/
                blob_path = f"{country}/{org_name}/intake/{patient_id}/transcripts/session_{now_tag}{session_suffix}.json"
                await upload_json_to_blob(blob_path, transcript)
                logger.info("✅ Transcript uploaded to Blob")
                logger.info(f"📄 Transcript Path: {blob_path}")
                logger.info(f"📦 Container: {AZURE_CONTAINER_NAME}")
            except Exception as e:
                logger.error(f"❌ Failed to upload transcript: {e}")
        
        async def write_session_report_to_blob():
            try:
                report = ctx.make_session_report().to_dict()
                
                # Extract metadata for path construction
                meta = ctx._agent_meta or {}
                country = get_country_from_metadata(ctx)
                provider_info = meta.get("healthcareProvider", {})
                org_name = provider_info.get("name", "default_org").replace(" ", "_").lower()
                patient_id = userdata.patient_id or "unknown"
                
                now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                
                # Determine session type
                is_resume = bool(userdata.resume_conversation_id)
                session_suffix = f"_resume_{userdata.resume_conversation_id[:8]}" if is_resume else ""
                
                # Reports in transcripts folder: {country}/{org_name}/intake/{patient_id}/transcripts/
                blob_path = f"{country}/{org_name}/intake/{patient_id}/transcripts/session_report_{now_tag}{session_suffix}.json"
                await upload_json_to_blob(blob_path, report)
                logger.info("✅ Session report uploaded to Blob")
                logger.info(f"📊 Report Path: {blob_path}")
                logger.info(f"📦 Container: {AZURE_CONTAINER_NAME}")
            except Exception as e:
                logger.error(f"❌ Failed to upload session report: {e}")
        
        # storing session transcripts and report in blob 
        if store_data_in_blob:
            ctx.add_shutdown_callback(write_transcript_to_blob)
            ctx.add_shutdown_callback(write_session_report_to_blob)
        
        # Add avatar cleanup callback
        async def cleanup_avatar():
            if hasattr(userdata, 'bey_avatar') and userdata.bey_avatar:
                try:
                    print("🎭 Stopping Bey avatar...")
                    if hasattr(userdata.bey_avatar, 'stop'):
                        await asyncio.wait_for(userdata.bey_avatar.stop(), timeout=3.0)
                    print("✅ Bey avatar stopped")
                except Exception as e:
                    logger.error(f"❌ Avatar cleanup failed: {e}")
                finally:
                    userdata.bey_avatar = None
        
        ctx.add_shutdown_callback(cleanup_avatar)
        
        # Initialize Mem0
        if userdata.use_memory:
            try:
                print("🧠 Initializing Mem0...")
                userdata.mem0 = AsyncMemory(config=MEM0_CONFIG)
                print("✅ Mem0 initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Mem0: {e}")
                userdata.mem0 = None
        
        # ── Agent created FIRST so we can read its stt/tts/llm for the session ──────
        # Agent Creation
        starting_agent_type = determine_starting_agent(scenario_data)

        # Fetch resume conversation context if resuming
        resume_context = ""
        print(f"\n{'='*70}\n🔥 ENTRYPOINT FIRED for room: {resume_conversation_id}\n{'='*70}\n")
        if resume_conversation_id:
            resume_context = await fetch_resume_conversation_context(resume_conversation_id)

        agent_kwargs = {
            "agent_type": starting_agent_type,
            "room_name": ctx.room.name,
            "hcp_gender": hcp_gender,
            "language": language,
            "user_name": user_name,
            "patient_id": patient_id,
            "country_code": country_code,
            "scenario_data": scenario_data,
            "resume_conversation_id": resume_conversation_id,
            "resume_context": resume_context,
        }

        # Create initial agent directly
        initial_agent = UniversalAgent(**agent_kwargs)
        
        # Initialize Langfuse Voice Metrics Tracker
        voice_metrics_tracker = LangfuseVoiceMetricsTracker(
            room_name=ctx.room.name,
            patient_id=str(patient_id),
            patient_name=user_name,
            organization_id=scenario_data.get("healthcareProvider", {}).get("name", "unknown"),
            language=language,
            agent_type=starting_agent_type,
            country=country_code,
            trace_provider=trace_provider,
        )

        async def close_tracker():
            await voice_metrics_tracker.aclose()
        
        ctx.add_shutdown_callback(close_tracker)
        
        # ─────────────────────────────────────────────────────────────────────────
        # AVATAR HANDLER (BeyondPresence Integration)
        # ─────────────────────────────────────────────────────────────────────────
        
        # Check if this is an avatar agent
        is_avatar = starting_agent_type in [
            "heather_avatar_agent",
            "maya_avatar_agent", 
            "narcolepsy_avatar_agent",
            "gi_procedure_companion_pre_post_care_avatar_agent",
        ]
        
        bey_avatar = None
        is_audio_enabled = True  # Default to audio enabled
        
        if is_avatar:
            print(f"🎭 Initializing avatar for agent type: {starting_agent_type}")
            
            # Define avatar_id mapping for different avatar agents
            avatar_map = {
                "heather_avatar_agent": BEY_AVATAR_ID,  
                "maya_avatar_agent": BEY_AVATAR_ID,     
                "narcolepsy_avatar_agent": BEY_AVATAR_ID,
                "gi_procedure_companion_pre_post_care_avatar_agent": BEY_AVATAR_ID,
            }
            
            avatar_id = avatar_map.get(starting_agent_type, BEY_AVATAR_ID)
            api_key = BEY_API_KEY
            
            # Check if required Bey configuration is available
            if not api_key or not avatar_id:
                logger.error(
                    f"[AVATAR] Missing Bey config for {starting_agent_type}. "
                    f"API_KEY={'OK' if api_key else 'MISSING'}, "
                    f"AVATAR_ID={'OK' if avatar_id else 'MISSING'}"
                )
                print(f"⚠️ Avatar config missing, falling back to audio-only mode")
                print(f"💡 Set BEY_API_KEY and BEY_AVATAR_ID environment variables")
            else:
                try:
                    print(f"🎭 Creating Bey avatar session...")
                    
                    # Create BeyondPresence avatar session with available parameters
                    avatar_params = {
                        "api_key": api_key,
                        "avatar_id": avatar_id,
                    }
                    
                    # Add optional parameters if available
                    if BEY_BASE_URL:
                        avatar_params["base_url"] = BEY_BASE_URL
                    if BEY_MODEL_ID:
                        avatar_params["model_id"] = BEY_MODEL_ID
                    
                    bey_avatar = bey.AvatarSession(**avatar_params)
                    
                    # Store avatar in userdata for cleanup
                    userdata.bey_avatar = bey_avatar
                    
                    print(f"✅ Bey avatar created successfully for {starting_agent_type}")
                    print(f"🎭 Avatar config - ID: {avatar_id[:8]}...")
                    if BEY_MODEL_ID:
                        print(f"🎭 Using model: {BEY_MODEL_ID}")
                    is_audio_enabled = False  # Disable audio when avatar is active
                    
                except Exception as e:
                    logger.error(f"[AVATAR] Bey failed for {starting_agent_type}: {e}. Falling back to audio.")
                    print(f"❌ Avatar initialization failed: {e}")
                    bey_avatar = None
                    is_audio_enabled = True
        else:
            print(f"🔊 Using audio-only mode for agent type: {starting_agent_type}")
        

        # ── CASCADED PIPELINE: pass STT / LLM / TTS to AgentSession ─────────────────
        # The agent's __init__ built and stored these components; we read them here.
        # To revert to realtime: replace the lines below with the commented line.
        session = AgentSession[UserData](
            stt=initial_agent._stt,               # Soniox STT
            llm=initial_agent._llm_obj,           # Azure OpenAI LLM
            tts=initial_agent._tts,               # Azure TTS (replaced Cartesia)
            vad=silero.VAD.load(min_silence_duration=0.4, min_speech_duration=0.1),
            preemptive_generation=True,           # Start generating before turn is committed
            userdata=userdata,
            max_tool_steps=15,                    # maximum number of consecutive tool calls (tool steps) allowed per LLM turn
            mcp_servers=initial_agent._mcp_servers,  # Native MCP servers from UniversalAgent

        )

        # Register Langfuse metrics events
        session.on("metrics_collected", voice_metrics_tracker.on_metrics_collected)
        session.on("user_input_transcribed", voice_metrics_tracker.on_user_input_transcribed)
        def on_conversation_item_added(ev):
            voice_metrics_tracker.on_conversation_item_added(ev)
            item = getattr(ev, "item", None)
            item_type = getattr(item, "type", "unknown")
            
            # Print tool calls
            if item_type == "function_call":
                print(f"\n{'='*50}")
                print(f"🚀 TOOL CALL: {getattr(item, 'name', 'unknown')}")
                print(f"📥 Arguments: {getattr(item, 'arguments', '{}')}")
                print(f"{'='*50}")
            elif item_type == "function_call_output":
                out_str = str(getattr(item, 'output', ''))
                truncated = out_str[:500] + "..." if len(out_str) > 500 else out_str
                tool_name = getattr(item, "name", "unknown")
                print(f"\n{'='*50}")
                print(f"✅ TOOL RESULT: {tool_name}")
                print(f"📤 Output: {truncated}")
                print(f"{'='*50}\n")
                
        session.on("conversation_item_added", on_conversation_item_added)
        
        # Also listen for function_tools_executed for detailed tool execution logs
        def on_tools_executed(ev):
            results = getattr(ev, "function_call_results", [])
            if results:
                print(f"\n🔧 {'='*50}")
                print(f"🔧 TOOL EXECUTION COMPLETE - {len(results)} tool(s)")
                for r in results:
                    fn_info = getattr(r, "function_info", None)
                    tool_name = getattr(fn_info, "name", "unknown") if fn_info else "unknown"
                    result_content = getattr(r, "result", "")
                    truncated = str(result_content)[:500]
                    print(f"   🔧 {tool_name} → {truncated}")
                print(f"🔧 {'='*50}\n")
                
        session.on("function_tools_executed", on_tools_executed)
        # ── REALTIME PIPELINE (revert): uncomment and delete the block above ──
        # session = AgentSession[UserData](userdata=userdata)
        
        # RAG Logic
        async def handle_user_turn_with_rag(text: str, agent_session: AgentSession, ud: UserData):
            if not agent_session.current_agent: 
                return
            rag_uid = str(ud.patient_id)
            
            if ud.use_memory and ud.mem0 and rag_uid and rag_uid != "None":
                # Search
                try:
                    results = await asyncio.wait_for(ud.mem0.search(text, user_id=rag_uid, limit=3), timeout=5.0)
                    memories = [r["memory"] for r in results["results"]]
                    if memories:
                        rag_context = "Relevant past context:\n" + "\n".join(f"- {m}" for m in memories[:3])
                        chat_ctx = agent_session.current_agent.chat_ctx.copy()
                        chat_ctx.add_message(role="system", content=rag_context)
                        await agent_session.current_agent.update_chat_ctx(chat_ctx)
                        print(f"***** Injected RAG Context *****")
                except Exception as e:
                    print(f"⚠️ RAG Search failed: {e}")
            
            # Generate Reply
            await agent_session.generate_reply()
            
            # Save to Memory
            if ud.use_memory and ud.mem0 and rag_uid and rag_uid != "None":
                try:
                    await ud.mem0.add([{"role": "user", "content": text}], user_id=rag_uid, infer=False)
                except Exception as e:
                    print(f"⚠️ Memory save failed: {e}")
                    
        
        # Start Session
        # Typing sound is NOT used for normal LLM thinking (reduces perceived latency).
        # It plays ONLY during tool/function call execution — see handler below.
        background_audio = BackgroundAudioPlayer()
        background_audio = BackgroundAudioPlayer(
            thinking_sound=[
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5, probability=0.8),
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.4),
            ],
        )

        # ── Tool-call typing sound ──────────────────────────────────────────────────
        # Plays looping typing audio while the agent executes a tool, stops on completion.
        _tool_audio: dict = {"handle": None}

        @session.on("conversation_item_added")
        def _on_tool_call_audio(event: ConversationItemAddedEvent):
            item = event.item
            item_type = getattr(item, "type", None)
            role = getattr(item, "role", None)

            # Debug: log every item so we can confirm the exact type/role values
            logger.debug(f"📋 conversation_item_added | type={item_type!r} role={role!r}")

            # Pipeline agents: function call = type 'function_call'
            # Fallback: some versions use role='assistant' with tool_calls attribute
            is_tool_start = (
                item_type == "function_call"
                or (role == "assistant" and bool(getattr(item, "tool_calls", None)))
            )
            # Pipeline agents: tool output = type 'function_call_output' or role 'tool'
            is_tool_end = (
                item_type == "function_call_output"
                or role == "tool"
            )

            if is_tool_start:
                logger.info("🔧 Tool call started — playing typing sound")
                _tool_audio["handle"] = background_audio.play(
                    [
                        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8, probability=0.8),
                        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7, probability=0.2),
                    ],
                    loop=True,
                )
            elif is_tool_end:
                if _tool_audio["handle"]:
                    logger.info("✅ Tool call complete — stopping typing sound")
                    _tool_audio["handle"].stop()
                    _tool_audio["handle"] = None

        is_audio_enabled = True

        # ⚠️  IMPORTANT: start background_audio BEFORE session.start()
        # so the player is ready when tool-call events fire.
        await background_audio.start(room=ctx.room, agent_session=session)

        # Pre-warm Medplum token before session starts (Issue 3 fix)
        # try:
        #     token_manager = MedplumTokenManager()
        #     await token_manager.ensure_valid_token()
        # except Exception as e:
        #     print(f"⚠️ Token pre-warm failed: {e}")

        # Start the avatar if available (must be done before session.start)
        if bey_avatar:
            try:
                print(f"🎭 Starting Bey avatar...")
                await bey_avatar.start(session, room=ctx.room)
                print(f"✅ Bey avatar started successfully")
                print(f"📹 Video track should now be available for frontend")
            except Exception as e:
                logger.error(f"[AVATAR] Failed to start Bey avatar: {e}")
                print(f"❌ Avatar start failed: {e}")
                # Fall back to audio mode
                is_audio_enabled = True
                bey_avatar = None
                userdata.bey_avatar = None

        await session.start(
            agent=initial_agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                text_enabled=True,
                close_on_disconnect=True
            ),
            room_output_options=RoomOutputOptions(audio_enabled=is_audio_enabled),
        )
        if store_data_in_blob:
            eng_started=await start_audio_egress(ctx,userdata)
            if eng_started:
                logger.info("✅ Audio Egress Service Started")
                logger.info(f"🎙️ Audio Path: {userdata.audio_blob_path}")
                logger.info(f"📦 Container: {userdata.audio_blob_container}")
                logger.info(f"🆔 Egress ID: {userdata.egress_id}")
            else:
                logger.warning("⚠️ Audio Egress Service failed to start")
        else:
            logger.info("No Egress Service Started")

        logger.info(f"✅ Agent Session STARTED for room: {ctx.room.name}")
        print(f"✅ Agent Session STARTED for room: {ctx.room.name}")

async def entrypoint(ctx: JobContext):
    try:
        await _entrypoint_impl(ctx)
    except Exception as e:
        print(f"🔥 CRITICAL ERROR IN ENTRYPOINT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    _shutting_down = False
    
    def signal_handler(signum, frame):
        global _shutting_down
        if _shutting_down: sys.exit(1)
        _shutting_down = True
        print("\n🛑 Shutdown signal received...")
        # Add explicit Mem0 cleanup if singleton is used
        sys.exit(0)

    try:
        if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
            print("❌ CRITICAL: Missing LiveKit configuration.")
            # sys.exit(1)
                
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            port=8208,
            worker_type=WorkerType.ROOM,
            agent_name=f"{ENVIRONMENT_PREFIX}-agent",
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            shutdown_process_timeout=10.0,
            load_threshold=0.99,
        ))
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"🔥 CRITICAL ERROR IN MAIN: {e}")