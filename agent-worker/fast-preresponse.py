import asyncio
import logging
import json
from collections.abc import AsyncIterable
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    metrics,
    MetricsCollectedEvent
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram, groq, openai, silero
from prometheus_client import start_http_server, Summary, Counter, Gauge
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, VADMetrics, EOUMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pre-response-agent")

load_dotenv()

# Define Prometheus metrics
LLM_LATENCY = Summary('livekit_llm_duration_ms', 'LLM latency in milliseconds', ['model'])
STT_LATENCY = Summary('livekit_stt_duration_ms', 'Speech-to-text latency in milliseconds', ['provider'])
TTS_LATENCY = Summary('livekit_tts_duration_ms', 'Text-to-speech latency in milliseconds', ['provider'])
EOU_LATENCY = Summary('livekit_eou_delay_ms', 'End-of-utterance delay in milliseconds')
TOTAL_CONVERSATION_LATENCY = Summary('livekit_total_conversation_latency_ms', 'Total conversation latency in milliseconds')

# Usage metrics
LLM_TOKENS = Counter('livekit_llm_tokens_total', 'Total LLM tokens processed', ['type', 'model'])
STT_DURATION = Counter('livekit_stt_duration_seconds_total', 'Total STT audio duration in seconds', ['provider'])
TTS_CHARS = Counter('livekit_tts_chars_total', 'Total TTS characters processed', ['provider'])
TOTAL_TOKENS = Counter('livekit_total_tokens_total', 'Total tokens processed')
CONVERSATION_TURNS = Counter('livekit_conversation_turns_total', 'Number of conversation turns')
ACTIVE_CONVERSATIONS = Gauge('livekit_active_conversations', 'Number of active conversations')

# Cost metrics
LLM_COST = Counter('livekit_llm_cost_total', 'Total LLM cost in USD', ['model'])
STT_COST = Counter('livekit_stt_cost_total', 'Total STT cost in USD', ['provider'])
TTS_COST = Counter('livekit_tts_cost_total', 'Total TTS cost in USD', ['provider'])
TOTAL_COST = Counter('livekit_total_cost_total', 'Total cost in USD')

# Initialize metrics with default values
def initialize_metrics():
    try:
        # Initialize latency metrics with default labels
        LLM_LATENCY.labels(model='llama-3.3-70b').observe(0)
        STT_LATENCY.labels(provider='deepgram').observe(0)
        TTS_LATENCY.labels(provider='openai').observe(0)
        EOU_LATENCY.observe(0)
        TOTAL_CONVERSATION_LATENCY.observe(0)
        
        # Initialize token counters
        LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b').inc(0)
        LLM_TOKENS.labels(type='completion', model='llama-3.3-70b').inc(0)
        STT_DURATION.labels(provider='deepgram').inc(0)
        TTS_CHARS.labels(provider='openai').inc(0)
        TOTAL_TOKENS.inc(0)
        CONVERSATION_TURNS.inc(0)
        
        # Initialize cost metrics
        LLM_COST.labels(model='llama-3.3-70b').inc(0)
        STT_COST.labels(provider='deepgram').inc(0)
        TTS_COST.labels(provider='openai').inc(0)
        TOTAL_COST.inc(0)
        
        logger.info("Successfully initialized all metrics with default values")
    except Exception as e:
        logger.error(f"Error initializing metrics: {e}")

class PreResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant",
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=openai.TTS(voice="nova")
        )
        self._fast_llm = groq.LLM(model="llama-3.1-8b-instant")
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Generate a very short instant response to the user's message with 5 to 10 words.",
                "Do not answer the questions directly. Examples: OK, Hm..., let me think about that, "
                "wait a moment, that's a good question, etc.",
            ],
        )

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Initialize metrics at startup
    initialize_metrics()

    # Store component latencies for total latency calculation
    current_turn_metrics = {
        'eou_delay': None,
        'llm_ttft': None,
        'tts_ttfb': None
    }

    def calculate_total_latency():
        if all(v is not None for v in current_turn_metrics.values()):
            # Total latency calculation breakdown (time it takes for the agent to respond to a user's utterance):
            # 1. eou_delay: Time from user stops speaking to end-of-utterance detection. This includes transcription_delay
            # 2. llm_ttft: Time to first token from LLM (Time To First Token)
            # 3. tts_ttfb: Time to first byte from TTS (Time To First Byte)
            total_ms = (current_turn_metrics['eou_delay'] + current_turn_metrics['llm_ttft'] + current_turn_metrics['tts_ttfb']) * 1000
            
            # Log individual components for debugging
            logger.debug(f"Latency components (ms): EOU={current_turn_metrics['eou_delay']*1000:.2f}, "
                        f"LLM={current_turn_metrics['llm_ttft']*1000:.2f}, "
                        f"TTS={current_turn_metrics['tts_ttfb']*1000:.2f}")
            
            try:
                # Log previous metric values
                prev_count = TOTAL_CONVERSATION_LATENCY._count.get()
                prev_sum = TOTAL_CONVERSATION_LATENCY._sum.get()
                
                # Ensure the metric is properly observed
                TOTAL_CONVERSATION_LATENCY.observe(total_ms)
                
                # Log metric changes
                new_count = TOTAL_CONVERSATION_LATENCY._count.get()
                new_sum = TOTAL_CONVERSATION_LATENCY._sum.get()
                logger.info(
                    "Updated total conversation latency metric",
                    extra={
                        "previous_count": prev_count,
                        "new_count": new_count,
                        "previous_sum": prev_sum,
                        "new_sum": new_sum,
                        "current_value_ms": total_ms,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error observing latency metric: {e}")
            
            logger.info(
                "Total Conversation Latency",
                extra={
                    "total_latency_ms": total_ms,
                    "eou_delay_ms": current_turn_metrics['eou_delay'] * 1000,
                    "llm_ttft_ms": current_turn_metrics['llm_ttft'] * 1000,
                    "tts_ttfb_ms": current_turn_metrics['tts_ttfb'] * 1000,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            # Reset metrics for next turn
            for k in current_turn_metrics:
                current_turn_metrics[k] = None

    session = AgentSession(
        turn_detection=EnglishModel(),
        stt=deepgram.STT(),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
    )
    
    usage_collector = metrics.UsageCollector()
    ACTIVE_CONVERSATIONS.inc()
    logger.info("Session initialized with metrics collector")

    @session.on("metrics_collected")
    def handle_metrics(ev: MetricsCollectedEvent):
        # Log all metrics
        metrics.log_metrics(ev.metrics)
        
        # Collect usage metrics first
        try:
            usage_collector.collect(ev.metrics)
            logger.debug(f"Usage metrics collected: {ev.metrics}")
            
            # Log current usage summary after each collection
            try:
                current_summary = usage_collector.get_summary()
                logger.debug(f"Current usage summary: {current_summary}")
                
                # Update Prometheus metrics with logging
                if hasattr(current_summary, 'llm_prompt_tokens'):
                    prev_value = LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b')._value.get()
                    LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b').inc(current_summary.llm_prompt_tokens)
                    logger.info(f"Updated LLM prompt tokens: {prev_value} -> {prev_value + current_summary.llm_prompt_tokens}")
                
                if hasattr(current_summary, 'llm_completion_tokens'):
                    prev_value = LLM_TOKENS.labels(type='completion', model='llama-3.3-70b')._value.get()
                    LLM_TOKENS.labels(type='completion', model='llama-3.3-70b').inc(current_summary.llm_completion_tokens)
                    logger.info(f"Updated LLM completion tokens: {prev_value} -> {prev_value + current_summary.llm_completion_tokens}")
                
                if hasattr(current_summary, 'stt_audio_duration'):
                    prev_value = STT_DURATION.labels(provider='deepgram')._value.get()
                    STT_DURATION.labels(provider='deepgram').inc(current_summary.stt_audio_duration)
                    logger.info(f"Updated STT duration: {prev_value} -> {prev_value + current_summary.stt_audio_duration}")
                
                if hasattr(current_summary, 'tts_characters_count'):
                    prev_value = TTS_CHARS.labels(provider='openai')._value.get()
                    TTS_CHARS.labels(provider='openai').inc(current_summary.tts_characters_count)
                    logger.info(f"Updated TTS characters: {prev_value} -> {prev_value + current_summary.tts_characters_count}")
                
                # Calculate and update costs with logging
                llm_cost = (getattr(current_summary, 'llm_prompt_tokens', 0) * 0.00001 + 
                           getattr(current_summary, 'llm_completion_tokens', 0) * 0.00003)
                stt_cost = getattr(current_summary, 'stt_audio_duration', 0) * 0.0001
                tts_cost = getattr(current_summary, 'tts_characters_count', 0) * 0.000015
                
                prev_llm_cost = LLM_COST.labels(model='llama-3.3-70b')._value.get()
                prev_stt_cost = STT_COST.labels(provider='deepgram')._value.get()
                prev_tts_cost = TTS_COST.labels(provider='openai')._value.get()
                
                LLM_COST.labels(model='llama-3.3-70b').inc(llm_cost)
                STT_COST.labels(provider='deepgram').inc(stt_cost)
                TTS_COST.labels(provider='openai').inc(tts_cost)
                TOTAL_COST.inc(llm_cost + stt_cost + tts_cost)
                
                logger.info(
                    "Updated cost metrics",
                    extra={
                        "llm_cost": f"{prev_llm_cost} -> {prev_llm_cost + llm_cost}",
                        "stt_cost": f"{prev_stt_cost} -> {prev_stt_cost + stt_cost}",
                        "tts_cost": f"{prev_tts_cost} -> {prev_tts_cost + tts_cost}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
        except Exception as e:
            logger.error(f"Error collecting usage metrics: {e}")
        
        # Track metrics based on their type
        if isinstance(ev.metrics, LLMMetrics):
            logger.debug(f"Processing LLM metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                duration_ms = ev.metrics.duration * 1000  # Convert to ms
                LLM_LATENCY.labels(model='llama-3.3-70b').observe(duration_ms)
                logger.debug(f"Observed LLM latency: {duration_ms}ms")
            if hasattr(ev.metrics, 'ttft'):
                current_turn_metrics['llm_ttft'] = ev.metrics.ttft
                calculate_total_latency()
            if hasattr(ev.metrics, 'total_tokens'):
                TOTAL_TOKENS.inc(ev.metrics.total_tokens)
                logger.info(
                    "LLM Metrics",
                    extra={
                        "latency_ms": getattr(ev.metrics, 'duration', 0) * 1000,
                        "total_tokens": ev.metrics.total_tokens,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        elif isinstance(ev.metrics, STTMetrics):
            logger.debug(f"Processing STT metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                duration_ms = ev.metrics.duration * 1000  # Convert to ms
                STT_LATENCY.labels(provider='deepgram').observe(duration_ms)
                logger.debug(f"Observed STT latency: {duration_ms}ms")
                logger.info(
                    "STT Metrics",
                    extra={
                        "latency_ms": duration_ms,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        elif isinstance(ev.metrics, TTSMetrics):
            logger.debug(f"Processing TTS metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                duration_ms = ev.metrics.duration * 1000  # Convert to ms
                TTS_LATENCY.labels(provider='openai').observe(duration_ms)
                logger.debug(f"Observed TTS latency: {duration_ms}ms")
            if hasattr(ev.metrics, 'ttfb'):
                current_turn_metrics['tts_ttfb'] = ev.metrics.ttfb
                calculate_total_latency()
            logger.info(
                "TTS Metrics",
                extra={
                    "latency_ms": getattr(ev.metrics, 'duration', 0) * 1000,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        elif isinstance(ev.metrics, VADMetrics):
            logger.debug(f"Processing VAD metrics: {ev.metrics}")
            # Log VAD metrics without assuming specific attributes
            logger.info(
                "VAD Metrics",
                extra={
                    "metrics": str(ev.metrics),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        elif isinstance(ev.metrics, EOUMetrics):
            logger.debug(f"Processing EOU metrics: {ev.metrics}")
            # Convert seconds to milliseconds for consistency with other metrics
            if hasattr(ev.metrics, 'end_of_utterance_delay'):
                delay_ms = ev.metrics.end_of_utterance_delay * 1000
                EOU_LATENCY.observe(delay_ms)
                logger.debug(f"Observed EOU delay: {delay_ms}ms")
                current_turn_metrics['eou_delay'] = ev.metrics.end_of_utterance_delay
                calculate_total_latency()
            
            logger.info(
                "EOU Metrics",
                extra={
                    "end_of_utterance_delay": getattr(ev.metrics, 'end_of_utterance_delay', 0),
                    "transcription_delay": getattr(ev.metrics, 'transcription_delay', 0),
                    "on_user_turn_completed_delay": getattr(ev.metrics, 'on_user_turn_completed_delay', 0),
                    "speech_id": getattr(ev.metrics, 'speech_id', ''),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        else:
            logger.debug(f"Received unknown metrics type: {type(ev.metrics)}")

    async def log_usage():
        try:
            summary = usage_collector.get_summary()
            logger.debug(f"Final usage summary: {summary}")
            
            # Get base metrics
            llm_prompt_tokens = getattr(summary, 'llm_prompt_tokens', 0)
            llm_completion_tokens = getattr(summary, 'llm_completion_tokens', 0)
            stt_duration = getattr(summary, 'stt_audio_duration', 0)
            tts_chars = getattr(summary, 'tts_characters_count', 0)
            
            # Calculate totals and costs
            total_tokens = llm_prompt_tokens + llm_completion_tokens
            # Rough cost estimates (adjust these based on your actual pricing)
            llm_cost = (llm_prompt_tokens * 0.00001 + llm_completion_tokens * 0.00003)  # $0.01/1K tokens for input, $0.03/1K for output
            stt_cost = stt_duration * 0.0001  # $0.0001 per second
            tts_cost = tts_chars * 0.000015  # $0.000015 per character
            
            # Convert UsageSummary to a dictionary of its attributes
            summary_dict = {
                "llm": {
                    "prompt_tokens": llm_prompt_tokens,
                    "prompt_cached_tokens": getattr(summary, 'llm_prompt_cached_tokens', 0),
                    "completion_tokens": llm_completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost": round(llm_cost, 6)
                } if any(hasattr(summary, attr) for attr in ['llm_prompt_tokens', 'llm_completion_tokens']) else None,
                "stt": {
                    "audio_duration": stt_duration,
                    "estimated_cost": round(stt_cost, 6)
                } if hasattr(summary, 'stt_audio_duration') else None,
                "tts": {
                    "characters_count": tts_chars,
                    "estimated_cost": round(tts_cost, 6)
                } if hasattr(summary, 'tts_characters_count') else None,
                "totals": {
                    "total_tokens": total_tokens,
                    "total_cost": round(llm_cost + stt_cost + tts_cost, 6)
                }
            }
            
            logger.info(
                "Session Summary",
                extra={
                    "usage_summary": json.dumps(summary_dict),
                    "active_conversations": ACTIVE_CONVERSATIONS._value.get(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
        finally:
            ACTIVE_CONVERSATIONS.dec()

    # Register metrics handler before starting the session
    logger.info("Registering metrics handler")
    await session.start(PreResponseAgent(), room=ctx.room)
    ctx.add_shutdown_callback(log_usage)


if __name__ == "__main__":
    try:
        # Initialize metrics before starting the server
        initialize_metrics()
        
        # Start up the server to expose the metrics.
        logger.info("Starting Prometheus metrics server on port 9100...")
        start_http_server(9100, addr='0.0.0.0')
        logger.info("Prometheus metrics server started successfully")
        
        # Verify metrics are accessible and contain our custom metrics
        import requests
        try:
            response = requests.get('http://localhost:9100/metrics')
            if response.status_code == 200:
                logger.info("Successfully verified metrics endpoint is accessible")
                # Log available metrics
                metrics_list = [line for line in response.text.split('\n') if line and not line.startswith('#')]
                logger.info(f"Available metrics on endpoint: {len(metrics_list)}")
                
                # Check for our custom metrics
                custom_metrics = [m for m in metrics_list if m.startswith('livekit_')]
                logger.info(f"Found {len(custom_metrics)} custom metrics:")
                for metric in custom_metrics:
                    logger.info(f"Custom metric: {metric}")
                
                if not custom_metrics:
                    logger.error("No custom metrics found in the metrics endpoint!")
            else:
                logger.error(f"Metrics endpoint returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error accessing metrics endpoint: {e}")
        
        # Initialize the agent
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise