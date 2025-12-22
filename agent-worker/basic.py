import logging
import os
import atexit
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent
)
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram, groq, openai, silero
from prometheus_client import (
    Counter, 
    Gauge, 
    CollectorRegistry, 
    multiprocess
)
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("basic-agent")

load_dotenv()

# Set up multiprocess mode
PROMETHEUS_MULTIPROC_DIR = '/tmp/prometheus_multiproc'
os.environ['prometheus_multiproc_dir'] = PROMETHEUS_MULTIPROC_DIR

# Get the current file name without extension
AGENT_TYPE = os.path.splitext(os.path.basename(__file__))[0]

# Create a shared registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
logger.info("Initialized multiprocess collector with shared registry")

# Define Prometheus metrics with multiprocess mode
LLM_LATENCY = Gauge('livekit_llm_duration_ms', 'LLM latency in milliseconds', ['model', 'agent_type'], registry=registry)
STT_LATENCY = Gauge('livekit_stt_duration_ms', 'Speech-to-text latency in milliseconds', ['provider', 'agent_type'], registry=registry)
TTS_LATENCY = Gauge('livekit_tts_duration_ms', 'Text-to-speech latency in milliseconds', ['provider', 'agent_type'], registry=registry)
EOU_LATENCY = Gauge('livekit_eou_delay_ms', 'End-of-utterance delay in milliseconds', ['agent_type'], registry=registry)
TOTAL_CONVERSATION_LATENCY = Gauge('livekit_total_conversation_latency_ms', 'Current conversation latency in milliseconds', ['agent_type'], registry=registry)

# Usage metrics with multiprocess mode
LLM_TOKENS = Counter('livekit_llm_tokens_total', 'Total LLM tokens processed', ['type', 'model'], registry=registry)
STT_DURATION = Counter('livekit_stt_duration_seconds_total', 'Total STT audio duration in seconds', ['provider'], registry=registry)
TTS_CHARS = Counter('livekit_tts_chars_total', 'Total TTS characters processed', ['provider'], registry=registry)
TOTAL_TOKENS = Counter('livekit_total_tokens_total', 'Total tokens processed', registry=registry)
CONVERSATION_TURNS = Counter('livekit_conversation_turns_total', 'Number of conversation turns', ['agent_type', 'room'], registry=registry)
ACTIVE_CONVERSATIONS = Gauge('livekit_active_conversations', 'Number of active conversations', ['agent_type'], multiprocess_mode='liveall', registry=registry)

# Cost metrics with multiprocess mode
LLM_COST = Gauge('livekit_llm_cost_total', 'Total LLM cost in USD', ['model'], registry=registry)
STT_COST = Gauge('livekit_stt_cost_total', 'Total STT cost in USD', ['provider'], registry=registry)
TTS_COST = Gauge('livekit_tts_cost_total', 'Total TTS cost in USD', ['provider'], registry=registry)
# Configure multiprocess mode for usage counters
for metric in [LLM_TOKENS, STT_DURATION, TTS_CHARS, TOTAL_TOKENS, CONVERSATION_TURNS]:
    metric._multiprocess_mode = 'livesum'
    logger.debug(f"Configured multiprocess mode for counter: {metric._name}")
# Configure cost metrics to use liveall mode for single aggregated value
for metric in [LLM_COST, STT_COST, TTS_COST]:
    metric._multiprocess_mode = 'liveall'
    logger.debug(f"Configured multiprocess mode for cost metric: {metric._name}")

# Initialize metrics with default values
def initialize_metrics():
    try:
        logger.info("Starting metrics initialization...")
        
        # Initialize latency metrics with default labels
        LLM_LATENCY.labels(model='llama-3.3-70b', agent_type=AGENT_TYPE).set(0)
        STT_LATENCY.labels(provider='deepgram', agent_type=AGENT_TYPE).set(0)
        TTS_LATENCY.labels(provider='openai', agent_type=AGENT_TYPE).set(0)
        EOU_LATENCY.labels(agent_type=AGENT_TYPE).set(0)
        TOTAL_CONVERSATION_LATENCY.labels(agent_type=AGENT_TYPE).set(0)
        
        # Initialize token counters (important for multiprocess mode)
        LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b').inc(0)
        LLM_TOKENS.labels(type='completion', model='llama-3.3-70b').inc(0)
        STT_DURATION.labels(provider='deepgram').inc(0)
        TTS_CHARS.labels(provider='openai').inc(0)
        TOTAL_TOKENS.inc(0)
        
        # Initialize cost metrics
        LLM_COST.labels(model='llama-3.3-70b').inc(0)
        STT_COST.labels(provider='deepgram').inc(0)
        TTS_COST.labels(provider='openai').inc(0)
        
        logger.info("Successfully initialized all metrics with default values")
    except Exception as e:
        logger.error(f"Error initializing metrics: {e}")
        raise

class BasicAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant. Always respond concisely in less than 6 sentences.",
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
        )

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Extract room from ctx (fallback to 'unknown' if not present)
    room = getattr(ctx, 'room', None) or 'unknown'

    # Store component latencies for total latency calculation
    current_turn_metrics = {
        'turn_id': None,  # Unique ID for each turn
        'eou_delay': None,
        'llm_ttft': None,
        'tts_ttfb': None,
        'room': room
    }
    turn_id_counter = 0  # Simple incrementing counter for turn IDs

    def start_new_turn():
        nonlocal turn_id_counter
        turn_id_counter += 1
        current_turn_metrics['turn_id'] = turn_id_counter
        current_turn_metrics['eou_delay'] = None
        current_turn_metrics['llm_ttft'] = None
        current_turn_metrics['tts_ttfb'] = None
        CONVERSATION_TURNS.labels(agent_type=AGENT_TYPE, room=room).inc()
        return turn_id_counter

    def calculate_total_latency():
        if all(current_turn_metrics[k] is not None for k in ['eou_delay', 'llm_ttft', 'tts_ttfb']):
            total_ms = int((current_turn_metrics['eou_delay'] + 
                          current_turn_metrics['llm_ttft'] + 
                          current_turn_metrics['tts_ttfb']) * 1000)
            TOTAL_CONVERSATION_LATENCY.labels(agent_type=AGENT_TYPE).set(total_ms)
            logger.info(f"Total latency: {total_ms}ms")
            start_new_turn()

    session = AgentSession(
        # turn_detection=EnglishModel(),  # Turn detection disabled
        stt=deepgram.STTv2(
            model="flux-general-en",  # Using Deepgram Flux model
            eager_eot_threshold=0.5,
        ),
        # stt=deepgram.STT(),
        tts=groq.TTS(model="playai-tts", voice="Arista-PlayAI"),
        # tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(min_silence_duration=0.3, activation_threshold=0.4),
        preemptive_generation=True,
    )
    
    usage_collector = metrics.UsageCollector()
    ACTIVE_CONVERSATIONS.labels(agent_type=AGENT_TYPE).inc()
    atexit.register(lambda: ACTIVE_CONVERSATIONS.labels(agent_type=AGENT_TYPE).dec())
    logger.info("Session initialized with metrics collector")

    @session.on("metrics_collected")
    def handle_metrics(ev: MetricsCollectedEvent):
        # Determine the turn_id for this metric event
        # If the event has a turn/session ID, use it; otherwise, use a new turn for EOU metrics (start of user turn)
        event_turn_id = getattr(ev.metrics, 'turn_id', None)
        if isinstance(ev.metrics, EOUMetrics):
            # Start a new turn for each EOU metric (end of user utterance)
            event_turn_id = start_new_turn()
        elif current_turn_metrics['turn_id'] is None:
            # If no turn is active, start one
            event_turn_id = start_new_turn()
        else:
            event_turn_id = current_turn_metrics['turn_id']

        # If the event's turn_id does not match the current, reset state (only if event has a turn_id)
        if event_turn_id is not None and event_turn_id != current_turn_metrics['turn_id']:
            logger.warning(f"Metric event turn_id {event_turn_id} does not match current turn_id {current_turn_metrics['turn_id']}. Resetting state.")
            start_new_turn()

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
                
                # Get current values from metrics (for multiprocess mode)
                current_prompt_tokens = LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b')._value.get() or 0
                current_completion_tokens = LLM_TOKENS.labels(type='completion', model='llama-3.3-70b')._value.get() or 0
                current_stt_duration = STT_DURATION.labels(provider='deepgram')._value.get() or 0
                current_tts_chars = TTS_CHARS.labels(provider='openai')._value.get() or 0
                
                # Update Prometheus metrics with logging
                if hasattr(current_summary, 'llm_prompt_tokens'):
                    new_prompt_tokens = current_summary.llm_prompt_tokens
                    if new_prompt_tokens > current_prompt_tokens:
                        LLM_TOKENS.labels(type='prompt', model='llama-3.3-70b').inc(new_prompt_tokens - current_prompt_tokens)
                
                if hasattr(current_summary, 'llm_completion_tokens'):
                    new_completion_tokens = current_summary.llm_completion_tokens
                    if new_completion_tokens > current_completion_tokens:
                        LLM_TOKENS.labels(type='completion', model='llama-3.3-70b').inc(new_completion_tokens - current_completion_tokens)
                
                if hasattr(current_summary, 'stt_audio_duration'):
                    new_stt_duration = current_summary.stt_audio_duration
                    if new_stt_duration > current_stt_duration:
                        STT_DURATION.labels(provider='deepgram').inc(new_stt_duration - current_stt_duration)
                
                if hasattr(current_summary, 'tts_characters_count'):
                    new_tts_chars = current_summary.tts_characters_count
                    if new_tts_chars > current_tts_chars:
                        TTS_CHARS.labels(provider='openai').inc(new_tts_chars - current_tts_chars)
                
                # Calculate costs from current summary values
                llm_cost = (getattr(current_summary, 'llm_prompt_tokens', 0) * 0.00001 +  # $0.01 per 1K input tokens
                           getattr(current_summary, 'llm_completion_tokens', 0) * 0.00003)  # $0.03 per 1K output tokens
                stt_cost = getattr(current_summary, 'stt_audio_duration', 0) * 0.0001  # $0.0001 per second
                tts_cost = getattr(current_summary, 'tts_characters_count', 0) * 0.000015  # $0.000015 per character
                
                # Update cost metrics (these are Gauges, so set() is fine)
                LLM_COST.labels(model='llama-3.3-70b').set(llm_cost)
                STT_COST.labels(provider='deepgram').set(stt_cost)
                TTS_COST.labels(provider='openai').set(tts_cost)
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
        except Exception as e:
            logger.error(f"Error collecting usage metrics: {e}")
        
        # Track metrics based on their type
        # Log the actual type to help debug
        metric_type = type(ev.metrics).__name__
        logger.debug(f"Received metrics type: {metric_type}, isinstance checks: LLM={isinstance(ev.metrics, LLMMetrics)}, STT={isinstance(ev.metrics, STTMetrics)}, TTS={isinstance(ev.metrics, TTSMetrics)}, EOU={isinstance(ev.metrics, EOUMetrics)}")
        
        if isinstance(ev.metrics, LLMMetrics):
            logger.debug(f"Processing LLM metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                # The amount of time (seconds) it took for the LLM to generate the entire completion.
                duration_ms = ev.metrics.duration * 1000  # Convert to ms
                logger.debug(f"Observed LLM response generation latency: {duration_ms}ms")
            if hasattr(ev.metrics, 'ttft'):
                current_turn_metrics['llm_ttft'] = ev.metrics.ttft
                LLM_LATENCY.labels(model='llama-3.3-70b', agent_type=AGENT_TYPE).set(current_turn_metrics['llm_ttft']*1000)
                logger.info(f"Updated LLM latency: {current_turn_metrics['llm_ttft']*1000}ms")
                calculate_total_latency()
            if hasattr(ev.metrics, 'total_tokens'):
                TOTAL_TOKENS.inc(ev.metrics.total_tokens)
        
        elif isinstance(ev.metrics, STTMetrics):
            logger.debug(f"Processing STT metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                duration_ms = ev.metrics.duration * 1000  # Convert to ms but will be 0 for streaming STT. This latency is counted in the end_of_utterance_delay
                STT_LATENCY.labels(provider='deepgram', agent_type=AGENT_TYPE).set(duration_ms)
                logger.debug(f"Observed STT latencyto generate transcript: {duration_ms}ms")
        
        elif isinstance(ev.metrics, TTSMetrics):
            logger.debug(f"Processing TTS metrics: {ev.metrics}")
            if hasattr(ev.metrics, 'duration'):
                duration_ms = ev.metrics.duration * 1000  # Convert to ms
                # The amount of time (seconds) it took for the TTS model to generate the entire audio output.
                logger.debug(f"Observed TTS latency: {duration_ms}ms")
            if hasattr(ev.metrics, 'ttfb'):
                current_turn_metrics['tts_ttfb'] = ev.metrics.ttfb
                TTS_LATENCY.labels(provider='openai', agent_type=AGENT_TYPE).set(current_turn_metrics['tts_ttfb']*1000)
                logger.info(f"Updated TTS latency: {current_turn_metrics['tts_ttfb']*1000}ms")
                calculate_total_latency()
        
        elif isinstance(ev.metrics, EOUMetrics):
            logger.debug(f"Processing EOU metrics: {ev.metrics}")
            # Convert seconds to milliseconds for consistency with other metrics
            if hasattr(ev.metrics, 'end_of_utterance_delay'):
                delay_ms = ev.metrics.end_of_utterance_delay * 1000
                EOU_LATENCY.labels(agent_type=AGENT_TYPE).set(delay_ms)
                logger.debug(f"Observed EOU delay: {delay_ms}ms")
                current_turn_metrics['eou_delay'] = ev.metrics.end_of_utterance_delay
                logger.info(f"Updated EOU delay: {delay_ms}ms")
                calculate_total_latency()
        else:
            logger.debug(f"Received unknown metrics type: {type(ev.metrics)}")

    async def log_usage():
        try:
            summary = usage_collector.get_summary()
            logger.info(f"Session summary: {summary}")
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
        finally:
            ACTIVE_CONVERSATIONS.labels(agent_type=AGENT_TYPE).dec()

    await session.start(BasicAgent(), room=ctx.room)
    ctx.add_shutdown_callback(log_usage)


async def prewarm(proc: JobContext):
    """Download model files before starting the agent."""
    logger.info("Prewarming: downloading model files...")
    # Download turn detector model
    await EnglishModel.download_files()
    logger.info("Model files downloaded successfully")


if __name__ == "__main__":
    initialize_metrics()
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm
    ))