# agent-worker

A real-time voice agent implementation using LiveKit, featuring fast pre-response capabilities and comprehensive metrics collection. Here we implement the logic for the voice AI worker to explore different functions:
- fast-preresponse.py

This example uses Llama 3.1 8B and 70B. Initial quick response comes from 8B to optimize latency, and then the 70B takes over to handle the complex task.
- fast-preresponse-ollama.py

Same idea but using Ollama instead of Groq

## Architecture

```mermaid
graph TD
    subgraph "Agent Worker"
        A[Entrypoint] --> B[AgentSession]
        B <--> C[PreResponseAgent]
        
        subgraph "Metrics Collection"
            D[Metrics Collector] --> E[Prometheus Metrics]
            D --> F[Usage Summary]
        end
        
        subgraph "Components"
            G[STT/Deepgram] --> B
            H[TTS/OpenAI] --> B
            I[VAD/Silero] --> B
            J[LLM/Groq] --> B
        end
        subgraph "Fast Pre-Response"
            K[Fast LLM] --> L[Instant Response]
        end
        subgraph "Metrics Flow"
            O[LLM Metrics] --> D
            P[STT Metrics] --> D
            Q[TTS Metrics] --> D
            R[VAD Metrics] --> D
            S[EOU Metrics] --> D
        end
    end
    U[User] <-->|WebRTC audio| M
    subgraph "External Services"
        M[LiveKit Room] <--> B
        N[Prometheus Server] --> E
    end
```

![LiveKit Agent Architecture Diagram](assets/ArchitectureDiagram.png)

## Components

- **PreResponseAgent**: Main agent implementation with fast pre-response capabilities
- **Metrics Collection**: Comprehensive metrics tracking for:
  - LLM usage and latency
  - STT processing
  - TTS generation
  - VAD detection
  - End-of-utterance detection
  - Cost tracking
- **Fast Pre-Response**: Quick acknowledgment system using a smaller LLM model

## Metrics

The agent collects and exposes the following metrics:

- **Latency Metrics**:
  - LLM processing time
  - STT processing time
  - TTS generation time
  - End-of-utterance delay
  - Total conversation latency

- **Usage Metrics**:
  - LLM tokens (prompt and completion)
  - STT audio duration
  - TTS character count
  - Total tokens processed

- **Cost Metrics**:
  - LLM usage costs
  - STT processing costs
  - TTS generation costs
  - Total costs