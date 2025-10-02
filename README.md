# webrtc-agent-livekit

Build real-time voice AI agents powered by [LiveKit Agent](https://github.com/livekit/agents), Small Language Models (SLMs), and WebRTC.

This project is a quickstart template to run locally or with 3rd party integrations. It showcases how to combine WebRTC, LiveKit’s Agent framework, and open-source tools like Whisper and Llama to prototype low-latency voice assistants for real-time applications.

## 🧠 What’s Inside

- 🌐 **WebRTC + LiveKit**: Real-time media transport with WebRTC powered by LiveKit.
- 🤖 **LiveKit Agent**: Modular plugin-based framework for voice AI agents.
- 🗣️ **STT + TTS Support**: Plug in Whisper, Deepgram, ElevenLabs, or others.
- 💬 **LLM Integration**: Use local LLaMA models or connect to AWS/ OpenAI / Anthropic APIs.
- 🧪 **Local Dev**: Run everything locally with Docker Compose or Python virtual env.

THERE ARE 2 IMPLEMENTATIONS OF THE AI AGENT:
- [fast-preresponse.py](./agent-worker/fast-preresponse.py) using 3rd party services and the complete metrics capture in place.
- [fast-preresponse-ollama.py](./agent-worker/fast-preresponse-ollama.py) which is only using open source souftware and can run locally without internet.

Just update [Dockerfile](./agent-worker/Dockerfile) to use one or another. More info [here](./agent-worker/README.md).

---

## 🚀 Quick Start (Local)

1. Clone:
```bash
# Clone the repo
git clone https://github.com/agonza1/webrtc-agent-livekit.git
cd webrtc-agent-livekit
```

2. Install dependencies docker and docker compose

3. If you want to also run the example frontend, copy and rename the [`.env.example`](./agents-playground/.env.example) file to `.env.local` and fill in the necessary environment variables. You can also update the YML files to configure the different services. For example, agents-playground.yml:

```
LIVEKIT_API_KEY=<your API KEY> #change it in livekit.yaml
LIVEKIT_API_SECRET=<Your API Secret> #change it in livekit.yaml
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880 #wss://<Your Cloud URL>
```

4. Run docker-compose:

```bash
  docker compose up --build
```
Make sure that at least the services "agent-playground", "agent-worker", "livekit" and "redis" in the docker-compose are uncommented and the envs are updated.

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

6. Connect to a room

## Monitoring

The solution uses both **Prometheus/Grafana** and **PeerMetrics** for comprehensive WebRTC monitoring.

### Prometheus & Grafana
The traditional monitoring stack with end-to-end flow:
Agent worker → writes metrics to shared temp folder → agent_metrics exposes them → Prometheus scrapes them → Grafana displays them.

[Agent Worker](./agent-worker/) live metrics are exposed on port 9100 and can be accessed at:
```
http://localhost:9100/metrics
```

Grafana is available in [http://localhost:3001](http://localhost:3001) with default user/password: admin/admin
A default dashboard is setup to visualize basic real time voice agents information.

### PeerMetrics WebRTC Analytics
**PeerMetrics** provides specialized WebRTC monitoring and analytics, tracking connection quality, media performance, and network statistics.

**Setup Requirements:**
Before using PeerMetrics, you need to run database migrations for both services. The migrations must be run in the correct order:

**API Migrations (first time only)**
```bash
# Start a shell in the API container
docker compose run peermetrics-api sh

# Inside the container, create models
python manage.py makemigrations app

# Run PeerMetrics app migrations
python manage.py migrate app

# Exit the container
exit
```

**Access Points:**
- **PeerMetrics API**: [http://localhost:3002](http://localhost:3002) - API endpoint for metrics collection
- **PeerMetrics Dashboard**: [http://localhost:3003](http://localhost:3003) - Web interface for analytics

**Configuration:**
PeerMetrics integration is configured in [src/config/peerMetrics.ts](./agents-playground/src/config/peerMetrics.ts) and automatically tracks:
- Connection quality metrics (RTT, packet loss, jitter)
- Media performance (audio/video bitrates, resolution, frame rates)
- Network statistics (ICE connection state, candidate pairs)
- User events (mute/unmute, page visibility changes)

For detailed setup instructions, see [PEERMETRICS_SETUP.md](./agents-playground/PEERMETRICS_SETUP.md).

## 🙏 Credits

This project is built on top of amazing open-source tools and services:

- **[LiveKit](https://livekit.io/) and [LiveKit Agents](https://github.com/livekit/agents)** - WebRTC Framework for building voice AI agents
- **[Ollama](https://ollama.ai)** - Local LLM inference engine
- **[Llama](https://llama.meta.com/)** - Open-source large language models by Meta
- **[Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)** - Open-source text-to-speech model
- **[PeerMetrics](https://github.com/peermetrics)** - WebRTC monitoring and analytics platform
- **[Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)** - Metrics collection, monitoring and visualization