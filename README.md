# webrtc-agent-livekit

Build real-time voice AI agents powered by [LiveKit Agent](https://github.com/livekit/agents), Small Language Models (SLMs), and WebRTC.

This project is a quickstart template to run locally or with AWS integrations. It showcases how to combine WebRTC, LiveKit’s Agent framework, and open-source tools like Whisper and Llama to prototype low-latency voice assistants for real-time applications.

## 🧠 What’s Inside

- 🌐 **WebRTC + LiveKit**: Real-time media transport with WebRTC powered by LiveKit.
- 🤖 **LiveKit Agent**: Modular plugin-based framework for voice AI agents.
- 🗣️ **STT + TTS Support**: Plug in Whisper, Deepgram, ElevenLabs, or others.
- 💬 **LLM Integration**: Use local LLaMA models or connect to AWS/ OpenAI / Anthropic APIs.
- 🧪 **Local Dev**: Run everything locally with Docker Compose or Python virtual env.

---

## 🚀 Quick Start (Local)

1. Clone:
```bash
# Clone the repo
git clone https://github.com/your-org/webrtc-agent-livekit.git
cd webrtc-agent-livekit
```

2. Install dependencies docker and docker compose

3. If you want to also run the example frontend, copy and rename the `.env.example` file to `.env.local` and fill in the necessary environment variables. You can also update the YML files to configure the different services. For example, agents-playground.yml:

```
LIVEKIT_API_KEY=<your API KEY> #change it in livekit.yaml
LIVEKIT_API_SECRET=<Your API Secret> #change it in livekit.yaml
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880 #wss://<Your Cloud URL>
```

4. Run docker-compose:

```bash
  docker compose up --build
```
Make sure that at least the services "agent-playground", "agent-worker", "livekit" and "redis" in the docker-compose are uncommented.

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

6. Connect to a room

## Monitoring

The solution is build using Prometheus and grafana. The end to end flow is:
Agent worker → writes metrics to shared temp folder → agent_metrics exposes them → Prometheus scrapes them → Grafana displays them.

[Agent Worker](./agent-worker/) Metrics are exposed on port 9100 and can be accessed at:
```
http://localhost:9100/metrics
```

Grafana is available in [http://localhost:3001](http://localhost:3001) with default user/password: admin/admin
A default dashboard is setup to visualize basic real time voice agents information.