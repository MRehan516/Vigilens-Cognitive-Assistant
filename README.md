# Vigilens: Multimodal Cognitive Anchor for Dementia Care

## Overview
Vigilens is an ambient, real-time cognitive assistant designed to run on smart glasses for patients experiencing task sequencing failure and object agnosia. Unlike traditional text-based LLMs, Vigilens operates purely on low-latency voice and vision, acting as an empathetic, continuous caregiver.

## Technical Architecture
This prototype leverages the `google-genai` SDK and the `gemini-2.0-flash` model via a bidirectional WebSocket connection. 
* **Concurrency:** Implemented a custom `asyncio` event loop to manage blocking I/O operations and ensure heartbeats are maintained.
* **Multimodal Streaming:** Concurrently processes 16kHz PCM audio via PyAudio and encodes 1-FPS live video frames via OpenCV natively into base64 without starving the WebSocket keepalive ping.
* **Persona Engineering:** System instructions are strictly tuned for clinical empathy, concise responses, and graceful interruption handling.

## Running Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your API key: `GEMINI_API_KEY="your_api_key"`
4. Execute the engine: `python main.py`

## Production Roadmap
For deployment in noisy clinical environments, the next iteration will replace raw local audio handling with a WebRTC pipeline (AEC/ANS) and an edge-based Silero Voice Activity Detection (VAD) model to optimize token usage and prevent buffer overflow.

*Engineered by Mohd Rehan Arvi | Hackathon Submission 2026*
