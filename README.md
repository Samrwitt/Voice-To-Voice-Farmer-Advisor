# Voice-to-Voice Farmer Advisor: Technical Architecture

A production-ready, full-duplex conversational AI system designed to provide real-time agricultural advice to Amharic-speaking farmers. This system utilizes a modular microservice architecture to handle low-latency voice processing and high-accuracy knowledge retrieval.

## 🚀 Performance & Accuracy Overview

| Component | Technology | Speed (Latency) | Accuracy / Quality |
| :--- | :--- | :--- | :--- |
| **VAD** | Silero VAD v5 | < 10ms (Real-time) | 99% Speech Detection |
| **ASR** | Whisper (Fine-tuned Amharic) | 1.2s - 2.5s (Short utt) | ~35% WER (Telephone 8kHz) |
| **RAG** | pgvector + MiniLM-L12 | ~450ms (Retrieval) | High (Domain Specific) |
| **TTS** | gTTS / edge-tts | 2.0s - 4.0s (Synthesis) | Natural Amharic Prosody |
| **Streaming** | WebSocket (PCM16 Binary) | < 50ms (Network Jitter) | Lossless Audio Delivery |

---

ASR Accuracy Note

The current ASR achieves approximately 35% Word Error Rate (WER) on challenging 8kHz telephone-bandwidth Amharic speech.

The relatively high WER is influenced by both acoustic degradation and Amharic-specific language characteristics, including:

Homophones and phonetically similar words, which can be difficult to distinguish from audio alone.
Rich morphology, where a single root can generate many inflected word forms.
Dialect and pronunciation variation across speakers and regions.
Orthographic variation and differences between spoken and written forms.
Code-switching between Amharic, English, and other local languages.
Limited large-scale Amharic speech datasets compared with high-resource languages.
Telephone-quality audio, background noise, microphone variation, and real-world field conditions.

Because raw WER does not always represent whether the system understood the farmer correctly, the project also evaluates semantic accuracy, intent recognition, agricultural entity extraction, RAG retrieval quality, and end-to-end task success.

For example, an ASR transcription may contain individual word errors while still correctly identifying the farmer's crop, symptom, disease, pest, or requested action. The downstream semantic correction and RAG layers are designed to recover this context before generating agricultural advice.

## 🛠 Toolchain & Microservices

### 1. Voice Activity Detection (VAD Service)
*   **Engine**: Silero VAD (State-of-the-Art Neural VAD).
*   **Role**: Acts as the "ears" of the system. It monitors the raw audio stream, identifies speech boundaries, and handles the full-duplex logic.
*   **Optimization**: Implements a circular buffer and "Speech Pad" (200ms) to ensure the start of sentences isn't clipped.

### 2. Automatic Speech Recognition (ASR Service)
*   **Engine**: `faster-whisper` (CTranslate2 optimized).
*   **Model**: Fine-tuned Whisper-Small for Amharic.
*   **Accuracy**: Optimized for telephone-bandwidth (8kHz) audio with data augmentation to handle noisy field environments.
*   **Semantic Correction**: Uses a hosted LLM layer (Gemini/Groq) to correct transcription hallucinations in real-time.

### 3. Retrieval-Augmented Generation (RAG Service)
*   **Engine**: FastAPI + pgvector (PostgreSQL).
*   **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`.
*   **Knowledge Base**: Multi-turn diagnostic trees for Amharic agricultural queries (Soil acidity, pest control, crop rotation).
*   **Logic**: Automatically escalates low-confidence queries to human experts via the PostgreSQL `escalations` table.

### 4. Text-to-Speech (TTS Service)
*   **Engine**: gTTS (Google) with fallback to local providers.
*   **Format**: Returns 16000Hz PCM16 Mono WAV.
*   **Streaming Strategy**: Sentences are split and streamed as soon as they are ready, reducing perceived latency to < 1s.

### 5. Telephony Gateway & Frontend
*   **Gateway**: FastAPI WebSocket proxy that routes control events (JSON) and media data (Binary) between the browser and the VAD service.
*   **Frontend**: Vanilla JS with **Web Audio API**. Uses a jitter-buffered `AudioWorklet`-like playback queue for seamless binary PCM streaming.
*   **Security**: Implements user-gesture based `AudioContext` priming to bypass modern browser audio blocks.

---

## 🏗 Deployment & Scaling
The system is fully containerized using **Docker Compose**.
*   **Full-Duplex**: The WebSocket connection remains open for the duration of the call, allowing the user to interrupt the advisor (Barge-in support).
*   **Shared Storage**: Uses Docker Volumes for high-speed file access between ASR and VAD services.
*   **Resilience**: Implements `safe_send` locking to prevent race conditions during concurrent audio/event transmission.

## 📊 Evaluation
The system was benchmarked using real-world agricultural queries. The modular architecture allows for independent scaling—for example, upgrading the ASR model without affecting the RAG logic.
