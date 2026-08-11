# Voice-to-Voice Farmer Advisor

A production-oriented, full-duplex conversational AI system designed to provide real-time agricultural guidance to Amharic-speaking farmers.

The platform combines streaming audio processing, domain-adapted Amharic speech recognition, retrieval-augmented generation, conversational reasoning, and speech synthesis in a modular microservice architecture.

It is designed for challenging real-world conditions such as telephone-bandwidth audio, background noise, regional pronunciation variation, limited connectivity, and domain-specific agricultural vocabulary.

---

## 🚀 System Overview

The system processes a farmer's spoken question through the following pipeline:

```text
Farmer
  │
  ▼
Web / Voice Client
  │
  ▼
WebSocket Gateway
  │
  ▼
Voice Activity Detection
  │
  ▼
Amharic ASR
  │
  ▼
Context-Aware Transcript Normalization
  │
  ▼
Retrieval-Augmented Generation
  │
  ▼
Agricultural Response Generation
  │
  ▼
Text-to-Speech
  │
  ▼
Streaming Voice Response
  │
  └────────────── Barge-in / Interruption ──────────────┐
                                                        │
                                                        ▼
                                                   VAD / Gateway
```

The connection remains active throughout the conversation, allowing farmers to interrupt the generated response and continue speaking naturally.

---

## 🚀 Performance & Evaluation Overview

| Component     | Technology                         | Typical Latency           | Evaluation                           |
| :------------ | :--------------------------------- | :------------------------ | :----------------------------------- |
| **VAD**       | Silero VAD v5                      | <10 ms/frame              | Low-latency speech segmentation      |
| **ASR**       | Fine-tuned Whisper-Small           | 1.2–2.5 s/short utterance | ~35% WER on challenging 8 kHz speech |
| **RAG**       | PostgreSQL + pgvector + MiniLM-L12 | ~450 ms                   | Domain-specific semantic retrieval   |
| **TTS**       | gTTS / edge-tts                    | 2–4 s synthesis           | Amharic speech generation            |
| **Streaming** | WebSocket + PCM16                  | <50 ms network jitter     | Full-duplex audio transport          |

> Performance values represent observed development benchmarks and depend on hardware, network conditions, utterance length, and audio quality.

---

# 🛠 Architecture & Microservices

## 1. Voice Activity Detection Service

The VAD service continuously analyzes incoming audio and determines when the farmer starts and stops speaking.

### Technology

* **Silero VAD v5**
* Streaming PCM audio processing
* Circular audio buffering

### Responsibilities

* Detect speech boundaries
* Ignore silence and background audio
* Trigger ASR only when valid speech is detected
* Detect user interruption during system playback
* Support full-duplex conversational behavior

### Speech Padding

A short pre-speech buffer of approximately **200 ms** is preserved before the detected speech boundary.

This helps prevent the beginning of words from being clipped when speech detection activates slightly after the farmer begins speaking.

---

## 2. Automatic Speech Recognition Service

The ASR service converts spoken Amharic into text.

### Technology

* `faster-whisper`
* CTranslate2
* Fine-tuned Whisper-Small
* Amharic domain adaptation

### Audio Pipeline

Incoming audio is normalized before inference and converted into the format expected by the ASR model.

The system is optimized for difficult audio conditions including:

* 8 kHz telephone-bandwidth speech
* Environmental noise
* Different microphone qualities
* Variable speaking speed
* Regional pronunciation differences
* Low-volume speech

---

## ASR Accuracy

The current model achieves approximately:

```text
~35% Word Error Rate
```

on challenging **8 kHz telephone-quality Amharic speech**.

This result should be interpreted in the context of both the acoustic environment and the linguistic characteristics of Amharic.

### Factors Affecting WER

Amharic ASR presents several challenges:

* **Phonetically similar words and homophones**
* **Rich morphology**, where a root can generate many inflected word forms
* **Regional pronunciation and dialect variation**
* **Orthographic variation**
* **Differences between spoken and written forms**
* **Code-switching with English and other Ethiopian languages**
* **Limited large-scale labeled Amharic speech datasets**
* **Background noise and microphone variability**
* **Loss of acoustic information in 8 kHz telephone audio**

Because of these factors, raw WER alone does not fully represent the practical usefulness of the system.

For example, a transcript may contain several incorrect words while still correctly identifying:

* The crop
* The observed symptom
* The suspected pest
* The disease
* The location
* The farmer's intended question

The downstream reasoning pipeline can therefore still produce a useful response even when the transcription is not word-for-word perfect.

---

## 3. Context-Aware Transcript Normalization

Raw ASR output is optionally passed through a contextual post-processing layer before retrieval.

### Purpose

The post-processing layer attempts to:

* Correct likely transcription errors
* Normalize agricultural terminology
* Resolve phonetically similar words using context
* Restore punctuation
* Normalize spelling variation
* Preserve crop, pest, disease, and chemical names

---

## 4. Retrieval-Augmented Generation Service

The RAG service retrieves relevant agricultural knowledge before generating a response.

### Technology

* **FastAPI**
* **PostgreSQL**
* **pgvector**
* `paraphrase-multilingual-MiniLM-L12-v2`

### Retrieval Flow

```text
Normalized Farmer Query
        │
        ▼
Embedding Generation
        │
        ▼
pgvector Similarity Search
        │
        ▼
Relevant Agricultural Documents
        │
        ▼
LLM Context Construction
        │
        ▼
Grounded Agricultural Response
```

The retrieval layer helps constrain generated answers to curated agricultural knowledge rather than relying entirely on the language model's internal knowledge.

---

## Agricultural Knowledge Base

The knowledge base contains domain-specific agricultural content including:

* Soil acidity and soil management
* Crop rotation
* Fertilizer usage
* Pest identification
* Pest control
* Crop diseases
* Irrigation practices
* Plant symptoms
* Crop management
* Preventive farming practices

The system also supports multi-step diagnostic conversations.

For example:

```text
Farmer:
"My maize leaves are becoming yellow."

Advisor:
"Are the lower leaves turning yellow first, or the younger leaves?"

Farmer:
"The lower leaves."

Advisor:
"Is the yellowing uniform, or does it appear between the leaf veins?"
```

Instead of immediately generating a diagnosis, the system can gather additional evidence before recommending an action.

---

## 5. Confidence-Aware Escalation

The system supports escalation when automated advice is insufficient or uncertain.

Possible escalation signals include:

* Low retrieval similarity
* Missing agricultural context
* Conflicting symptoms
* Unsupported treatment requests
* Repeated misunderstanding
* Potentially high-risk recommendations

Low-confidence cases can be written to a PostgreSQL table such as:

```text
escalations
```

for later review by an agricultural expert.

A typical escalation record can include:

```text
conversation_id
farmer_query
asr_transcript
normalized_query
retrieved_documents
confidence_score
generated_response
timestamp
status
```

This creates a human-in-the-loop path for cases where automated advice should not be trusted without expert review.

---

## 6. Response Generation

The response-generation layer combines:

```text
Conversation History
        +
Retrieved Agricultural Knowledge
        +
Current Farmer Question
        +
System Safety Rules
```

The generated answer is designed to be:

* Concise
* Conversational
* Easy to understand
* Appropriate for spoken interaction
* Grounded in retrieved agricultural information

Long technical explanations are avoided because the primary user interface is voice.

---

## 7. Text-to-Speech Service

The TTS service converts the generated Amharic response back into speech.

### Engines

* gTTS
* edge-tts
* Configurable fallback providers

### Audio Processing

Generated audio is converted into a standardized streaming format:

```text
16 kHz
PCM16
Mono
```

where required by the playback pipeline.

### Sentence-Level Streaming

Responses are divided into sentence-sized units.

Instead of waiting for the entire response to be synthesized:

```text
Sentence 1 → TTS → Stream
Sentence 2 → TTS → Stream
Sentence 3 → TTS → Stream
```

This reduces perceived waiting time and allows the farmer to begin hearing the answer before the complete response has been synthesized.

---

# 🎙 Full-Duplex Voice Interaction

The system supports **barge-in**, allowing the farmer to interrupt the AI while it is speaking.

Example:

```text
AI:
"You should first check whether the leaves—"

Farmer:
"No, the problem is on the fruit."

AI playback stops.

VAD detects new speech.

ASR processes the interruption.

The conversation continues using the new information.
```

This interaction model makes the system behave more like a natural telephone conversation than a traditional push-to-talk assistant.

---

## Interruption Flow

```text
TTS Playback
     │
     ▼
VAD Detects Farmer Speech
     │
     ├── Stop Playback
     ├── Cancel Pending TTS
     ├── Cancel / Ignore Previous Generation
     │
     ▼
Process New Farmer Utterance
```

---

# 🌐 Voice Gateway & Web Client

## Gateway

The gateway is implemented using **FastAPI WebSockets**.

It transports two categories of data.

### Control Events

JSON messages are used for events such as:

```json
{
  "type": "speech_start"
}
```

```json
{
  "type": "speech_end"
}
```

```json
{
  "type": "interrupt"
}
```

```json
{
  "type": "transcript",
  "text": "..."
}
```

### Media

Raw binary audio frames are transmitted separately using PCM16.

This avoids the overhead of encoding continuous audio inside JSON.

---

## Frontend

The browser client is implemented using:

* Vanilla JavaScript
* Web Audio API
* WebSocket
* PCM audio streaming

The playback system maintains a small jitter-buffered queue so incoming audio chunks can be played continuously.

Browser audio initialization is triggered through a user interaction before the conversation begins to comply with modern browser autoplay restrictions.

---

# 🔄 Real-Time Conversation Flow

```text
Microphone
   │
   ▼
Browser Audio Capture
   │
   ▼
WebSocket
   │
   ▼
VAD
   │
   ▼
Utterance Buffer
   │
   ▼
ASR
   │
   ▼
Transcript Normalization
   │
   ▼
Embedding
   │
   ▼
pgvector Retrieval
   │
   ▼
LLM Response
   │
   ▼
Sentence Splitter
   │
   ▼
TTS
   │
   ▼
PCM16 Streaming
   │
   ▼
Browser Playback
```

---

# ⚡ Concurrency & Streaming

The application handles several operations concurrently:

```text
Audio reception
VAD processing
ASR inference
LLM generation
TTS synthesis
Audio playback
Interruption detection
```

A synchronized `safe_send` mechanism is used where necessary to prevent multiple asynchronous tasks from writing conflicting frames to the same WebSocket connection.

Cancellation logic prevents obsolete responses from continuing after a farmer interrupts the system.

---

# 🏗 Deployment

The platform is containerized using **Docker Compose**.

Example service structure:

```text
services/
│
├── gateway
├── vad-service
├── asr-service
├── rag-service
├── tts-service
├── postgres
└── redis
```

Each major subsystem can be deployed and scaled independently.

---

## Example Scaling Strategy

```text
              ┌── ASR Worker 1
Gateway ──────┼── ASR Worker 2
              └── ASR Worker 3

              ┌── RAG Worker 1
              └── RAG Worker 2

              ┌── TTS Worker 1
              └── TTS Worker 2
```

ASR workers can therefore be scaled independently without changing the retrieval or speech synthesis layers.

This is particularly important because ASR inference is typically more computationally expensive than vector retrieval.

---

# 💾 Storage

PostgreSQL stores:

* Agricultural knowledge
* Document metadata
* Vector embeddings
* Conversations
* Escalations
* Evaluation records

`pgvector` provides vector similarity search directly inside PostgreSQL.

Redis can optionally be used for:

* Short-lived conversation state
* Caching
* Queue coordination
* Session metadata
* Distributed worker communication

Persistent Docker volumes are used for database storage, model files, cached artifacts, or recordings where required.

Real-time audio transport itself remains primarily WebSocket/in-memory based rather than relying on shared filesystem communication.

---

# 📊 Evaluation Strategy

The system is evaluated at both the component and end-to-end levels.

## ASR

Primary metrics:

```text
Word Error Rate (WER)
Character Error Rate (CER)
```

Additional analysis includes performance by:

* Audio quality
* Speaker
* Dialect
* Noise level
* Utterance length
* Agricultural terminology

---

## RAG

Retrieval quality can be evaluated using metrics such as:

```text
Recall@K
Precision@K
MRR
Hit Rate
```

Evaluation queries are paired with expected agricultural documents to measure whether the correct evidence is retrieved.

---

## Semantic Understanding

Because voice assistants do not require perfect transcripts to succeed, semantic evaluation is also performed.

Metrics can include:

* Intent recognition accuracy
* Crop entity recognition
* Pest entity recognition
* Disease entity recognition
* Symptom extraction accuracy
* Treatment/action extraction
* Semantic similarity with reference intent

---

## End-to-End Evaluation

The most important metric is whether the system successfully helps the farmer.

Example end-to-end metrics include:

```text
Task Success Rate
Response Groundedness
Answer Correctness
Escalation Accuracy
Average Response Latency
Barge-in Success Rate
```

This is particularly important for Amharic speech recognition because a transcription can have a relatively high WER while still preserving the farmer's intended meaning.

---

# 🧪 Example Interaction

```text
Farmer:
የቲማቲሜ ቅጠል ቢጫ እየሆነ ነው።

ASR:
Recognizes the Amharic speech.

Normalization:
Identifies "tomato" and "yellowing leaves".

RAG:
Retrieves relevant information about nutrient deficiency,
watering problems, and common tomato diseases.

Advisor:
Asks a follow-up question to distinguish possible causes.

Farmer:
Provides additional symptoms.

Advisor:
Returns a more targeted recommendation.
```

The system prioritizes **diagnostic questioning** over immediately producing a potentially unreliable recommendation.

---

# 🛡 Reliability & Safety

Agricultural advice can affect crop yield, farmer income, and potentially human or environmental safety.

For that reason, the system is designed around several safeguards:

* Retrieval-grounded responses
* Confidence-aware escalation
* Preservation of original ASR transcripts
* Human expert review path
* Conversation logging
* Diagnostic follow-up questions
* Avoidance of unsupported recommendations
* Cancellation of outdated responses after interruption

Future versions can add stricter validation for recommendations involving pesticides, fertilizers, veterinary products, or regulated agricultural chemicals.

---

# 🔧 Technology Stack

| Layer                      | Technology                      |
| :------------------------- | :------------------------------ |
| **Backend**                | FastAPI, Python                 |
| **Streaming**              | WebSocket, PCM16                |
| **VAD**                    | Silero VAD                      |
| **ASR**                    | faster-whisper, CTranslate2     |
| **ASR Model**              | Fine-tuned Whisper-Small        |
| **Embeddings**             | multilingual MiniLM-L12         |
| **Vector Search**          | pgvector                        |
| **Database**               | PostgreSQL                      |
| **Caching / Coordination** | Redis                           |
| **LLM**                    | Gemini / Groq-compatible models |
| **TTS**                    | gTTS / edge-tts                 |
| **Frontend**               | JavaScript, Web Audio API       |
| **Deployment**             | Docker, Docker Compose          |

---

# 🎯 Design Goals

The architecture is designed around five primary goals:

1. **Low-latency voice interaction**
2. **Robust Amharic speech processing**
3. **Grounded agricultural responses**
4. **Natural interruption and conversational flow**
5. **Independent scaling and replacement of AI components**

The modular architecture makes it possible to upgrade individual components without redesigning the entire system.

For example:

```text
Whisper-Small
      ↓
Whisper-Medium / Large
```

or

```text
MiniLM embeddings
      ↓
A stronger multilingual embedding model
```

can be introduced without changing the overall voice interaction architecture.

---

# 🚧 Current Limitations

The current implementation still has several limitations:

* Approximately **35% WER** under difficult telephone-bandwidth Amharic conditions
* Limited availability of large agricultural Amharic speech datasets
* Dependence on hosted LLM/TTS providers for some configurations
* Variation in ASR accuracy between speakers and environments
* Retrieval quality depends heavily on knowledge-base coverage
* Agricultural terminology may differ significantly between regions
* End-to-end latency can increase when external APIs are used

These limitations are treated as measurable engineering problems rather than hidden behind the user interface.

---

# 🔮 Future Improvements

Planned improvements include:

* Larger and more diverse Amharic ASR training datasets
* Agricultural vocabulary-aware decoding
* Improved text normalization
* Stronger multilingual embedding models
* RAG reranking
* Hybrid lexical + vector retrieval
* Streaming ASR
* Streaming TTS
* Local Amharic TTS models
* Model quantization
* GPU inference optimization
* SIP / PSTN telephony integration
* Offline or edge deployments
* Additional Ethiopian languages
* Expert feedback loops
* Automated evaluation dashboards
* Agricultural entity-specific evaluation datasets

---

# 📌 Project Goal

The goal of this project is not simply to build a chatbot with speech input.

It is to explore how a reliable conversational AI system can combine:

```text
Speech Recognition
        +
Language Understanding
        +
Knowledge Retrieval
        +
Conversational Reasoning
        +
Speech Generation
```

to make agricultural information more accessible to farmers who may prefer speaking in Amharic rather than interacting with text-based digital systems.
