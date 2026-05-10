# Voice-To-Voice Farmer Advisor (Ethiopia)

An AI-powered voice-to-voice system designed to provide Ethiopian farmers with expert agricultural advice in Amharic. The system uses a microservices architecture to handle telephony, speech recognition, natural language understanding (RAG), and speech synthesis.

## 🌟 Key Features

- **Amharic Speech-to-Text**: Uses OpenAI's Whisper model for accurate transcription.
- **RAG-Based Logic Service**: Retrieves information from a knowledge base using vector similarity search.
- **Amharic Text-to-Speech**: Generates natural-sounding Amharic speech.
- **Admin Dashboard**: A web interface to view escalation queues and manage the system.
- **Dockerized Deployment**: All services are containerized for easy deployment and management.

## 🏗️ Architecture

1.  **Telephony Service**: Handles incoming calls, audio capture, and playback.
2.  **ASR Service**: Transcribes Amharic speech to text using `faster-whisper`.
3.  **Ollama Service**: Provides semantic correction and grammar cleanup for transcripts.
4.  **Logic Service**: Processes text queries, performs RAG, and generates responses.
5.  **TTS Service**: Converts text to speech.
6.  **Admin Dashboard**: A web interface for system monitoring and management.

## 📂 Project Structure

```
Voice-To-Voice-Farmer-Advisor/
├── telephony_service/      # SIP/VoIP client and audio processing
├── asr_service/            # Speech-to-text service (Whisper + Ollama)
├── logic_service/          # RAG pipeline and business logic
├── tts_service/            # Text-to-speech service
├── admin_dashboard/        # Streamlit-based admin interface
├── data/                   # Data files (knowledge base, embeddings)
├── models/                 # Local model storage (ASR, etc.)
├── .env                    # Environment variables
└── docker-compose.yml      # Docker orchestration
```

## ⚙️ Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- NVIDIA GPU (Optional, but recommended for ASR performance)

## 🚀 Quick Start

1.  **Clone the repository**.

2.  **Configure Environment Variables**:
    Copy the example environment file and fill in your credentials:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` with your specific configuration.

3.  **Build and Start Services**:
    ```bash
    docker-compose up --build -d
    ```

4.  **Pull the Semantic Correction Model**:
    ```bash
    docker exec -it ollama_service ollama pull qwen2.5:7b
    ```

5.  **Access the Services**:
    - **Admin Dashboard**: [http://localhost:8501](http://localhost:8501)
    - **ASR Service**: [http://localhost:8001](http://localhost:8001)
    - **Logic Service**: [http://localhost:8000](http://localhost:8000)
    - **TTS Service**: [http://localhost:8002](http://localhost:8002)
    - **Telephony Service**: [http://localhost:5060](http://localhost:5060)

## 🧪 Testing

### Test ASR Service

You can test the transcription by uploading an Amharic `.wav` file:

```bash
curl -X POST http://localhost:8001/transcribe \
  -F "file=@test_audio.wav"
```

### Test Logic Service

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "የግብርና መረጃ እፈልጋለሁ", "phone_number": "+251912345678", "session_id": "test_session"}'
```

### Test TTS Service

```bash
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "ሰላም! እንዴት ልርዳዎት?"}' \
  -o response.mp3
```

## 📂 Service Details

### Telephony Service

Handles SIP communication and audio streaming. Currently configured for local testing but can be connected to a SIP server.

### STT Service

Uses OpenAI's Whisper model for Amharic speech recognition. Runs on port 8001.

### Logic Service
Uses a RAG (Retrieval-Augmented Generation) pipeline to fetch relevant agricultural advice from the local knowledge base based on the transcribed text.

### TTS Service

Uses gTTS for Amharic text-to-speech. Runs on port 8002.

### Admin Dashboard

Provides a web interface for:
- Viewing escalation queues
- Monitoring system health
- Accessing logs

## 🤝 Contributing

1.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
2.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
3.  Push to the branch (`git push origin feature/AmazingFeature`).
4.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.