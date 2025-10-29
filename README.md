# Wolof Pronunciation Game

A web-based pronunciation learning game for Wolof language with AI-powered scoring.

## Live Demo

🌍 **https://wolof.mithran.org** - Accessible through Cloudflare tunnel

## Features

- **Interactive Gameplay**: Practice Wolof pronunciation with immediate feedback
- **AI Scoring**: Powered by speech recognition models
- **Modern UI**: React-based frontend with smooth animations
- **Cloudflare Deployment**: Global CDN with Workers proxy
- **Real-time Processing**: Audio transcription and similarity scoring

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React UI      │    │  Cloudflare      │    │  Local Server   │
│   (Browser)     │◄──►│   Worker         │◄──►│   (Port 49152)  │
│                 │    │  (Proxy/CORS)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Local Development

```bash
# Install dependencies
conda create -n wolof-game python=3.10 -y
conda activate wolof-game
pip install flask flask-cors torch transformers librosa requests numpy nvidia-riva-client python-dotenv

# Create .env file with your NVIDIA API key
echo "PARAKEET_API_KEY=your-api-key-here" > .env

# Start the game server
PORT=49152 python parakeet_holistic_server.py

# Access the game at http://localhost:49152
```

### Production Deployment

The game is deployed using Cloudflare Workers:

- **Frontend**: React app served via Cloudflare Workers
- **Backend**: Local server exposed through Cloudflare tunnel
- **API Proxy**: Workers handle CORS and proxy requests to local server

## Game Phrases

The game includes sample Wolof phrases:

- "Nanga def?" (How are you?) - Beginner
- "Maa ngi fi." (I am here.) - Beginner
- "Xaatim ngaa?" (How are you?) - Beginner
- "Noppali na." (It is good.) - Intermediate
- "Fan la?" (When?) - Intermediate

## Audio Processing

The game uses **holistic pronunciation scoring** combining multiple analysis methods:

### Dual-Model Scoring System
1. **Acoustic Similarity (60% weight)**:
   - Wav2Vec2 encoder extracts latent embeddings from speech audio
   - Cosine similarity compares user vs. reference pronunciation
   - Captures prosody, rhythm, and acoustic features

2. **Transcription Accuracy (40% weight)**:
   - NVIDIA Parakeet gRPC API transcribes user and reference audio
   - Sequence matching measures pronunciation correctness
   - Identifies specific word/sound errors

3. **AI-Powered Feedback**:
   - Nemotron-Nano-9B analyzes both scores + transcription diffs
   - Provides specific, actionable pronunciation coaching
   - Points out exact differences and improvement areas

### How It Works
- **Reference Audio**: 10 native Wolof phrases from Zenodo CommonVoice dataset
- **Embedding Model**: facebook/wav2vec2-large-960h-lv60-self (1024-dim vectors)
- **Transcription ASR**: NVIDIA Parakeet-1.1B (multilingual gRPC API)
- **Holistic Score**: Raw similarity (0-1 scale): `0.6 × acoustic + 0.4 × transcription`
- **AI Coach**: Nemotron analyzes transcription diffs and gives targeted feedback
- **Chat Feature**: Ask the AI coach questions about Wolof pronunciation anytime

## Technical Stack

- **Frontend**: React 18 + Web Audio API
- **Backend**: Flask + Python 3.10
- **Speech Processing**:
  - Wav2Vec2 (facebook/wav2vec2-large-960h-lv60-self) for acoustic embeddings
  - NVIDIA Parakeet gRPC API (parakeet-1.1b-multilingual) for transcription
  - Nemotron-Nano-9B for AI coaching with transcription diff analysis
- **Deployment**: Cloudflare Workers + Tunnel
- **Audio**: Real Wolof clips (16kHz MP3) from native speakers
- **ML Inference**: CUDA-accelerated on local GPU (RTX 5060 Ti)
- **Environment**: python-dotenv for secure API key management

## Dataset

This game uses **10 native Wolof audio phrases** from the Zenodo CommonVoice Wolof TTS dataset:

**Citation:**
```
Wolof Text-to-Speech (TTS) Dataset
Zenodo CommonVoice Corpus
Source: https://zenodo.org/record/4516532
License: CC-BY 4.0
Speakers: 2 native Wolof actors (1 male, 1 female)
Total clips: ~40,000 short phrases
```

### Dataset Details
- **Audio Format**: MP3 (16kHz, mono)
- **Content**: Short conversational Wolof phrases (3-8 words)
- **Quality**: Clean, native pronunciation
- **Location**: `/datasets/zenodo-wolof-tts/data-commonvoice/`

### Additional Resources
- **OpenSLR Wolof Speech Corpus**: Used for validation (not included in game)
- **Analysis Report**: See `datasets/WOLOF_PRONUNCIATION_GAME_DATA_REPORT.md`

### License & Attribution
Audio files are used under **CC-BY 4.0 license**. We gratefully acknowledge the contributors to the Zenodo CommonVoice Wolof corpus for making this educational game possible.

## Contributing

1. Fork this repository
2. Create a feature branch
3. Add new Wolof phrases and audio samples
4. Submit a pull request

## License

MIT License - see LICENSE file for details