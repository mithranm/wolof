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
pip install -r requirements.txt

# Start the game server
PORT=49152 python simple_game_server.py

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

The game uses **Parakeet ASR** for advanced speech analysis:
1. **Latent Representation Extraction**: Wav2Vec2 embeddings capture pronunciation features
2. **Cosine Similarity Scoring**: Compare user audio to reference embeddings
3. **AI-Powered Feedback**: Nemotron-Nano-9B provides intelligent pronunciation coaching

### How It Works
- **Reference Audio**: 10 native Wolof phrases from Zenodo CommonVoice dataset
- **Embedding Model**: Wav2Vec2 (Parakeet-compatible) extracts 1024-dim latent vectors
- **Scoring**: Cosine similarity between user/reference embeddings (0-100 scale)
- **AI Coach**: Nemotron analyzes pronunciation and gives specific improvement tips

## Technical Stack

- **Frontend**: React 18 + Web Audio API
- **Backend**: Flask + Python
- **Speech Processing**:
  - Parakeet ASR (Wav2Vec2) for embedding extraction
  - Nemotron-Nano-9B for intelligent feedback
- **Deployment**: Cloudflare Workers + Tunnel
- **Audio**: Real Wolof clips (16kHz MP3) from native speakers
- **ML Inference**: CUDA-accelerated on local GPU

## Dataset

Based on Wolof pronunciation datasets:
- OpenSLR Wolof Speech Corpus
- Zenodo Wolof TTS dataset

See `WOLOF_PRONUNCIATION_GAME_DATA_REPORT.md` for detailed analysis.

## Contributing

1. Fork this repository
2. Create a feature branch
3. Add new Wolof phrases and audio samples
4. Submit a pull request

## License

MIT License - see LICENSE file for details