#!/usr/bin/env python3
"""
Wolof Pronunciation Game Server - Holistic Scoring
Combines:
  - Wav2Vec2 encoder embeddings for acoustic similarity
  - NVIDIA Parakeet gRPC transcription for pronunciation accuracy
  - Nemotron AI for intelligent feedback with transcription diff analysis
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import json
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModel
from pathlib import Path
import tempfile
import requests
from difflib import SequenceMatcher
import riva.client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# NVIDIA Parakeet gRPC config
PARAKEET_SERVER = "grpc.nvcf.nvidia.com:443"
PARAKEET_FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"
PARAKEET_API_KEY = os.environ.get('PARAKEET_API_KEY', '')

# Nemotron config from Maxwell
NEMOTRON_API_BASE = "http://100.116.54.128:7777"
NEMOTRON_MODEL = "C:\\dev\\models\\Nemotron-Nano-9B-v2\\nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

# Load game phrases
with open('game_phrases.json', 'r', encoding='utf-8') as f:
    GAME_PHRASES = json.load(f)

# Initialize Wav2Vec2 for embeddings (Parakeet encoder not available in transformers)
print("🚀 Loading Wav2Vec2 encoder for embeddings...")
model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"✅ Wav2Vec2 encoder loaded on {device}")

# Initialize Riva ASR client
print("🚀 Connecting to NVIDIA Parakeet gRPC API...")
try:
    auth = riva.client.Auth(
        uri=PARAKEET_SERVER,
        use_ssl=True,
        metadata_args=[
            ["function-id", PARAKEET_FUNCTION_ID],
            ["authorization", f"Bearer {PARAKEET_API_KEY}"]
        ]
    )
    asr_service = riva.client.ASRService(auth)
    print("✅ Parakeet gRPC connected")
except Exception as e:
    print(f"⚠️ Parakeet gRPC connection failed: {e}")
    asr_service = None

# Pre-compute reference data
print("📊 Pre-computing reference embeddings + transcriptions...")
reference_data = {}


def extract_wav2vec2_embedding(audio_path):
    """Extract latent representation from audio using Wav2Vec2 encoder."""
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, duration=10)

        # Process audio for Parakeet
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Extract hidden states from encoder
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"❌ Error extracting Parakeet embedding: {e}")
        return None


def transcribe_with_parakeet_grpc(audio_path):
    """Transcribe audio using NVIDIA Parakeet gRPC API."""
    if asr_service is None:
        return None

    try:
        # Load audio in format Riva expects
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()

        # Configure transcription
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code="en-US",  # Parakeet is multilingual, will detect
            max_alternatives=1,
            enable_automatic_punctuation=False
        )

        # Transcribe
        response = asr_service.offline_recognize(audio_bytes, config)

        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            return transcript.strip()
        else:
            return None
    except Exception as e:
        print(f"⚠️ Parakeet gRPC transcription failed: {e}")
        return None


# Pre-compute all reference data
for phrase in GAME_PHRASES:
    audio_path = phrase['audio_path']
    if Path(audio_path).exists():
        # Extract embedding
        emb = extract_wav2vec2_embedding(audio_path)

        # Get transcription
        transcript = transcribe_with_parakeet_grpc(audio_path)

        if emb is not None:
            reference_data[phrase['id']] = {
                'embedding': emb,
                'transcript': transcript,
                'sentence': phrase['sentence']
            }

            transcript_preview = (transcript[:40] + "...") if transcript else "⚠️ gRPC failed"
            print(f"  ✓ {phrase['sentence'][:35]}... → {transcript_preview}")
    else:
        print(f"  ✗ Audio not found: {audio_path}")

print(f"✅ Computed {len(reference_data)} embeddings + transcriptions")


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def text_similarity(ref_text, user_text):
    """Calculate similarity between two transcriptions."""
    if not ref_text or not user_text:
        return 0.0

    ref_text = ' '.join(ref_text.lower().split())
    user_text = ' '.join(user_text.lower().split())

    return SequenceMatcher(None, ref_text, user_text).ratio()


def call_nemotron_for_feedback(acoustic_sim, transcription_sim, ref_text, user_transcript, ref_transcript):
    """Get intelligent holistic feedback from Nemotron."""
    try:
        context = f"""Student attempted to say: "{ref_text}"

Reference transcription: "{ref_transcript or 'N/A'}"
Student transcription: "{user_transcript or 'N/A'}"

Acoustic similarity: {acoustic_sim:.3f} (cosine similarity, Parakeet encoder embeddings)
Transcription similarity: {transcription_sim:.3f} (sequence match ratio)

Provide 2-3 sentences of specific, encouraging feedback:
1. Point out the specific differences between the reference and student transcription
2. Mention what the similarity scores mean
3. Give actionable advice on which sounds or words to improve"""

        response = requests.post(
            f"{NEMOTRON_API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": NEMOTRON_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert Wolof pronunciation coach. Give specific, actionable, encouraging feedback based on transcription and acoustic analysis."},
                    {"role": "user", "content": context}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Nemotron feedback failed: {e}")

    return None


def score_pronunciation(user_embedding, user_transcript, ref_embedding, ref_transcript, phrase_text):
    """
    Holistic pronunciation scoring combining:
    - Acoustic similarity (Parakeet encoder embeddings) - raw cosine similarity
    - Transcription accuracy (Parakeet gRPC ASR diff) - raw sequence match ratio
    """
    # 1. Acoustic similarity (raw cosine similarity 0-1)
    acoustic_sim = cosine_similarity(user_embedding, ref_embedding)

    # 2. Transcription accuracy (raw sequence match ratio 0-1)
    if ref_transcript and user_transcript:
        transcription_sim = text_similarity(ref_transcript, user_transcript)
    else:
        transcription_sim = 0.0

    # 3. Combined holistic score (weighted average of raw similarities)
    holistic_score = (0.6 * acoustic_sim) + (0.4 * transcription_sim)

    # Get AI feedback with raw scores
    ai_feedback = call_nemotron_for_feedback(
        acoustic_sim,
        transcription_sim,
        phrase_text,
        user_transcript or "",
        ref_transcript or ""
    )

    if ai_feedback:
        feedback = ai_feedback
        ai_powered = True
    else:
        # Fallback feedback based on raw scores
        if holistic_score >= 0.85:
            feedback = "🎉 Excellent! Your pronunciation sounds native!"
        elif holistic_score >= 0.70:
            feedback = "👏 Great job! Very close to perfect!"
        elif holistic_score >= 0.55:
            feedback = "💪 Good effort! Keep practicing the sounds!"
        elif holistic_score >= 0.35:
            feedback = "🎧 Getting there! Listen and try matching the rhythm!"
        else:
            feedback = "📢 Let's try again! Focus on the pronunciation!"
        ai_powered = False

    return {
        "score": float(round(holistic_score, 3)),
        "acoustic_similarity": float(round(acoustic_sim, 3)),
        "transcription_similarity": float(round(transcription_sim, 3)),
        "reference_transcript": ref_transcript,
        "user_transcript": user_transcript,
        "feedback": feedback,
        "ai_feedback": ai_powered
    }


@app.route('/')
def index():
    return send_file('static/index.html')


@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "wav2vec2+parakeet-grpc",
        "phrases_loaded": len(GAME_PHRASES),
        "data_ready": len(reference_data),
        "device": device,
        "grpc_connected": asr_service is not None
    })


@app.route('/api/phrases')
def get_phrases():
    phrases_for_client = [
        {
            "id": p["id"],
            "sentence": p["sentence"],
            "word_count": p["word_count"],
            "difficulty": "beginner" if p["word_count"] <= 4 else "intermediate"
        }
        for p in GAME_PHRASES
        if p["id"] in reference_data
    ]

    return jsonify({
        "phrases": phrases_for_client,
        "total": len(phrases_for_client)
    })


@app.route('/api/audio/<phrase_id>')
def get_audio(phrase_id):
    phrase = next((p for p in GAME_PHRASES if p['id'] == phrase_id), None)

    if not phrase or not Path(phrase['audio_path']).exists():
        return jsonify({"error": "Audio not found"}), 404

    return send_file(phrase['audio_path'], mimetype='audio/mpeg')


@app.route('/api/score', methods=['POST'])
def score_user_pronunciation():
    """Score user pronunciation using Parakeet encoder + gRPC transcription."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400

        phrase_id = request.form.get('phrase_id')
        if not phrase_id or phrase_id not in reference_data:
            return jsonify({"error": "Invalid phrase_id"}), 400

        # Save uploaded audio
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Extract user embedding
            user_embedding = extract_wav2vec2_embedding(tmp_path)
            if user_embedding is None:
                return jsonify({"error": "Failed to process audio"}), 400

            # Transcribe user audio
            user_transcript = transcribe_with_parakeet_grpc(tmp_path)

            # Get reference data
            ref_data = reference_data[phrase_id]
            ref_embedding = ref_data['embedding']
            ref_transcript = ref_data['transcript']
            phrase_text = ref_data['sentence']

            # Score pronunciation holistically
            result = score_pronunciation(
                user_embedding,
                user_transcript,
                ref_embedding,
                ref_transcript,
                phrase_text
            )

            return jsonify({
                **result,
                "reference_sentence": phrase_text,
                "model": "wav2vec2-large + parakeet-1.1b-grpc",
                "embedding_dim": len(user_embedding)
            })

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"❌ Error scoring: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_nemotron():
    """Chat with Nemotron for Wolof learning support."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = requests.post(
            f"{NEMOTRON_API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": NEMOTRON_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful Wolof language learning assistant. Answer questions about Wolof pronunciation, grammar, and culture. Be concise and encouraging."},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()
            return jsonify({
                "response": ai_response,
                "model": "nemotron-nano-9b"
            })
        else:
            return jsonify({"error": "AI service unavailable"}), 503

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": "Chat service temporarily unavailable"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 49152))
    print(f"\n🚀 Wolof Pronunciation Game Server - Holistic Scoring")
    print(f"📡 Running on http://localhost:{port}")
    print(f"🎯 {len(GAME_PHRASES)} phrases loaded")
    print(f"🧠 {len(reference_data)} Wav2Vec2 embeddings + Parakeet transcriptions ready")
    print(f"🎵 Real Wolof audio from Zenodo dataset")
    print(f"🤖 Nemotron AI coach: {NEMOTRON_API_BASE}")
    print(f"🎙️  Acoustic embeddings: facebook/wav2vec2-large-960h-lv60-self")
    print(f"🎙️  Transcription ASR: NVIDIA Parakeet-1.1b (gRPC at {PARAKEET_SERVER})")
    print(f"⚡ Holistic scoring: 60% acoustic similarity + 40% transcription accuracy")
    app.run(host='0.0.0.0', port=port, debug=False)
