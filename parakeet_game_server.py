#!/usr/bin/env python3
"""
Wolof Pronunciation Game Server with Parakeet ASR Embeddings
Uses real Wolof audio and latent representation scoring
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import json
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import tempfile

app = Flask(__name__)
CORS(app)

# Load game phrases
with open('game_phrases.json', 'r', encoding='utf-8') as f:
    GAME_PHRASES = json.load(f)

# Initialize Parakeet/Wav2Vec2 model for embeddings
print("🚀 Loading Parakeet-compatible model...")
model_name = "facebook/wav2vec2-large-960h-lv60-self"  # Use Wav2Vec2 as Parakeet proxy
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

# Pre-compute reference embeddings
print("📊 Pre-computing reference embeddings...")
reference_embeddings = {}

def extract_embedding(audio_path):
    """Extract latent representation from audio using Wav2Vec2."""
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]

        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"❌ Error extracting embedding from {audio_path}: {e}")
        return None

# Pre-compute all reference embeddings
for phrase in GAME_PHRASES:
    if Path(phrase['audio_path']).exists():
        emb = extract_embedding(phrase['audio_path'])
        if emb is not None:
            reference_embeddings[phrase['id']] = emb
            print(f"  ✓ {phrase['sentence'][:40]}...")

print(f"✅ Computed {len(reference_embeddings)} reference embeddings")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def score_pronunciation(user_embedding, reference_embedding):
    """Score pronunciation using cosine similarity of embeddings."""
    similarity = cosine_similarity(user_embedding, reference_embedding)

    # Convert similarity to 0-100 score
    # Similarity typically ranges from 0.5-1.0 for similar speech
    score = max(0, min(100, (similarity - 0.4) * 167))  # Scale 0.4-1.0 -> 0-100

    if score >= 90:
        feedback = "🎉 Excellent! Native-like pronunciation!"
    elif score >= 75:
        feedback = "👏 Great job! Very close to the reference!"
    elif score >= 60:
        feedback = "💪 Good effort! Keep practicing!"
    elif score >= 40:
        feedback = "🎧 Getting there! Listen carefully and try again!"
    else:
        feedback = "📢 Try again! Pay attention to the sounds!"

    return {
        "score": round(score, 1),
        "similarity": round(float(similarity), 3),
        "feedback": feedback
    }

@app.route('/')
def index():
    """Serve the game frontend."""
    return send_file('static/index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "wav2vec2-parakeet-proxy",
        "phrases_loaded": len(GAME_PHRASES),
        "embeddings_ready": len(reference_embeddings),
        "device": device
    })

@app.route('/api/phrases')
def get_phrases():
    """Get available game phrases."""
    # Remove audio_path from client response
    phrases_for_client = [
        {
            "id": p["id"],
            "sentence": p["sentence"],
            "word_count": p["word_count"],
            "difficulty": "beginner" if p["word_count"] <= 4 else "intermediate"
        }
        for p in GAME_PHRASES
        if p["id"] in reference_embeddings  # Only phrases with embeddings
    ]

    return jsonify({
        "phrases": phrases_for_client,
        "total": len(phrases_for_client)
    })

@app.route('/api/audio/<phrase_id>')
def get_audio(phrase_id):
    """Serve reference audio for a phrase."""
    phrase = next((p for p in GAME_PHRASES if p['id'] == phrase_id), None)

    if not phrase or not Path(phrase['audio_path']).exists():
        return jsonify({"error": "Audio not found"}), 404

    return send_file(phrase['audio_path'], mimetype='audio/mpeg')

@app.route('/api/score', methods=['POST'])
def score_user_pronunciation():
    """Score user pronunciation using Parakeet embeddings."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400

        phrase_id = request.form.get('phrase_id')
        if not phrase_id or phrase_id not in reference_embeddings:
            return jsonify({"error": "Invalid phrase_id"}), 400

        # Save uploaded audio temporarily
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Extract user embedding
            user_embedding = extract_embedding(tmp_path)

            if user_embedding is None:
                return jsonify({"error": "Failed to process audio"}), 400

            # Get reference embedding
            ref_embedding = reference_embeddings[phrase_id]

            # Score pronunciation
            result = score_pronunciation(user_embedding, ref_embedding)

            # Get phrase info
            phrase = next(p for p in GAME_PHRASES if p['id'] == phrase_id)

            return jsonify({
                **result,
                "reference_sentence": phrase['sentence'],
                "model": "wav2vec2-embeddings",
                "embedding_dim": len(user_embedding)
            })

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"❌ Error scoring: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 49152))
    print(f"\n🚀 Wolof Pronunciation Game Server")
    print(f"📡 Running on http://localhost:{port}")
    print(f"🎯 {len(GAME_PHRASES)} phrases loaded")
    print(f"🧠 {len(reference_embeddings)} embeddings ready")
    print(f"⚡ Device: {device.upper()}")
    app.run(host='0.0.0.0', port=port, debug=False)