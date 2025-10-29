#!/usr/bin/env python3
"""
Wolof Pronunciation Game Server with Audio Feature Embeddings
Uses MFCC features for pronunciation scoring (lightweight, no model loading)
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import json
import librosa
import numpy as np
from pathlib import Path
import tempfile

app = Flask(__name__)
CORS(app)

# Load game phrases
with open('game_phrases.json', 'r', encoding='utf-8') as f:
    GAME_PHRASES = json.load(f)

print(f"✅ Loaded {len(GAME_PHRASES)} game phrases")

# Pre-compute reference embeddings using MFCC
print("📊 Pre-computing MFCC embeddings from real Wolof audio...")
reference_embeddings = {}

def extract_mfcc_embedding(audio_path, n_mfcc=40):
    """Extract MFCC features as audio embedding."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, duration=5)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Use mean across time as embedding
        embedding = np.mean(mfccs, axis=1)

        return embedding
    except Exception as e:
        print(f"❌ Error extracting MFCC from {audio_path}: {e}")
        return None

# Pre-compute all reference embeddings from real Wolof audio
for phrase in GAME_PHRASES:
    audio_path = phrase['audio_path']
    if Path(audio_path).exists():
        emb = extract_mfcc_embedding(audio_path)
        if emb is not None:
            reference_embeddings[phrase['id']] = emb
            print(f"  ✓ {phrase['sentence'][:50]}...")
    else:
        print(f"  ✗ Audio not found: {audio_path}")

print(f"✅ Computed {len(reference_embeddings)} reference embeddings")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def score_pronunciation(user_embedding, reference_embedding):
    """Score pronunciation using cosine similarity of MFCC embeddings."""
    similarity = cosine_similarity(user_embedding, reference_embedding)

    # Convert similarity to 0-100 score
    # MFCC similarity typically ranges from 0.3-0.9
    score = max(0, min(100, (similarity - 0.2) * 125))  # Scale 0.2-1.0 -> 0-100

    if score >= 85:
        feedback = "🎉 Excellent! Native-like pronunciation!"
    elif score >= 70:
        feedback = "👏 Great job! Very close to the reference!"
    elif score >= 55:
        feedback = "💪 Good effort! Keep practicing!"
    elif score >= 35:
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
        "model": "mfcc-embeddings",
        "phrases_loaded": len(GAME_PHRASES),
        "embeddings_ready": len(reference_embeddings),
        "embedding_dim": 40
    })

@app.route('/api/phrases')
def get_phrases():
    """Get available game phrases."""
    phrases_for_client = [
        {
            "id": p["id"],
            "sentence": p["sentence"],
            "word_count": p["word_count"],
            "difficulty": "beginner" if p["word_count"] <= 4 else "intermediate"
        }
        for p in GAME_PHRASES
        if p["id"] in reference_embeddings
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
    """Score user pronunciation using MFCC embeddings."""
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
            # Extract user MFCC embedding
            user_embedding = extract_mfcc_embedding(tmp_path)

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
                "model": "mfcc-features",
                "embedding_dim": 40
            })

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"❌ Error scoring: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 49152))
    print(f"\n🚀 Wolof Pronunciation Game Server")
    print(f"📡 Running on http://localhost:{port}")
    print(f"🎯 {len(GAME_PHRASES)} phrases loaded")
    print(f"🧠 {len(reference_embeddings)} MFCC embeddings ready")
    print(f"🎵 Real Wolof audio from Zenodo dataset")
    app.run(host='0.0.0.0', port=port, debug=False)