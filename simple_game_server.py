#!/usr/bin/env python3
"""
Simple Wolof Pronunciation Game Server - Test Version
Serves the game without heavy ML models for quick deployment
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Sample game phrases
GAME_PHRASES = [
    {
        "id": "greeting_1",
        "sentence": "Nanga def?",
        "translation": "How are you?",
        "difficulty": "beginner"
    },
    {
        "id": "greeting_2",
        "sentence": "Maa ngi fi.",
        "translation": "I am here.",
        "difficulty": "beginner"
    },
    {
        "id": "question_1",
        "sentence": "Xaatim ngaa?",
        "translation": "How are you?",
        "difficulty": "beginner"
    },
    {
        "id": "statement_1",
        "sentence": "Noppali na.",
        "translation": "It is good.",
        "difficulty": "intermediate"
    },
    {
        "id": "question_2",
        "sentence": "Fan la?",
        "translation": "When?",
        "difficulty": "intermediate"
    }
]

@app.route('/')
def index():
    """Serve the game frontend."""
    return send_from_directory('static', 'index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": False,
        "phrases_loaded": len(GAME_PHRASES) > 0,
        "version": "simple-test"
    })

@app.route('/api/phrases')
def get_phrases():
    """Get available game phrases."""
    return jsonify({
        "phrases": GAME_PHRASES,
        "total": len(GAME_PHRASES)
    })

@app.route('/api/score', methods=['POST'])
def score_pronunciation():
    """Mock scoring for testing."""
    # Return a mock score for now
    import random
    score = random.randint(60, 95)

    if score >= 90:
        feedback = "Excellent! Native-like pronunciation! 🎉"
    elif score >= 75:
        feedback = "Great job! Very good pronunciation! 👏"
    elif score >= 60:
        feedback = "Good effort! Keep practicing! 💪"
    else:
        feedback = "Getting there! Listen carefully and try again! 🎧"

    return jsonify({
        "score": score,
        "similarity": score / 100.0,
        "feedback": feedback,
        "user_transcription": "mock transcription",
        "reference_sentence": "Nanga def?",
        "translation": "How are you?",
        "model": "simple-test"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 49152))
    print(f"🚀 Simple Wolof Game Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)