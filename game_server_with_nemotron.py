#!/usr/bin/env python3
"""
Wolof Pronunciation Game Server with Nemotron AI
Intelligent pronunciation scoring using Nemotron LLM from Maxwell
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import requests
import random

app = Flask(__name__)
CORS(app)

# Load Maxwell LM registry
MAXWELL_REGISTRY_PATH = "/home/mithranmohanraj/.maxwell/lm_registry.json"

def load_maxwell_config():
    """Load Maxwell LM registry configuration."""
    try:
        with open(MAXWELL_REGISTRY_PATH, 'r') as f:
            config = json.load(f)

        # Find Nemotron model
        for llm in config.get('llms', []):
            if llm['name'] == 'nemotron-nano-9b':
                return llm

        return None
    except Exception as e:
        print(f"Failed to load Maxwell config: {e}")
        return None

# Load Nemotron config
nemotron_config = load_maxwell_config()

# Sample game phrases with more context
GAME_PHRASES = [
    {
        "id": "greeting_1",
        "sentence": "Nanga def?",
        "translation": "How are you?",
        "difficulty": "beginner",
        "phonetic_hint": "nan-ga def",
        "pronunciation_tips": "Clear 'n' sounds, 'def' like 'deh-f'"
    },
    {
        "id": "greeting_2",
        "sentence": "Maa ngi fi.",
        "translation": "I am here.",
        "difficulty": "beginner",
        "phonetic_hint": "maa ngee fee",
        "pronunciation_tips": "Soft 'ngi' sound, clear 'fi'"
    },
    {
        "id": "question_1",
        "sentence": "Xaatim ngaa?",
        "translation": "How are you?",
        "difficulty": "beginner",
        "phonetic_hint": "ha-teem ngaa",
        "pronunciation_tips": "Click 'X' sound, emphasis on 'tim'"
    },
    {
        "id": "statement_1",
        "sentence": "Noppali na.",
        "translation": "It is good.",
        "difficulty": "intermediate",
        "phonetic_hint": "nop-pa-lee na",
        "pronunciation_tips": "Double 'p' sound, rolling 'l' optional"
    },
    {
        "id": "question_2",
        "sentence": "Fan la?",
        "translation": "When?",
        "difficulty": "intermediate",
        "phonetic_hint": "fan la",
        "pronunciation_tips": "Clear 'f' sound, short 'a' vowel"
    }
]

def call_nemotron(prompt, temperature=0.3):
    """Make a call to Nemotron LLM for intelligent analysis."""
    if not nemotron_config:
        return None

    try:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": nemotron_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Wolof language teacher and pronunciation coach. Provide concise, encouraging feedback on pronunciation accuracy. Focus on specific sounds and give actionable advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 300
        }

        response = requests.post(
            f"{nemotron_config['api_base']}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"Nemotron API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error calling Nemotron: {e}")
        return None

def analyze_pronunciation_with_nemotron(phrase, user_transcription, phrase_info):
    """Use Nemotron to analyze pronunciation and provide intelligent feedback."""

    prompt = f"""
Analyze this Wolof pronunciation practice:

TARGET PHRASE: "{phrase_info['sentence']}"
TRANSLATION: "{phrase_info['translation']}"
PHONETIC GUIDE: "{phrase_info.get('phonetic_hint', 'N/A')}"
TIPS: {phrase_info.get('pronunciation_tips', 'N/A')}

USER TRANSCRIPTION: "{user_transcription}"

Please analyze the pronunciation and provide:
1. A score from 0-100
2. Specific feedback on pronunciation quality
3. Encouraging advice for improvement

Format your response as JSON:
{{
    "score": <0-100>,
    "feedback": "<specific feedback>",
    "improvement_tips": "<actionable advice>"
}}
"""

    response = call_nemotron(prompt)

    if response:
        try:
            # Try to parse JSON response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return None

            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: extract score from text response
            try:
                score_match = response.search(r'(\d{1,3})/100')
                if score_match:
                    score = int(score_match.group(1))
                    return {
                        "score": min(100, max(0, score)),
                        "feedback": response[:200] + "..." if len(response) > 200 else response,
                        "improvement_tips": "Keep practicing with the phonetic guides!"
                    }
            except:
                pass

    return None

@app.route('/')
def index():
    """Serve the game frontend."""
    return send_from_directory('static', 'index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": nemotron_config is not None,
        "phrases_loaded": len(GAME_PHRASES) > 0,
        "version": "nemotron-integrated",
        "nemotron_available": nemotron_config is not None
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
    """Score pronunciation using Nemotron for intelligent feedback."""

    phrase_id = request.form.get('phrase_id')

    # Find the target phrase
    phrase_info = next((p for p in GAME_PHRASES if p['id'] == phrase_id), None)
    if not phrase_info:
        return jsonify({"error": "Phrase not found"}), 404

    # Generate mock transcription (in real implementation, this would come from ASR)
    user_transcription = request.form.get('transcription', generate_mock_transcription(phrase_info['sentence']))

    # Try to get Nemotron analysis
    nemotron_result = analyze_pronunciation_with_nemotron(
        phrase_info['sentence'],
        user_transcription,
        phrase_info
    )

    if nemotron_result:
        # Use Nemotron's intelligent scoring
        return jsonify({
            "score": nemotron_result["score"],
            "feedback": nemotron_result["feedback"],
            "improvement_tips": nemotron_result.get("improvement_tips", ""),
            "user_transcription": user_transcription,
            "reference_sentence": phrase_info["sentence"],
            "translation": phrase_info["translation"],
            "model": "nemotron-nano-9b",
            "phonetic_hint": phrase_info.get("phonetic_hint", ""),
            "difficulty": phrase_info.get("difficulty", "unknown")
        })
    else:
        # Fallback to basic scoring
        base_score = calculate_basic_score(phrase_info['sentence'], user_transcription)
        feedback = generate_basic_feedback(base_score)

        return jsonify({
            "score": base_score,
            "feedback": feedback,
            "improvement_tips": phrase_info.get("pronunciation_tips", "Listen carefully to the pronunciation guide."),
            "user_transcription": user_transcription,
            "reference_sentence": phrase_info["sentence"],
            "translation": phrase_info["translation"],
            "model": "basic-fallback",
            "phonetic_hint": phrase_info.get("phonetic_hint", ""),
            "difficulty": phrase_info.get("difficulty", "unknown")
        })

def generate_mock_transcription(target_phrase):
    """Generate a mock transcription with some errors to simulate real ASR."""
    # Add some common pronunciation variations
    variations = [
        target_phrase,  # Perfect match
        target_phrase.replace("?", ""),  # Missing question mark
        target_phrase.replace("ng", "n"),  # Common ng sound issue
        target_phrase.replace("aa", "a"),  # Vowel length issue
    ]
    return random.choice(variations)

def calculate_basic_score(target, user_input):
    """Basic similarity scoring."""
    if target.lower() == user_input.lower():
        return random.randint(85, 95)
    elif target.lower().replace("?", "") == user_input.lower().replace("?", ""):
        return random.randint(75, 85)
    else:
        # Count character matches
        matches = sum(1 for a, b in zip(target.lower(), user_input.lower()) if a == b)
        similarity = matches / max(len(target), len(user_input))
        return int(similarity * 70) + random.randint(10, 20)

def generate_basic_feedback(score):
    """Generate basic feedback based on score."""
    if score >= 90:
        return "Excellent! Near-native pronunciation! 🎉"
    elif score >= 75:
        return "Great job! Very good pronunciation! 👏"
    elif score >= 60:
        return "Good effort! Keep practicing! 💪"
    elif score >= 40:
        return "Getting there! Listen carefully and try again! 🎧"
    else:
        return "Try again! Pay attention to the phonetic guide! 📢"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 49152))
    print(f"🚀 Wolof Game Server with Nemotron AI starting on http://localhost:{port}")
    if nemotron_config:
        print(f"✅ Nemotron connected at {nemotron_config['api_base']}")
    else:
        print("⚠️  Nemotron not available - using fallback scoring")
    app.run(host='0.0.0.0', port=port, debug=False)