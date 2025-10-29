#!/usr/bin/env python3
"""
Create curated game dataset from Zenodo Wolof TTS
Extract 100 diverse phrases with audio paths
"""

import pandas as pd
import json
from pathlib import Path

# Load female training set (more data available)
tsv_path = "/home/mithranmohanraj/Documents/shortest-hack/datasets/zenodo-wolof-tts/female_train.tsv"
audio_base = "/home/mithranmohanraj/Documents/shortest-hack/datasets/zenodo-wolof-tts/data-commonvoice"

print(f"📂 Loading dataset from {tsv_path}...")
df = pd.read_csv(tsv_path, sep='\t')

print(f"✅ Loaded {len(df)} total phrases")

# Select diverse phrases for the game
game_set = df[
    (df['word_count'].between(3, 8)) &  # Not too short or long
    (df['caracter_count'] < 60) &       # Manageable length
    (df['is_valid'] == True)            # Only validated clips
].sample(100, random_state=42)

print(f"🎯 Selected {len(game_set)} game phrases")

# Create game phrases with full paths
game_phrases = []
for idx, row in game_set.iterrows():
    # Construct full audio path
    audio_path = f"{audio_base}/{row['path']}"

    # Check if file exists
    if Path(audio_path).exists():
        phrase = {
            "id": row['id'],
            "sentence": row['sentence'],
            "word_count": int(row['word_count']),
            "char_count": int(row['caracter_count']),
            "audio_path": audio_path,
            "difficulty": "beginner" if row['word_count'] <= 4 else "intermediate" if row['word_count'] <= 6 else "advanced"
        }
        game_phrases.append(phrase)

print(f"✅ Found {len(game_phrases)} phrases with valid audio files")

# Save to JSON
output_path = "game_phrases.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(game_phrases, f, ensure_ascii=False, indent=2)

print(f"💾 Saved to {output_path}")

# Print sample phrases
print("\n📝 Sample phrases:")
for phrase in game_phrases[:5]:
    print(f"  - {phrase['sentence']} ({phrase['difficulty']}, {phrase['word_count']} words)")

print(f"\n📊 Statistics:")
print(f"  - Average words: {sum(p['word_count'] for p in game_phrases) / len(game_phrases):.1f}")
print(f"  - Beginner: {sum(1 for p in game_phrases if p['difficulty'] == 'beginner')}")
print(f"  - Intermediate: {sum(1 for p in game_phrases if p['difficulty'] == 'intermediate')}")
print(f"  - Advanced: {sum(1 for p in game_phrases if p['difficulty'] == 'advanced')}")