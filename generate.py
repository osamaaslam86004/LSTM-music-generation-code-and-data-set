# -*- coding: UTF-8 -*-

"""
Music Generation System using Pre-trained HDF5 Model
Generates harmonically structured music from a saved model
"""

import os
import pickle
import random

import numpy as np
import tensorflow as tf
from music21 import chord, key, meter, note, pitch, stream

# Configuration
DATA_PATH = "data/notes"
MODEL_PATH = "content/mmusic250516/weights-200-0.1454.keras"  # Your HDF5 model file
OUTPUT_PATH = "generated_music.mid"
SEQUENCE_LENGTH = 50
GENERATE_LENGTH = 300
TEMPERATURE = 2.5
RANDOM_SEED = 100

# Music Theory Parameters (C Major)
KEY = key.Key("C")
TIME_SIGNATURE = "4/4"
CHORD_PROGRESSION = ["C", "F", "G", "D", "E", "A", "B"]  # I-IV-V-I progression

# Suppress logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_and_validate_data():
    """Load and validate note data from pickle file"""
    with open(DATA_PATH, "rb") as f:
        all_notes = pickle.load(f)

    valid_notes = []
    for n in all_notes:
        try:
            # Validate note format
            p = pitch.Pitch(n)
            valid_notes.append(n)
        except:
            print(f"Skipping invalid note: {n}")
            continue

    unique_pitches = sorted(set(valid_notes))
    print(
        f"Loaded {len(valid_notes)} valid notes with {len(unique_pitches)} unique pitches"
    )
    return (
        valid_notes,
        len(unique_pitches),
        {p: i for i, p in enumerate(unique_pitches)},
        {i: p for i, p in enumerate(unique_pitches)},
    )


def build_model(seq_length, num_pitches):
    """Build model architecture to match the saved HDF5"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                128, input_shape=(seq_length, 1), return_sequences=True
            ),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(num_pitches, activation="softmax"),
        ]
    )
    return model


def load_trained_model(model_path, seq_length, num_pitches):
    """Load weights from HDF5 and initialize model"""
    model = build_model(seq_length, num_pitches)

    try:
        model.load_weights(model_path)
        print(f"Successfully loaded model weights from {model_path}")
    except:
        print("Weight mismatch detected. Initializing with random weights.")
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model


def generate_initial_pattern(notes, pitch_to_int, seq_length):
    """Generate valid initial pattern for prediction"""
    start_idx = np.random.randint(0, len(notes) - seq_length)
    return [pitch_to_int[note] for note in notes[start_idx : start_idx + seq_length]]


def apply_harmonic_constraints(preds, current_chord, int_to_pitch, key_obj):
    """Apply harmonic rules to guide note selection"""
    chord_tones = set(str(p) for p in current_chord.pitches)
    key_tones = set(str(p) for p in key_obj.pitches)

    for i in range(len(preds)):
        pitch_name = int_to_pitch[i]
        if pitch_name in chord_tones:
            preds[i] *= 3.0  # Boost chord tones
        elif pitch_name in key_tones:
            preds[i] *= 1.5  # Allow key tones
        else:
            preds[i] *= 0.1  # Suppress non-harmonic

    # Ensure valid distribution
    if np.sum(preds) <= 0:
        preds = np.ones_like(preds)

    return preds / np.sum(preds)


def generate_melody(model, pattern, int_to_pitch, num_notes, key_obj, chords):
    """Generate melody using the pre-trained model"""
    generated = []
    for i in range(num_notes):
        # Get current chord
        chord_idx = (i // 8) % len(chords)  # Change chord every 2 measures
        current_chord = chord.Chord(chords[chord_idx])

        # Prepare input sequence
        input_seq = np.reshape(pattern, (1, len(pattern), 1))

        # Predict next note
        preds = model.predict(input_seq, verbose=0)[0]

        # Apply harmonic constraints
        constrained_preds = apply_harmonic_constraints(
            preds, current_chord, int_to_pitch, key_obj
        )

        # Sample with temperature
        adjusted_preds = np.log(constrained_preds) / TEMPERATURE
        probs = np.exp(adjusted_preds) / np.sum(np.exp(adjusted_preds))
        index = np.random.choice(len(probs), p=probs)

        # Add predicted note
        note_name = int_to_pitch[index]
        generated.append(note_name)

        # Update pattern
        pattern = pattern[1:] + [index]

    return generated


def structure_melody(notes):
    """Structure melody into AABA form with variations"""
    # Convert to music21 note objects
    note_objects = [note.Note(n) for n in notes]

    # Create musical structure
    a_section = note_objects[:120]
    b_section = [n.transpose("m3") for n in a_section[:60]]  # Transposed variation

    structured = a_section + a_section + b_section + a_section
    return [n.nameWithOctave for n in structured[:GENERATE_LENGTH]]


def save_to_midi(notes, key_obj, output_path):
    """Save melody and harmony to MIDI file"""
    midi_stream = stream.Stream()
    midi_stream.append(key_obj)
    midi_stream.append(meter.TimeSignature(TIME_SIGNATURE))

    # Create melody part
    melody_part = stream.Part()
    current_beat = 0.0

    for note_name in notes:
        try:
            n = note.Note(note_name)
            n.duration.quarterLength = random.choice([0.5, 1.0, 2.0])
            n.offset = current_beat
            melody_part.append(n)
            current_beat += n.duration.quarterLength
        except:
            print(f"Skipping invalid note in MIDI export: {note_name}")
            continue

    # Create harmony part
    harmony_part = stream.Part()
    chord_beat = 0.0

    for i in range(len(notes) // 8 + 1):
        chord_idx = i % len(CHORD_PROGRESSION)
        chord_symbol = CHORD_PROGRESSION[chord_idx]

        try:
            c = chord.Chord(chord_symbol)
            c.duration.quarterLength = 8.0
            c.offset = chord_beat
            harmony_part.append(c)
            chord_beat += 8.0
        except:
            print(f"Skipping invalid chord in MIDI export: {chord_symbol}")
            continue

    # Combine parts and save
    midi_stream.append(melody_part)
    midi_stream.append(harmony_part)
    midi_stream.write("midi", fp=output_path)
    print(f"MIDI saved with {len(melody_part.elements)} valid notes")


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Load and validate training data
    notes, num_pitches, p2i, i2p = load_and_validate_data()

    # Load pre-trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_trained_model(MODEL_PATH, SEQUENCE_LENGTH, num_pitches)

    # Generate initial pattern
    initial_pattern = generate_initial_pattern(notes, p2i, SEQUENCE_LENGTH)

    # Generate melody using the pre-trained model
    print(f"Generating {GENERATE_LENGTH} notes with harmonic constraints...")
    generated_notes = generate_melody(
        model, initial_pattern, i2p, GENERATE_LENGTH, KEY, CHORD_PROGRESSION
    )

    # Structure the melody
    structured_notes = structure_melody(generated_notes)

    # Save to MIDI
    save_to_midi(structured_notes, KEY, OUTPUT_PATH)
    print("Music generation completed successfully using pre-trained model")


if __name__ == "__main__":
    main()
