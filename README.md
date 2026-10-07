# Speech Emotion Detection from Audio

A machine learning application that predicts emotions from short speech clips.

## Overview

This project uses the RAVDESS dataset to classify four emotions:

- Angry
- Happy
- Sad
- Neutral

A Random Forest classifier is trained using MFCC-based audio features.

## Dataset

RAVDESS contains speech recordings from 24 professional actors.

Only four emotions were used:

- Neutral
- Happy
- Sad
- Angry

The final dataset contains 672 clips.

Class distribution:

- Angry: 192
- Happy: 192
- Sad: 192
- Neutral: 96

## Actor-Based Train/Test Split

To evaluate the model on unseen speakers:

- Actors 1–18 → Training
- Actors 19–24 → Testing

Training samples: 504

Testing samples: 168

## Feature Extraction

MFCC features were extracted using Librosa.

For each of the 40 MFCC coefficients, the following statistics were calculated:

- Mean
- Standard deviation
- Minimum
- Maximum

Therefore:

40 × 4 = 160 features per audio clip.

## Model

Random Forest Classifier

Configuration:

- n_estimators = 300
- class_weight = balanced
- random_state = 42

## Results

Accuracy: 60.12%

Macro F1: 0.5404

### Class Performance

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Angry | 0.72 | 0.81 | 0.76 |
| Happy | 0.63 | 0.60 | 0.62 |
| Sad | 0.56 | 0.60 | 0.58 |
| Neutral | 0.25 | 0.17 | 0.20 |

## Main Observation

The model performs best on angry speech.

The main confusion occurs between neutral and sad speech. Neutral speech has relatively low recall, suggesting that the current MFCC-based representation has difficulty distinguishing neutral speech from subdued sad speech.

## Dashboard

The Streamlit dashboard allows users to:

1. Upload a WAV audio file
2. Listen to the audio
3. View its waveform
4. Predict the emotion
5. View probabilities for all four emotions

## Project Structure

```text
speech-emotion-detection/
├── app.py
├── speech_emotion_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
└── notebook/