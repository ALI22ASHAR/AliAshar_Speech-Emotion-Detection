# 🎙️ Speech Emotion Detection from Audio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green.svg)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application that predicts human emotions from speech audio recordings in real time. Built with **Streamlit**, **Librosa**, and **Scikit-learn**, the application extracts acoustic MFCC features from speech clips and classifies them into distinct emotional states with confidence scores, waveform plots, and Mel-spectrogram visualizations.

---

## 1. Project Overview

Speech Emotion Recognition (SER) is a subfield of affective computing and natural language understanding that identifies emotional states from vocal characteristics regardless of verbal content. 

This project implements a complete machine learning pipeline:
- Ingestion and preprocessing of speech recordings.
- Acoustic feature extraction using **Mel-Frequency Cepstral Coefficients (MFCCs)**.
- Training an ensemble **Random Forest Classifier** with balanced class weighting.
- Providing an interactive **Streamlit** dashboard for users to upload WAV audio files, visualize speech signals, and view predicted emotions alongside complete class probability distributions.

---

## 2. Approach & Methodology

The project follows a modular, speaker-independent machine learning pipeline:

```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│  Audio Input    │ ───>  │  Audio Preprocessing │ ───>  │  Feature Extraction  │
│  (.WAV Format)  │       │  (Resample @ 22 kHz) │       │  (40 MFCCs × 4 Stats)│
└─────────────────┘       └──────────────────────┘       └──────────┬───────────┘
                                                                    │ (160 Features)
                                                                    ▼
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│ Streamlit UI    │ <───  │ Probabilities & Class│ <───  │    Random Forest     │
│ & Visualizations│       │ (Happy/Sad/Angry/...)│       │      Classifier      │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘
```

1. **Audio Standardization:** Audio signals are loaded and resampled at a uniform sample rate of **22,050 Hz**.
2. **Feature Extraction:** 40 Mel-Frequency Cepstral Coefficients (MFCCs) are computed across audio time frames. For each coefficient, four statistical summaries are extracted:
   - **Mean**: Central tendency of frequency bands
   - **Standard Deviation**: Pitch/energy variability
   - **Minimum**: Lower spectral bound
   - **Maximum**: Peak spectral energy
   - *Total feature vector size:* `40 × 4 = 160 numerical features per clip`.
3. **Actor-Independent Splitting:** To avoid data leakage and prevent the model from memorizing speaker-specific voice pitch, a speaker-based split is implemented:
   - **Training Set:** Actors 1 to 18 (504 clips)
   - **Testing Set:** Actors 19 to 24 (168 clips)
4. **Classification:** A Random Forest model evaluates the 160 acoustic features to generate prediction probabilities across all target emotion categories.
5. **Interactive Inference:** The Streamlit app runs inference on uploaded audio clips, generates confidence bar charts, and dynamically plots time-domain waveforms and Mel-spectrograms.

---

## 3. Technologies and Libraries Used

| Technology / Library | Purpose |
| :--- | :--- |
| **Python 3.9+** | Core programming language |
| **Streamlit** | Interactive web application dashboard and visualization UI |
| **Librosa** | Audio analysis, MFCC computation, and spectrogram rendering |
| **Scikit-learn** | Random Forest model training, evaluation metrics, and train/test split |
| **NumPy & Pandas** | Array manipulation, statistical feature aggregation, and tabular data |
| **Matplotlib & Seaborn** | Signal plotting (waveforms, Mel-spectrograms, confusion matrix) |
| **Joblib** | Serialization and loading of the trained model artifact |
| **SoundFile** | Audio reading and audio format processing |

---

## 4. Dataset Information

This project is trained and evaluated on the benchmark **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

- **Speakers:** 24 professional actors (12 female, 12 male) vocalizing lexically-matched statements with neutral North American accents.
- **Emotions Selected:** 4 primary classes:
  - 😠 **Angry** (192 clips)
  - 😄 **Happy** (192 clips)
  - 😢 **Sad** (192 clips)
  - 😐 **Neutral** (96 clips)
- **Total Dataset Size:** 672 speech audio clips.
- **Split Strategy:**
  - **Train split:** 504 samples (75%) from Actors 1–18
  - **Test split:** 168 samples (25%) from Actors 19–24 (unseen speakers)

### Class Distribution
![Class Distribution](assets/Class%20Distribution.png)

---

## 5. Models Used

### Primary Classifier: Random Forest Classifier
- **Algorithm:** Ensemble of decision trees with bootstrap aggregation (`RandomForestClassifier`).
- **Hyperparameters:**
  - `n_estimators = 300`: 300 ensemble estimators to ensure stable variance reduction.
  - `class_weight = "balanced"`: Automatically adjusts weights inversely proportional to class frequencies to handle the lower sample count of the Neutral class.
  - `random_state = 42`: Ensures deterministic reproducibility.
  - `n_jobs = -1`: Multi-threaded parallel processing during training.
- **Storage:** Persisted as a self-contained bundle via `joblib` (`speech_emotion_model.joblib`), storing both the trained model and associated class labels.

---

## 6. Results / Evaluation

The model was evaluated strictly on **unseen actors (Actors 19–24)** to simulate real-world generalization:

### Overall Metrics
- **Test Accuracy:** `60.12%`
- **Macro F1-Score:** `0.5404`
- **Weighted F1-Score:** `0.5900`

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Angry** | **0.72** | **0.81** | **0.76** | 48 |
| **Happy** | **0.63** | **0.60** | **0.62** | 48 |
| **Sad** | **0.56** | **0.60** | **0.58** | 48 |
| **Neutral** | **0.25** | **0.17** | **0.20** | 24 |

### Confusion Matrix
![Confusion Matrix](assets/Confusion%20Matrix.png)

### Key Observations:
- **Angry speech** achieved the highest detection accuracy (F1: `0.76`, Recall: `0.81`), as it exhibits sharp energy peaks, higher pitch variation, and distinct spectral centroids.
- **Neutral speech** presented the greatest challenge (F1: `0.20`), frequently exhibiting subtle acoustic overlaps with subdued sad speech.

---

## 7. Installation Requirements

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed
- Git installed
- *(Optional but recommended)* A virtual environment (`venv` or `conda`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ALI22ASHAR/Speech-Emotion-Detection.git
cd Speech-Emotion-Detection
```

### Step 2: Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. How to Run the Project

### Running the Streamlit Web App
Launch the interactive dashboard with:
```bash
streamlit run app.py
```
Once started, the application will automatically open in your default browser at `http://localhost:8501`.

### Using the App:
1. **Upload Audio:** Upload any standard speech clip in `.wav` format.
2. **Audio Playback:** Use the built-in media player to listen to the speech sample.
3. **Signal Inspection:** Examine the dynamic **Waveform** and **Mel-Spectrogram**.
4. **Emotion Prediction:** Review the predicted dominant emotion along with the confidence breakdown across all 4 categories.

### Exploring the Notebook
To explore data preprocessing, feature engineering, and model training:
```bash
jupyter notebook notebook/speech-emotion-detection.ipynb
```

---

## 9. Any Other Important Information

### Project Directory Structure
```text
Speech-Emotion-Detection/
│
├── assets/                          # Visualization plots and charts
│   ├── Class Distribution.png
│   ├── Confusion Matrix.png
│   ├── Mel Spectogram.png
│   └── Waveform.png
│
├── notebook/                        # Research and model exploration
│   └── speech-emotion-detection.ipynb
│
├── app.py                           # Streamlit web application
├── speech_emotion_model.joblib      # Serialized trained model artifact
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Ignored files and directories
└── README.md                        # Documentation
```

### Audio Input Recommendations:
- Input audio must be in **WAV** format.
- For best results, use clips of **2 to 5 seconds** containing clear speech with minimal background noise.
- If using stereo audio, Librosa automatically converts it to mono during feature extraction.

### Making the GitHub Repository Public:
If your repository is currently private, make it public by following these steps:
1. Go to your repository on GitHub: `https://github.com/ALI22ASHAR/Speech-Emotion-Detection`
2. Click on **Settings** (tab at the top right of the repo).
3. Scroll down to the bottom **Danger Zone** section.
4. Under **Change repository visibility**, click **Change visibility** ➔ Select **Make public**.
5. Confirm the selection to ensure your repository is accessible to everyone.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).