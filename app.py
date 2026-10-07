import os
import tempfile

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "speech_emotion_model.joblib"

SAMPLE_RATE = 22050
N_MFCC = 40


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    artifact = joblib.load(MODEL_PATH)

    return artifact


artifact = load_model()

model = artifact["model"]
labels = artifact["labels"]

# Make sure labels are a normal Python list
labels = list(labels)


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.min(mfcc, axis=1),
        np.max(mfcc, axis=1)
    ])

    return features


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎙️ Speech Emotion Detection")

st.markdown(
    """
    Upload a short WAV audio clip and the model will
    predict the speaker's emotion.
    """
)

st.info(
    "Supported emotions: Angry, Happy, Sad, Neutral"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.write(
        "This application uses a Random Forest classifier "
        "trained on MFCC-based audio features."
    )

    st.write("**Feature extraction:**")

    st.write(
        "- 40 MFCC coefficients\n"
        "- Mean\n"
        "- Standard deviation\n"
        "- Minimum\n"
        "- Maximum"
    )

    st.write("**Total features:** 160")

    st.write("**Training actors:** 1–18")
    st.write("**Testing actors:** 19–24")


# --------------------------------------------------
# AUDIO UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# --------------------------------------------------
# PROCESS AUDIO
# --------------------------------------------------

if uploaded_file is not None:

    st.subheader("Uploaded Audio")

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name


    try:

        # ------------------------------------------
        # LOAD AUDIO
        # ------------------------------------------

        audio, sr = librosa.load(
            temp_path,
            sr=SAMPLE_RATE
        )


        # ------------------------------------------
        # WAVEFORM
        # ------------------------------------------

        st.subheader("Waveform")

        fig, ax = plt.subplots(
            figsize=(12, 3)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

        plt.close(fig)


        # ------------------------------------------
        # EXTRACT FEATURES
        # ------------------------------------------

        features = extract_features(
            temp_path
        )

        # Reshape for model
        features_for_model = features.reshape(
            1,
            -1
        )


        # ------------------------------------------
        # VERIFY FEATURES
        # ------------------------------------------

        if features.shape[0] != 160:

            st.error(
                f"Expected 160 features but got "
                f"{features.shape[0]}."
            )

            st.stop()


        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        prediction = model.predict(
            features_for_model
        )[0]


        probabilities = model.predict_proba(
            features_for_model
        )[0]


        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"Predicted Emotion: {prediction.upper()}"
        )


        # ------------------------------------------
        # PROBABILITIES
        # ------------------------------------------

        st.subheader("Emotion Probabilities")

        probability_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probabilities
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        )


        st.bar_chart(
            probability_df.set_index("Emotion")
        )


        st.dataframe(
            probability_df.style.format(
                {
                    "Probability": "{:.2f}%"
                }
            ),
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Could not process the audio: {e}"
        )


    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)



st.divider()

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Accuracy",
        "60.12%"
    )

with col2:

    st.metric(
        "Macro F1",
        "0.5404"
    )


st.subheader("Confusion Matrix")

if os.path.exists(
    "assets/Confusion Matrix.png"
):

    st.image(
        "assets/Confusion Matrix.png",
        use_container_width=True
    )


st.subheader("Class Distribution")

if os.path.exists(
    "assets/Class Distribution.png"
):

    st.image(
        "assets/Class Distribution.png",
        use_container_width=True
    )