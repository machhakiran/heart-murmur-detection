"""
CardioMurmur AI - Web Application
AI-Powered Heart Murmur Detection System using Audio Signal Processing and BiLSTM.
"""
import io
import os
import streamlit as st
import numpy as np

# Ensure local writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".matplotlib_cache"))

from src.config import CONFIG
from src.inference import MurmurPredictor
from src.visualizer import AudioVisualizer

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioMurmur AI | Heart Murmur Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .badge-normal {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-murmur {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-arrhythmia {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .disclaimer-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #94a3b8;
        border-radius: 0 8px 8px 0;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Predictor Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model from Hugging Face Hub / Local cache...")
def get_predictor(hf_repo_id: str):
    """Cache and load the predictor to avoid reloading on every re-render."""
    return MurmurPredictor(hf_repo_id=hf_repo_id)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.title("CardioMurmur AI")
    st.caption("Deep Learning Phonocardiogram Auscultation")
    st.markdown("---")

    st.subheader("Model Configuration")
    hf_repo_input = st.text_input(
        "Hugging Face Model Repo",
        value=os.getenv("HF_MODEL_REPO", CONFIG.default_hf_repo),
        help="Repository ID on Hugging Face Model Hub containing heart_murmur_lstm.pt"
    )

    # Initialize predictor
    predictor = get_predictor(hf_repo_input.strip() if hf_repo_input else None)

    st.markdown("**Model Source:**")
    st.info(f"📍 {predictor.model_source}")
    st.markdown(f"**Compute Device:** `{predictor.device.upper()}`")

    st.markdown("---")
    st.subheader("Signal Pipeline Specs")
    st.markdown(f"- **Target Sampling Rate:** `{CONFIG.audio.sample_rate} Hz`")
    st.markdown(f"- **Clip Analysis Window:** `{CONFIG.audio.duration} seconds`")
    st.markdown(f"- **Feature Dimension:** `60 Acoustic Bins`")
    st.markdown(f"- **Architecture:** `Bidirectional LSTM + Attention`")

    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("🔗 [Kaggle Heartbeat Dataset](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound)")
    st.markdown("🔗 [Hugging Face Model Hub](https://huggingface.co/models)")
    st.markdown("🔗 [GitHub Repository](https://github.com)")


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🫀 AI-Powered Heart Murmur Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated phonocardiogram (PCG) acoustic feature analysis and deep learning sequence classification for preliminary cardiac triage.</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Audio Input Section
# -----------------------------------------------------------------------------
st.markdown("### 1. Audio Recording Input")

input_mode = st.radio(
    "Select Sound Source:",
    options=["Clinical Benchmark Samples (Ready to Test)", "Upload Heart Sound (.wav, .mp3, .ogg)"],
    horizontal=True
)

audio_bytes = None
audio_source_label = ""

sample_paths = {
    "Normal Heartbeat (Regular S1/S2 Lub-Dub)": "samples/normal_sample.wav",
    "Systolic Heart Murmur (Turbulent Flow)": "samples/murmur_sample.wav",
    "Extrasystole (Premature Beat / Arrhythmia)": "samples/extrasystole_sample.wav",
}

if input_mode == "Clinical Benchmark Samples (Ready to Test)":
    selected_sample = st.selectbox("Choose Benchmark Audio:", list(sample_paths.keys()))
    sample_path = sample_paths[selected_sample]

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()
        audio_source_label = selected_sample
    else:
        st.warning(f"Sample file `{sample_path}` not found. Run sample generator first.")

else:
    uploaded_file = st.file_uploader(
        "Upload phonocardiogram recording (digital stethoscope or microphone):",
        type=["wav", "mp3", "ogg"],
        help="Upload an audio clip of a heart sound recording."
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_source_label = uploaded_file.name

# -----------------------------------------------------------------------------
# Playback and Trigger Analysis
# -----------------------------------------------------------------------------
if audio_bytes is not None:
    st.markdown("---")
    st.markdown("### 2. Audio Playback & Signal Review")

    col_play, col_action = st.columns([2, 1])
    with col_play:
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"Audio Source: **{audio_source_label}**")

    with col_action:
        st.write("")
        analyze_btn = st.button("🔍 Run Diagnostic Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Extracting acoustic features and evaluating with BiLSTM..."):
            try:
                results = predictor.predict(io.BytesIO(audio_bytes))
                st.session_state["prediction_results"] = results
            except Exception as e:
                st.error(f"Error during audio processing: {str(e)}")

# -----------------------------------------------------------------------------
# Display Diagnostic Analysis Results
# -----------------------------------------------------------------------------
if "prediction_results" in st.session_state and audio_bytes is not None:
    results = st.session_state["prediction_results"]
    pred_class = results["predicted_class"]
    confidence = results["confidence"]
    probs = results["probabilities"]
    interpretation = results["interpretation"]
    y_signal = results["raw_audio"]
    sr = results["sample_rate"]
    mel_db, times, freqs = results["spectrogram"]

    st.markdown("---")
    st.markdown("### 3. Diagnostic Assessment")

    # Metric Banner Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Predicted Condition**")
        badge_cls = "badge-normal" if pred_class == "Normal" else ("badge-murmur" if pred_class == "Murmur" else "badge-arrhythmia")
        st.markdown(f'<h3>{interpretation["title"]}</h3>', unsafe_allow_html=True)
        st.markdown(f'<span class="{badge_cls}">{interpretation["severity"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Confidence Level**")
        st.markdown(f"<h3>{confidence * 100:.1f}%</h3>", unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Clinical Recommendation**")
        if pred_class == "Normal":
            st.success("Routine checkup. No acute cardiac murmur patterns detected.")
        elif pred_class == "Murmur":
            st.error("Cardiac follow-up advised: Echocardiogram evaluation recommended.")
        else:
            st.warning("Follow-up advised: 12-lead ECG review recommended.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Clinical Notes
    with st.expander("ℹ️ Clinical Description & Pathophysiology Details", expanded=True):
        st.write(interpretation["details"])

    # Visualizations
    st.markdown("### 4. Acoustic Signal & Sequence Visualizations")
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        # Probability Bar Chart
        fig_prob = AudioVisualizer.plot_probabilities(probs, pred_class)
        st.pyplot(fig_prob)

    with col_chart2:
        # Waveform Plot
        fig_wave = AudioVisualizer.plot_waveform(y_signal, sr, title=f"Phonocardiogram Signal - {pred_class}")
        st.pyplot(fig_wave)

    # Full Mel Spectrogram
    fig_spec = AudioVisualizer.plot_spectrogram(
        mel_db, times, freqs, title="Mel Spectrogram (Frequency vs. Time Energy Distribution)"
    )
    st.pyplot(fig_spec)

# Clinical Disclaimer Banner
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Medical Device Disclaimer:</strong> This application is developed for educational, screening, and research purposes.
    It does not constitute a certified medical diagnosis. Always consult a qualified cardiologist or healthcare professional
    for clinical decisions.
</div>
""", unsafe_allow_html=True)
