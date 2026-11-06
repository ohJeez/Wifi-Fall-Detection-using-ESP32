import os
import io
import h5py
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import BaselineCNN
from preprocessing import preprocess

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="WiFi Fall Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
}

.hero p {
    margin-top: .45rem;
    color: #cbd5e1;
    font-size: 1.05rem;
}

.result {
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #374151;
}

.result h2 {
    margin: 0;
    font-size: 2rem;
}

.small {
    color: #94a3b8;
    font-size: .9rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_PATH = "weighted_cnn_best.pth"
THRESHOLD = 0.26

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineCNN()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device

# ============================================================
# H5 READER
# ============================================================
def read_csi_h5(file_source):
    with h5py.File(file_source, "r") as f:
        if "CSI_amps" not in f:
            raise KeyError(
                'The H5 file does not contain the expected "CSI_amps" dataset.'
            )
        csi = f["CSI_amps"][:]

    if csi.shape != (52, 500, 1):
        raise ValueError(
            f"Expected CSI shape (52, 500, 1), but received {csi.shape}."
        )

    return csi

# ============================================================
# PREDICTION
# ============================================================
def predict(csi, model, device):
    processed = preprocess(csi)

    x = torch.tensor(
        processed,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    nonfall_prob = float(probs[0].item())
    fall_prob = float(probs[1].item())

    prediction = "FALL DETECTED" if fall_prob >= THRESHOLD else "NO FALL"

    return prediction, fall_prob, nonfall_prob, processed

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📡 WiFi-Based Human Fall Detection</h1>
    <p>
        ESP32 CSI signal → preprocessing → CNN → fall probability → detection
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ System")

if not os.path.exists(MODEL_PATH):
    st.sidebar.error(
        f"Missing {MODEL_PATH}. Put your trained model beside app.py."
    )
    st.stop()

model, device = load_model()

st.sidebar.success(f"Model loaded • {device}")

st.sidebar.markdown("### Model Configuration")
st.sidebar.write("Architecture: Weighted CNN")
st.sidebar.write("Input: 52 × 500 CSI")
st.sidebar.write("Classes: Fall / Non-Fall")
st.sidebar.write(f"Decision threshold: {THRESHOLD}")

st.sidebar.markdown("### Test Performance")
st.sidebar.metric("Accuracy", "89.04%")
st.sidebar.metric("Precision", "86.05%")
st.sidebar.metric("Recall", "92.06%")
st.sidebar.metric("F1 Score", "88.96%")
st.sidebar.metric("ROC-AUC", "96.17%")

# ============================================================
# INPUT MODE
# ============================================================
st.subheader("1. Select CSI Input")

input_mode = st.radio(
    "Input mode",
    ["Upload H5 file", "Demo sample"],
    horizontal=True
)

selected_name = None
uploaded = None

if input_mode == "Upload H5 file":
    uploaded = st.file_uploader(
        "Upload an ESP32 CSI H5 file",
        type=["h5"]
    )

else:
    demo_dir = "demo_samples"

    demo_files = [
        f for f in os.listdir(demo_dir)
        if f.lower().endswith(".h5")
    ]

    if not demo_files:
        st.warning(
            "No demo H5 files found. Put a few fall/non-fall H5 files "
            "inside the demo_samples folder."
        )
    else:
        selected_name = st.selectbox(
            "Choose a demonstration sample",
            sorted(demo_files)
        )

analyze = st.button(
    "🔍 ANALYZE CSI",
    type="primary",
    use_container_width=True
)

# ============================================================
# PROCESS
# ============================================================
if analyze:

    if input_mode == "Upload H5 file" and uploaded is None:
        st.warning("Please upload an H5 file first.")
        st.stop()

    if input_mode == "Demo sample" and selected_name is None:
        st.warning("Please add/select a demo H5 file first.")
        st.stop()

    try:
        if input_mode == "Upload H5 file":
            file_source = io.BytesIO(uploaded.getvalue())
            display_name = uploaded.name
        else:
            file_source = os.path.join(demo_dir, selected_name)
            display_name = selected_name

        csi = read_csi_h5(file_source)

        prediction, fall_prob, nonfall_prob, processed = predict(
            csi, model, device
        )

        st.session_state["result"] = {
            "name": display_name,
            "csi": csi,
            "processed": processed,
            "prediction": prediction,
            "fall_prob": fall_prob,
            "nonfall_prob": nonfall_prob,
        }

    except Exception as e:
        st.error(f"Could not process the CSI file: {e}")
        st.stop()

# ============================================================
# DISPLAY RESULT
# ============================================================
if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("2. Input Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CSI Shape", str(result["csi"].shape))

    with c2:
        st.metric("Subcarriers", "52")

    with c3:
        st.metric("Time Samples", "500")

    st.caption(f"Analyzed sample: {result['name']}")

    # --------------------------------------------------------
    # Raw CSI
    # --------------------------------------------------------
    st.subheader("3. CSI Signal")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(
        result["csi"].squeeze(-1),
        aspect="auto"
    )
    ax.set_title("Raw CSI Amplitude")
    ax.set_xlabel("Time Samples")
    ax.set_ylabel("Subcarriers")
    st.pyplot(fig, clear_figure=True)

    # --------------------------------------------------------
    # Processed CSI
    # --------------------------------------------------------
    st.subheader("4. Preprocessed CSI")

    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    ax2.imshow(
        result["processed"],
        aspect="auto"
    )
    ax2.set_title("Normalized Power-Based CSI")
    ax2.set_xlabel("Subcarriers")
    ax2.set_ylabel("Time Samples")
    st.pyplot(fig2, clear_figure=True)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    st.subheader("5. AI Detection Result")

    if result["prediction"] == "FALL DETECTED":
        st.error("🚨 FALL DETECTED")
    else:
        st.success("✅ NO FALL DETECTED")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Fall Probability",
            f"{result['fall_prob'] * 100:.2f}%"
        )

    with p2:
        st.metric(
            "Non-Fall Probability",
            f"{result['nonfall_prob'] * 100:.2f}%"
        )

    with p3:
        st.metric(
            "Threshold",
            f"{THRESHOLD * 100:.0f}%"
        )

    # Probability progress bars
    st.markdown("### Probability")

    st.write("Fall")
    st.progress(min(result["fall_prob"], 1.0))

    st.write("Non-Fall")
    st.progress(min(result["nonfall_prob"], 1.0))

    # --------------------------------------------------------
    # Probability chart
    # --------------------------------------------------------
    st.subheader("6. Prediction Distribution")

    chart_data = {
        "Class": ["Non-Fall", "Fall"],
        "Probability": [
            result["nonfall_prob"],
            result["fall_prob"]
        ]
    }

    st.bar_chart(
        chart_data,
        x="Class",
        y="Probability"
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------
    st.subheader("7. Decision Explanation")

    if result["fall_prob"] >= THRESHOLD:
        st.warning(
            f"The CNN produced a fall probability of "
            f"{result['fall_prob'] * 100:.2f}%. "
            f"This is above the decision threshold of "
            f"{THRESHOLD * 100:.0f}%, so the system reports "
            f"**FALL DETECTED**."
        )
    else:
        st.info(
            f"The CNN produced a fall probability of "
            f"{result['fall_prob'] * 100:.2f}%. "
            f"This is below the decision threshold of "
            f"{THRESHOLD * 100:.0f}%, so the system reports "
            f"**NO FALL**."
        )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------
    st.subheader("8. System Pipeline")

    st.code(
        """
ESP32 CSI
   ↓
52 × 500 CSI Window
   ↓
Power Normalization
   ↓
Standardization
   ↓
Weighted CNN
   ↓
Fall Probability
   ↓
Threshold = 0.26
   ↓
FALL / NO FALL
        """,
        language="text"
    )

else:
    st.info(
        "Upload an H5 CSI file or select a demo sample, "
        "then click **ANALYZE CSI**."
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Prototype: WiFi CSI-Based Human Fall Detection | "
    "Recorded CSI demonstration. Live ESP32 integration can be added later."
)
