"""
Streamlit dashboard for deepfake detection.
Launch: streamlit run app.py
"""

import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image


@st.cache_resource
def load_all_models():
    """Load and cache all models in memory for fast inference."""
    models = {}
    try:
        from tensorflow.keras.models import load_model
        models['cnn_eyes'] = load_model('saved_models/cnn_eyes.h5')
        models['cnn_nose'] = load_model('saved_models/cnn_nose.h5')
        models['cnn_chin'] = load_model('saved_models/cnn_chin.h5')
        models['cnn_ears'] = load_model('saved_models/cnn_ears.h5')
        models['posture'] = load_model('saved_models/posture_model.h5')
    except:
        pass
    return models


def _models_status():
    try:
        from models import models_available

        return models_available()
    except Exception:
        keys = ["eyes", "nose", "chin", "ears", "face", "posture"]
        return {k: False for k in keys}


def _predict_deepfake(image, demo_mode=False):
    from ensemble import predict_deepfake

    return predict_deepfake(image, demo_mode=demo_mode)


def _predict_deepfake_video(video_path, demo_mode=False):
    from ensemble import predict_deepfake_video

    return predict_deepfake_video(video_path, demo_mode=demo_mode)


def _predict_demo(image=None, video_path=None):
    from ensemble import predict_deepfake_demo

    return predict_deepfake_demo(image=image, video_path=video_path)


def _is_video_file(uploaded):
    if hasattr(uploaded, "type") and uploaded.type.startswith("video"):
        return True
    name = getattr(uploaded, "name", "") or ""
    return name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def _save_uploaded_video(uploaded):
    suffix = Path(getattr(uploaded, "name", "video.mp4")).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "saved_models" / "metrics.json"
SAVED_DIR = ROOT / "saved_models"

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '''
    <style>
    .hero-panel {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(168, 85, 247, 0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .hero-panel h1 {
        margin: 0 0 8px;
        font-size: 2.7rem;
    }
    .hero-panel .lead {
        margin: 0;
        color: #d1d5db;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .model-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 10px 14px;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .pill-ok { background: #16a34a; color: white; }
    .pill-muted { background: rgba(148, 163, 184, 0.16); color: #f8fafc; }
    .pill-note { background: rgba(59, 130, 246, 0.14); color: #e0f2fe; }
    .status-note {
        margin-top: 16px;
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    .section-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 24px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .summary-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 18px;
    }
    .summary-card {
        background: rgba(31, 41, 55, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 18px;
        padding: 18px;
    }
    .summary-card strong { color: #f8fafc; }
    .summary-card p { margin: 8px 0 0; color: #94a3b8; }
    </style>
    ''',
    unsafe_allow_html=True,
)

MODEL_LABELS = {
    "eyes": "Eyes CNN",
    "nose": "Nose CNN",
    "chin": "Chin CNN",
    "ears": "Ears CNN",
    "face": "CViT Face",
    "posture": "Posture MLP",
    "video": "Video Ensemble",
}


def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_votes_chart(preds):
    """Matplotlib bar chart (avoids Streamlit/Altair typing_extensions conflict)."""
    labels = [MODEL_LABELS.get(k, k) for k in preds]
    scores = [1 if v == "FAKE" else 0 for v in preds.values()]
    colors = ["#dc3545" if s else "#28a745" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Vote (1 = FAKE, 0 = REAL)")
    ax.set_title("Model votes")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)


def render_verdict_badge(result):
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    demo = result.get("demo_mode", False)

    if demo:
        st.warning(result.get("message", "Demo mode active."))

    if verdict == "FAKE":
        st.markdown(
            f"""
            <div style="background:#dc3545;color:white;padding:24px;border-radius:12px;
            text-align:center;font-size:2rem;font-weight:bold;">
            🔴 FAKE — {confidence:.1f}% confidence
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict == "REAL":
        st.markdown(
            f"""
            <div style="background:#28a745;color:white;padding:24px;border-radius:12px;
            text-align:center;font-size:2rem;font-weight:bold;">
            🟢 REAL — {confidence:.1f}% confidence
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Could not determine verdict. {result.get('error', '')}")


def render_model_status(status):
    status_items = []
    for name in ["eyes", "nose", "chin", "ears", "face", "posture"]:
        label = MODEL_LABELS.get(name, name)
        if status.get(name):
            status_items.append(f"<div class='status-pill pill-ok'>{label}</div>")
        elif name in ("face", "posture"):
            status_items.append(f"<div class='status-pill pill-muted'>{label} (trainable)</div>")
        else:
            status_items.append(f"<div class='status-pill pill-note'>{label} (optional)</div>")
    st.markdown("<div class='model-status-grid'>" + "".join(status_items) + "</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='status-note'>Face and posture are part of the full ensemble and can be generated by running local training.</div>",
        unsafe_allow_html=True,
    )


def page_detection():
    st.markdown(
        "<div class='hero-panel'><h1>Deepfake Detection System</h1><p class='lead'>Upload an image or video to detect real vs fake faces with a hybrid ensemble. If the full ensemble is not available, the HuggingFace demo model delivers instant pretrained inference.</p></div>",
        unsafe_allow_html=True,
    )

    status = _models_status()
    trained_count = sum(status.values())
    if trained_count == 0:
        st.info("No region models are deployed locally. The HuggingFace demo model is the fastest way to get instant results.")

    col_demo, col_status = st.columns([1, 2])
    with col_demo:
        demo_mode = st.checkbox("Use HuggingFace demo model", value=False)
    with col_status:
        render_model_status(status)

    uploaded = st.file_uploader(
        "Upload a face image or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv", "webm"],
    )

    if uploaded:
        is_video = _is_video_file(uploaded)
        if is_video:
            video_path = _save_uploaded_video(uploaded)
            st.video(video_path)
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Detect", type="primary", width="stretch"):
            with st.spinner("Running inference..."):
                if demo_mode:
                    if is_video:
                        result = _predict_demo(video_path=video_path)
                    else:
                        result = _predict_demo(image=np.array(image))
                elif is_video:
                    result = _predict_deepfake_video(video_path, demo_mode=False)
                    if not any(status.values()):
                        result = _predict_demo(video_path=video_path)
                        st.warning("No trained models found — showing HuggingFace demo output.")
                else:
                    result = _predict_deepfake(np.array(image), demo_mode=False)
                    if not any(status.values()):
                        result = _predict_demo(image=np.array(image))
                        st.warning("No trained models found — showing HuggingFace demo output.")

            render_verdict_badge(result)

            preds = result.get("model_predictions", {})
            if preds:
                st.subheader("Model votes")
                render_votes_chart(preds)

                flagged = result.get("flagged_regions", [k for k, v in preds.items() if v == "FAKE"])
                if flagged:
                    st.subheader("Flagged regions")
                    st.write(", ".join(MODEL_LABELS.get(r, r) for r in flagged))
                else:
                    st.success("No regions flagged as FAKE.")

                with st.expander("Raw JSON response"):
                    st.json(result)
    elif demo_mode:
        st.info("Upload an image or video to run the HuggingFace demo model.")


def run_local_training():
    cmd = [sys.executable, str(ROOT / "train.py"), "--allow-synthetic"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def page_performance():
    st.title("Model Performance")
    if st.button("Train full ensemble locally", help="Runs train.py locally to create face/posture and region models."):
        with st.spinner("Training full model ensemble locally..."):
            code, out, err = run_local_training()
        if code == 0:
            st.success("Training finished successfully. Reload the page to refresh metrics.")
        else:
            st.error("Training failed. See logs below.")
        with st.expander("Training output"):
            st.text(out + "\n" + err)

    metrics = load_metrics()

    if not metrics:
        st.warning("Model not trained — run `python train.py` to generate metrics.")
        st.code("pip install -r requirements.txt\npython train.py", language="bash")
        return

    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Model": MODEL_LABELS.get(name, name),
                "Accuracy": f"{m.get('accuracy', 0):.3f}",
                "AUC": f"{m['auc']:.3f}" if m.get("auc") is not None else "N/A",
            }
        )
    st.subheader("Test accuracy per model")
    st.table(pd.DataFrame(rows))

    st.subheader("Confusion matrices")
    cols = st.columns(2)
    for i, (name, m) in enumerate(metrics.items()):
        cm = np.array(m.get("confusion_matrix", [[0, 0], [0, 0]]))
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["REAL", "FAKE"],
                yticklabels=["REAL", "FAKE"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(MODEL_LABELS.get(name, name))
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("ROC curves (all models)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in metrics.items():
        fpr = m.get("roc_fpr", [])
        tpr = m.get("roc_tpr", [])
        if fpr and tpr:
            label = MODEL_LABELS.get(name, name)
            auc = m.get("auc")
            auc_str = f" (AUC={auc:.3f})" if auc is not None else ""
            ax.plot(fpr, tpr, label=f"{label}{auc_str}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — FAKE class")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.sidebar.title("Navigation")
    st.sidebar.caption("App build: matplotlib charts (no Altair)")
    page = st.sidebar.radio("Go to", ["Detection", "Model Performance"])

    if page == "Detection":
        page_detection()
    else:
        page_performance()


if __name__ == "__main__":
    main()
