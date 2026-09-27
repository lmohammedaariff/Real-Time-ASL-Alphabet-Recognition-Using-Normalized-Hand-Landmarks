"""SignVision Streamlit portfolio dashboard."""
from pathlib import Path
import streamlit as st
import pandas as pd
from config import SETTINGS, LETTERS
from src.data.dataset_collector import append_sample, count_samples
from src.data.dataset_loader import load_landmark_csv
from src.preprocessing.landmark_processor import normalize_landmarks

st.set_page_config(page_title="SignVision", page_icon="🤟", layout="wide")
st.markdown("""<style>.block-container{padding-top:2rem}.hero{padding:2rem;border-radius:16px;background:linear-gradient(120deg,#102d3d,#1f7368);color:white}</style>""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Home", "Real-Time Recognition", "Dataset Collection", "Model Evaluation", "About Project"])
st.sidebar.caption("Local-first ASL alphabet recognition")

if page == "Home":
    st.markdown('<div class="hero"><h1>SignVision</h1><h3>Real-Time Sign Language Alphabet Recognition</h3><p>Landmark-based ASL fingerspelling research and demonstration toolkit.</p></div>', unsafe_allow_html=True)
    st.subheader("From hand pose to letter")
    st.write("Webcam → 21 hand landmarks → translation/scale normalization → classifier → temporal confidence filter → A–Z")
    a,b,c = st.columns(3); a.metric("Alphabet", "A–Z"); b.metric("Hand landmarks", "21"); c.metric("Video processing", "Local")
    st.info("Static handshape recognition is not continuous sign-language translation. J and Z involve motion and need temporal modeling.")
elif page == "Real-Time Recognition":
    st.header("Real-Time Recognition")
    model_file = SETTINGS.model_dir / "best_model.joblib"
    if not model_file.exists():
        st.warning("No trained model was found. Import/collect labeled A–Z samples, then run `python -m src.training.train`.")
    else:
        try:
            from functools import partial
            from streamlit_webrtc import WebRtcMode, webrtc_streamer
            from src.inference.live_video import LivePredictionProcessor

            st.write("The camera starts when this page opens. Hold one hand sign in view and the prediction, confidence, FPS, landmarks, and hand box update continuously.")
            st.caption("Allow camera access in your browser if prompted. Frames are processed on this machine and are not recorded.")
            context = webrtc_streamer(
                key="signvision-live-camera",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=partial(LivePredictionProcessor, model_path=model_file),
                media_stream_constraints={"video": True, "audio": False},
                desired_playing_state=True,
                async_processing=True,
            )
            if not context.state.playing:
                st.info("Click START above to turn on the camera.")
        except ImportError:
            st.error("Live browser video needs streamlit-webrtc and av. Install requirements.txt and restart the app.")
    from src.word_builder.word_builder import WordBuilder
    from src.utils.speech import speak
    if "word_builder" not in st.session_state:
        st.session_state.word_builder = WordBuilder()
    builder = st.session_state.word_builder
    st.subheader("Word builder")
    st.write(f"**Current word:** {builder.text or '—'}")
    typed = st.selectbox("Letter to add", LETTERS, key="builder_letter")
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Add letter"):
        # Controls are explicit; the cooldown prevents accidental double clicks.
        if builder.add_letter(typed): st.rerun()
        else: st.info("Please wait briefly before adding another letter.")
    if c2.button("Space"): builder.space(); st.rerun()
    if c3.button("Delete"): builder.delete(); st.rerun()
    if c4.button("Clear"): builder.clear(); st.rerun()
    if c5.button("Speak"):
        ok, message = speak(builder.text)
        (st.success if ok else st.warning)(message)
    save_path = st.text_input("Save word to", value="recognized_text.txt")
    if st.button("Save word"):
        try: builder.save(save_path); st.success(f"Saved to {save_path}")
        except OSError as exc: st.error(f"Could not save: {exc}")
elif page == "Dataset Collection":
    st.header("Dataset Collection")
    st.subheader("Build a starter A–Z dataset")
    st.write("Download a public hand-image split, extract MediaPipe landmarks, then train the included classifier. Image files are kept in a local cache and are not uploaded by SignVision.")
    st.code("python -m src.data.import_huggingface_dataset\npython -m src.training.train", language="powershell")
    st.caption("The source dataset card does not state a clear image license. Review its provenance and terms before sharing extracted data or model results.")
    st.divider()
    st.subheader("Automatic batch collection")
    st.write("For hands-free interval sampling from a local webcam, select a sign and collect a full batch with one command. Only valid detected landmark vectors are saved; webcam frames remain in memory.")
    st.code("python -m src.data.collect_webcam --label A --target 500", language="powershell")
    st.write("Repeat for each letter from A to Z. Press `q` to stop early. Tune the camera and sample interval with `--camera` and `--interval`.")
    st.subheader("Import a labeled image dataset")
    st.write("If you already downloaded an ASL Alphabet image dataset arranged in A–Z folders, convert detected hands into the same landmark CSV:")
    st.code('python -m src.data.import_image_dataset --input "path\\to\\asl_alphabet_train"', language="powershell")
    st.caption("Images without a detected hand are skipped. The importer saves landmarks and labels only; raw images are not copied into this project.")
    st.divider()
    st.write("Capture one browser webcam frame at a time; a valid detected hand is converted to 63 normalized landmark values. The frame itself is discarded.")
    letter = st.selectbox("Target letter", LETTERS)
    pic = st.camera_input("Capture a hand sample")
    if pic:
        if st.button("Save this landmark sample", type="primary"):
            try:
                import cv2, mediapipe as mp, numpy as np
                image = cv2.imdecode(np.frombuffer(pic.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
                    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    st.error("No hand detected. Improve lighting and show the full hand.")
                else:
                    from src.preprocessing.landmark_processor import extract_hand_features
                    n = append_sample(SETTINGS.dataset_path, extract_hand_features(result.multi_hand_landmarks[0]), letter)
                    st.success(f"Saved sample {n} for {letter}. Original frame was not saved.")
            except ImportError:
                st.error("Camera collection needs OpenCV and MediaPipe. Install requirements.txt.")
            except Exception as exc:
                st.error(f"Could not save sample: {exc}")
    n = count_samples(SETTINGS.dataset_path, letter)
    st.metric(f"Samples for {letter}", n); st.progress(min(n / SETTINGS.target_samples, 1.0), text=f"Target: {SETTINGS.target_samples}")
    if Path(SETTINGS.dataset_path).exists():
        try:
            counts = pd.read_csv(SETTINGS.dataset_path).label.value_counts().sort_index()
            st.bar_chart(counts)
        except Exception as exc: st.warning(f"Dataset summary unavailable: {exc}")
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    report = SETTINGS.reports_dir / "model_comparison.csv"
    if report.exists():
        df = pd.read_csv(report); st.dataframe(df, use_container_width=True)
        cols = st.columns(4)
        for col, metric in zip(cols, ["accuracy", "precision", "recall", "f1_score"]):
            col.metric(metric.replace("_", " ").title(), f"{df.iloc[0][metric]:.3f}")
        for f in sorted(SETTINGS.reports_dir.glob("*.png")):
            st.image(str(f), caption=f.stem)
        detail = SETTINGS.reports_dir / "classification_report.txt"
        if detail.exists(): st.text(detail.read_text(encoding="utf-8"))
    else: st.info("No evaluation report yet. Collect a labeled dataset, then run `python -m src.training.train`.")
elif page == "About Project":
    st.header("About SignVision")
    st.markdown("""**Architecture:** MediaPipe Hands produces 21 3D landmarks. Wrist-relative translation and max-extent scaling normalize the pose. Four scikit-learn estimators are compared on an identical stratified holdout. Confidence filtering and a temporal vote reduce frame-to-frame flicker.

**Privacy:** recognition is local. Collection only persists numerical landmarks after an explicit save action; camera frames are not uploaded or stored.

**Dataset:** collect balanced A–Z samples from multiple users, sessions, hand orientations and backgrounds. The pipeline consumes `data/landmarks/landmarks.csv`; it does not bundle a dataset or pretrained weights.

**Limitations:** static landmarks do not fully represent moving letters J and Z, two-handed signs, facial grammar or continuous signing. Treat this as an alphabet fingerspelling prototype.""")
