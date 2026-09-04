import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_MODEL = BASE_DIR / "Classification" / "plant_disease_cnn_model.keras"
YOLO_MODEL = BASE_DIR / "object detection" / "disaster_response_best.pt"
BASE_YOLO_MODEL = BASE_DIR / "object detection" / "yolov8n.pt"
CLASS_NAMES = [
    "Corn Common Rust",
    "Potato Early Blight",
    "Tomato Bacterial Spot",
]
DISEASE_SUGGESTIONS = {
    "Corn Common Rust": [
        "Remove severely affected leaves and dispose of them away from the crop.",
        "Improve airflow by maintaining suitable spacing between plants.",
        "Ask a local agronomist about an approved fungicide if symptoms spread.",
    ],
    "Potato Early Blight": [
        "Remove infected foliage and clear fallen plant debris from the soil.",
        "Water at the base of the plant and avoid wetting leaves overnight.",
        "Use crop rotation and ask a local agronomist about approved treatment.",
    ],
    "Tomato Bacterial Spot": [
        "Remove affected leaves and avoid handling healthy plants afterward.",
        "Avoid overhead watering and keep foliage dry with good plant spacing.",
        "Use disease-free seed or transplants and consult a local agronomist.",
    ],
}
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="CNN Vision Suite", page_icon="🖼️", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: .2rem; }
    .sub-header { color: #6b7280; margin-bottom: 1.5rem; }
    .card-box { border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 20px; margin-bottom: 15px; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .stAppDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stElementToolbar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier():
    import tensorflow as tf
    return tf.keras.models.load_model(CLASSIFIER_MODEL)


@st.cache_resource
def load_detector(model_path):
    from ultralytics import YOLO
    return YOLO(str(model_path))


def classify_image(image):
    image_array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    probabilities = load_classifier().predict(image_array[None, ...], verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return [(CLASS_NAMES[index], float(probabilities[index])) for index in order]


def detector_path():
    if YOLO_MODEL.exists():
        return YOLO_MODEL, "Trained disaster-response checkpoint"
    if BASE_YOLO_MODEL.exists():
        return BASE_YOLO_MODEL, "Base YOLOv8 checkpoint (train and export best.pt for custom classes)"
    return None, "No YOLO checkpoint found"


st.sidebar.title("🖼️ CNN Control Suite")
st.sidebar.caption("Image Classification & Object Detection")
section = st.sidebar.radio("Select Model Module:", ["🏠 Dashboard Overview", "🌿 Plant Disease Classification", "🚨 Disaster Response Detection"])

if section == "🏠 Dashboard Overview":
    st.markdown('<div class="main-header">Convolutional Neural Network (CNN) Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image and test the trained vision models</div>', unsafe_allow_html=True)
    detector_file, detector_detail = detector_path()
    classifier_status = "Ready" if CLASSIFIER_MODEL.exists() else "Missing model"
    detector_status = "Ready" if detector_file else "Missing model"

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("CNN modules", "2")
    metric_2.metric("Disease classes", len(CLASS_NAMES))
    metric_3.metric("Classifier input", "224 × 224")
    metric_4.metric("Detector status", detector_status)

    st.subheader("Model Results")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="card-box"><h3>🌿 Plant Disease Classification</h3>'
            f'<p><strong>{classifier_status}</strong> · Keras CNN</p>'
            f'<p>Classes: {", ".join(CLASS_NAMES)}</p>'
            f'<p>Input: RGB image resized to 224 × 224 pixels</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="card-box"><h3>🚨 Disaster Response Detection</h3>'
            f'<p><strong>{detector_status}</strong> · YOLOv8</p>'
            f'<p>{detector_detail}</p>'
            f'<p>Output: detected objects, bounding boxes, and confidence scores</p></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Information")
    info_left, info_right = st.columns(2)
    with info_left:
        st.info("Use Plant Disease Classification for leaf images. The result includes the top class and probabilities for every disease category.")
    with info_right:
        st.info("Use Disaster Response Detection for scene images. A trained disaster checkpoint is required for custom disaster classes.")

elif section == "🌿 Plant Disease Classification":
    st.markdown('<div class="main-header">🌿 Plant Disease Classification</div>', unsafe_allow_html=True)
    st.caption("CNN input: RGB image resized to 224 × 224 pixels")
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        preview, result = st.columns([1, 1])
        with preview:
            st.image(image, caption=uploaded.name, use_container_width=True)
        with result:
            with st.spinner("Running CNN inference..."):
                predictions = classify_image(image)
            label, confidence = predictions[0]
            st.success("Inference complete")
            st.metric("Predicted class", label)
            st.metric("Confidence", f"{confidence:.1%}")
            st.subheader("Suggested next steps")
            for suggestion in DISEASE_SUGGESTIONS[label]:
                st.markdown(f"- {suggestion}")
            st.caption("These suggestions are general guidance. Confirm the diagnosis with a plant-health professional before treatment.")
    else:
        st.info("Upload a JPG, PNG, or WEBP image to test the classifier.")

else:
    st.markdown('<div class="main-header">🚨 Disaster Response Detection</div>', unsafe_allow_html=True)
    detector_file, detail = detector_path()
    st.caption(detail)
    uploaded = st.file_uploader("Upload a disaster-response image", type=["jpg", "jpeg", "png", "webp"])
    if not detector_file:
        st.warning("The YOLO checkpoint is missing. Run the notebook training cells and export the trained model to object detection/disaster_response_best.pt.")
    elif uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running YOLO inference..."):
            result = load_detector(str(detector_file))(image, verbose=False)[0]
        st.image(result.plot(), caption=uploaded.name, use_container_width=True)
        if result.boxes is not None and len(result.boxes):
            rows = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                rows.append({"Class": result.names[class_id], "Confidence": f"{float(box.conf[0]):.1%}"})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected above the model confidence threshold.")
    else:
        st.info("Upload an image to test the detector.")