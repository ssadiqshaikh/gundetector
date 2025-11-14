import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "guns_yolov8_best.pt"  # this file was copied in the previous cell

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

def run_detection(pil_img, conf_thres=0.5):
    """
    Runs YOLO on a PIL image, returns:
      - annotated image (PIL)
      - list of boxes [(x1, y1, x2, y2, score)]
    """
    results = model(pil_img, conf=conf_thres, verbose=False)
    res = results[0]

    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes.xyxy is not None else []
    scores = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []

    # res.plot() returns a BGR numpy array (OpenCV style)
    img_annotated_bgr = res.plot()
    img_annotated_rgb = img_annotated_bgr[..., ::-1]
    img_pil = Image.fromarray(img_annotated_rgb)

    boxes = []
    for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(s)))

    return img_pil, boxes

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="YOLO Gun Detector", layout="centered")
st.title("🔫 Gun Detection with YOLOv8")

st.write(
    "Upload an image and the model will try to detect guns "
    "and draw bounding boxes around them."
)

conf_thres = st.slider("Detection confidence threshold", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original image")
    st.image(pil_img, use_column_width=True)

    with st.spinner("Running YOLO detection..."):
        img_det, boxes = run_detection(pil_img, conf_thres=conf_thres)

    st.subheader(f"Detections (found {len(boxes)} boxes ≥ {conf_thres:.2f})")
    st.image(img_det, use_column_width=True)

    if boxes:
        st.write("Boxes (x1, y1, x2, y2, score):")
        for b in boxes:
            st.write(b)
    else:
        st.info("No boxes above the chosen threshold.")
