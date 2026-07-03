"""
Pediatric Wrist Fracture Detection — KAN-YOLOv8
Streamlit web app
GRAZPEDWRI-DX project — TAMUCC
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Wrist Fracture Detection",
    page_icon="🦴",
    layout="wide",
)

# ── Class names & box colors (RGB, matches GRAZPEDWRI-DX 9 classes) ──────────
CLASS_COLORS = {
    "boneanomaly": (255, 0, 0),
    "bonelesion": (255, 128, 0),
    "foreignbody": (255, 165, 0),
    "fracture": (255, 0, 255),
    "metal": (0, 255, 255),
    "periostealreaction": (0, 255, 0),
    "pronatorsign": (0, 128, 255),
    "softtissue": (180, 180, 180),
    "text": (128, 0, 255),
}


@st.cache_resource(show_spinner="Loading KAN-YOLOv8 model...")
def load_model():
    """Load the trained YOLO weights once and cache across reruns."""
    from ultralytics import YOLO
    model = YOLO("best.pt")
    return model


def draw_boxes(image: Image.Image, result) -> Image.Image:
    """Draw bounding boxes + labels on a PIL image from a YOLO result."""
    from PIL import ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()

    names = result.names
    boxes = result.boxes
    if boxes is None:
        return img

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        color = CLASS_COLORS.get(label, (255, 0, 0))

        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf:.0%}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1, y1), text, fill=(0, 0, 0), font=font)

    return img


def load_image_any_bitdepth(uploaded_file) -> Image.Image:
    """
    Load an uploaded image robustly, including 16-bit PNGs from medical
    scanners, and normalize to standard 8-bit RGB for display + inference.
    """
    raw_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(raw_bytes))

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(img).astype(np.float32)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val * 255.0
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = img.convert("RGB")

    return img


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🦴 Pediatric Wrist Fracture Detection")
st.caption("KAN-YOLOv8 · GRAZPEDWRI-DX · TAMUCC Capstone Project")

with st.sidebar:
    st.header("About")
    st.write(
        "Upload a pediatric wrist X-ray to detect fractures and 8 other "
        "pathological findings using a KAN-enhanced YOLOv8 model."
    )
    st.write("**Supported formats:** JPG, JPEG, PNG (including 16-bit medical PNG)")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

uploaded_file = st.file_uploader(
    "Upload an X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = load_image_any_bitdepth(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        try:
            model = load_model()
            results = model.predict(image, conf=conf_threshold, verbose=False)
            result = results[0]
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    annotated = draw_boxes(image, result)
    with col2:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    if result.boxes is not None and len(result.boxes) > 0:
        st.subheader("Detected findings")
        names = result.names
        rows = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            conf = float(box.conf[0])
            rows.append({"Finding": label, "Confidence": f"{conf:.1%}"})
        st.table(rows)
    else:
        st.info("No findings detected above the confidence threshold.")
else:
    st.info("Upload an X-ray image above to get started.")

st.divider()
st.caption(
    "⚠️ Research prototype for a capstone project — not a certified "
    "diagnostic tool. Do not use for real clinical decisions."
)
