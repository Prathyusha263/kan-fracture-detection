"""
KAN-YOLOv8 Pediatric Wrist Fracture Detection
Streamlit web app — GRAZPEDWRI-DX project — TAMUCC Capstone
"""

import io
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="KAN-YOLOv8 Fracture Detection",
    page_icon="🩻",
    layout="wide",
)

# ── Color guide (RGB) — matches sidebar legend ───────────────────────────────
CLASS_COLORS = {
    "fracture": (46, 204, 113),            # green
    "periostealreaction": (236, 72, 153),  # pink
    "boneanomaly": (240, 210, 40),         # yellow
    "bonelesion": (34, 211, 238),          # cyan
    "pronatorsign": (249, 115, 22),        # orange
    "softtissue": (59, 130, 246),          # blue
    "metal": (156, 163, 175),              # gray
    "text": (156, 163, 175),               # gray
    "foreignbody": (244, 63, 94),          # red
}

COLOR_GUIDE = [
    ("Fracture", "#2ECC71"),
    ("Periosteal reaction", "#EC4899"),
    ("Bone Anomaly", "#F0D228"),
    ("Bone Lesion", "#22D3EE"),
    ("Pronator Sign", "#F97316"),
    ("Soft Tissue", "#3B82F6"),
    ("Metal or Text", "#9CA3AF"),
]

MODEL_METRICS = [
    ("mAP50", "0.649"),
    ("Precision", "0.724 ★"),
    ("Recall", "0.618"),
    ("Speed", "0.04s"),
    ("vs YOLOv8", "+2.4% Precision"),
]

RESEARCH_TEAM = [
    "Prathyusha Pentam",
    "Dimple Alekya Basimi",
    "Gowtham Kamle",
]

# ── Minimal CSS polish on top of the dark theme ─────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #161B22;
        border: 1px solid #2A313C;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #9CA3AF; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #E6EDF3; }
    .kan-banner {
        background: linear-gradient(90deg, #3A2A0E, #241804);
        border: 1px solid #F0A63A;
        border-radius: 8px;
        padding: 10px 16px;
        color: #F0C97A;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .legend-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.9rem; }
    .legend-swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading KAN-YOLOv8 model...")
def load_model():
    from ultralytics import YOLO
    return YOLO("best.pt")


def load_image_any_bitdepth(uploaded_file) -> Image.Image:
    raw_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(img).astype(np.float32)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def draw_boxes(image: Image.Image, result) -> Image.Image:
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
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        color = CLASS_COLORS.get(label, (240, 166, 58))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf:.0%}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1), text, fill=(10, 10, 10), font=font)

    return img


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩻 KAN-YOLOv8")
    st.caption("Fracture Detection AI")

    st.markdown("#### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

    st.markdown("#### 🧑‍🔬 Research Team")
    for name in RESEARCH_TEAM:
        st.write(f"• {name}")
    st.caption("Advisor: Dr. S. M. Mallikarjunaiah")
    st.caption("Dept. of Mathematics & Statistics, Texas A&M University–Corpus Christi")

    st.markdown("#### 📈 KAN-YOLOv8 Performance")
    for label, value in MODEL_METRICS:
        c1, c2 = st.columns([1.4, 1])
        c1.write(label)
        c2.write(f"**{value}**")

    with st.expander("❓ What is KAN?"):
        st.write(
            "**KAN = Kolmogorov-Arnold Network.** Replaces the standard "
            "MLP detection head with learnable spline activation functions.\n\n"
            "**Result:** Higher precision, fewer false alarms for doctors."
        )

    st.markdown("#### 🎨 Color Guide")
    for label, hex_color in COLOR_GUIDE:
        st.markdown(
            f'<div class="legend-row"><span class="legend-swatch" '
            f'style="background:{hex_color}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("## 🩻 KAN-YOLOv8 Pediatric Wrist Fracture Detection")
st.caption(
    "KAN-YOLOv8 — Kolmogorov-Arnold Network enhanced YOLOv8 | "
    "Trained on GRAZPEDWRI-DX | 20,327 X-ray images | 9 pathological classes | "
    "NVIDIA H100 GPU at TAMUCC CREST HPC"
)

st.markdown(
    '<div class="kan-banner">★ KAN Innovation: Replaced MLP head with '
    'Kolmogorov-Arnold Network layers → Precision improved from 0.700 to 0.724 '
    '→ Best Precision among ALL models!</div>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
    st.success("✅ KAN-YOLOv8 model loaded successfully and ready for detection!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ── Upload ───────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload X-ray Images")
st.write(
    f"Upload 1 to 10 pediatric wrist X-ray images — KAN-YOLOv8 detects all 9 "
    f"pathological classes with highest Precision of 0.724 in just 0.04 seconds!"
)

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload no more than 10 images at a time. Using the first 10.")
        uploaded_files = uploaded_files[:10]

    st.info(f"{len(uploaded_files)} image(s) ready — click Detect!")
    run_detection = st.button(
        "🔍 Detect with KAN-YOLOv8 — All Images", type="primary"
    )

    if run_detection:
        st.markdown("### 🎯 KAN-YOLOv8 Detection Results")

        per_image = []
        total_findings = 0
        images_with_fracture = 0
        total_time = 0.0

        with st.spinner("Running detection on all images..."):
            for uf in uploaded_files:
                uf.seek(0)
                try:
                    image = load_image_any_bitdepth(uf)
                except Exception as e:
                    per_image.append({"name": uf.name, "error": str(e)})
                    continue

                t0 = time.time()
                results = model.predict(image, conf=conf_threshold, verbose=False)
                elapsed = time.time() - t0
                total_time += elapsed

                result = results[0]
                names = result.names
                boxes = result.boxes
                detections = []
                has_fracture = False
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                        conf = float(box.conf[0])
                        detections.append((label, conf))
                        if label == "fracture":
                            has_fracture = True

                total_findings += len(detections)
                if has_fracture:
                    images_with_fracture += 1

                per_image.append({
                    "name": uf.name,
                    "image": image,
                    "annotated": draw_boxes(image, result),
                    "detections": detections,
                    "elapsed": elapsed,
                })

        # ── Batch summary ───────────────────────────────────────────────
        st.markdown("#### 📊 KAN-YOLOv8 Batch Summary")
        n_processed = len([p for p in per_image if "error" not in p])
        avg_time = total_time / n_processed if n_processed else 0

        cols = st.columns(4)
        stats = [
            ("Images Processed", str(n_processed)),
            ("Images with Fractures", str(images_with_fracture)),
            ("Total Findings", str(total_findings)),
            ("Avg Detection Time", f"{avg_time:.3f}s"),
        ]
        for col, (label, value) in zip(cols, stats):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        if images_with_fracture > 0:
            n_fracture_findings = sum(
                1 for p in per_image for (label, _) in p.get("detections", []) if label == "fracture"
            )
            st.error(
                f"🚨 FRACTURES DETECTED! {n_fracture_findings} fracture(s) across "
                f"{images_with_fracture} image(s). KAN-YOLOv8 Precision = 0.724 — "
                f"Please consult a radiologist immediately!"
            )
        else:
            st.success("No fractures detected in this batch.")

        # ── Individual results ─────────────────────────────────────────
        st.markdown("#### 🖼️ Individual Detection Results")
        for i, p in enumerate(per_image, start=1):
            if "error" in p:
                with st.expander(f"Image {i} — {p['name']} — ⚠️ ERROR"):
                    st.error(p["error"])
                continue

            status = "🟢 FINDINGS" if p["detections"] else "⚪ CLEAR"
            with st.expander(
                f"Image {i} — {p['name']} — {status} — {p['elapsed']:.2f}s",
                expanded=(i == 1),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original X-ray Image")
                    st.image(p["image"], use_container_width=True)
                with c2:
                    st.caption("KAN-YOLOv8 Detection Result")
                    st.image(p["annotated"], use_container_width=True)

                if p["detections"]:
                    st.table(
                        [{"Finding": lbl, "Confidence": f"{cf:.1%}"} for lbl, cf in p["detections"]]
                    )
                else:
                    st.write("No findings above the confidence threshold.")
else:
    st.info("Upload X-ray images above to get started.")

st.divider()
st.caption(
    "⚠️ Research prototype for a capstone project — not a certified "
    "diagnostic tool. Do not use for real clinical decisions."
)
