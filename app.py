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
    page_icon="🦴",
    layout="wide",
)

# ── Class metadata: color, description ──────────────────────────────────────
CLASS_INFO = {
    "fracture": {
        "color": (46, 204, 113),
        "hex": "#2ECC71",
        "desc": "Broken bone — immediate clinical attention required",
    },
    "periostealreaction": {
        "color": (236, 72, 153),
        "hex": "#EC4899",
        "desc": "Bone healing response — may indicate hidden fracture",
    },
    "boneanomaly": {
        "color": (167, 139, 250),
        "hex": "#A78BFA",
        "desc": "Abnormal bone shape or structure detected",
    },
    "bonelesion": {
        "color": (240, 210, 40),
        "hex": "#F0D228",
        "desc": "Damaged bone area — requires further investigation",
    },
    "foreignbody": {
        "color": (249, 115, 22),
        "hex": "#F97316",
        "desc": "Foreign object — must locate before any treatment",
    },
    "metal": {
        "color": (229, 231, 235),
        "hex": "#E5E7EB",
        "desc": "Metal implant from previous surgery detected",
    },
    "pronatorsign": {
        "color": (251, 146, 60),
        "hex": "#FB923C",
        "desc": "Indirect sign of hidden fracture — fat pad displaced",
    },
    "softtissue": {
        "color": (59, 130, 246),
        "hex": "#3B82F6",
        "desc": "Soft tissue swelling or injury around bone",
    },
    "text": {
        "color": (156, 163, 175),
        "hex": "#9CA3AF",
        "desc": "Text label or marker on X-ray image",
    },
}

CLASS_ICONS = {
    "fracture": "🦴",
    "periostealreaction": "⚡",
    "boneanomaly": "🔘",
    "bonelesion": "⚠️",
    "foreignbody": "🧲",
    "metal": "🔩",
    "pronatorsign": "📍",
    "softtissue": "✏️",
    "text": "📝",
}

COLOR_GUIDE = [
    ("Green = Fracture", CLASS_INFO["fracture"]["hex"]),
    ("Magenta = Periosteal Reaction", CLASS_INFO["periostealreaction"]["hex"]),
    ("Yellow = Bone Anomaly", CLASS_INFO["boneanomaly"]["hex"]),
    ("Cyan = Bone Lesion", CLASS_INFO["bonelesion"]["hex"]),
    ("Orange = Pronator Sign", CLASS_INFO["pronatorsign"]["hex"]),
    ("Blue = Soft Tissue", CLASS_INFO["softtissue"]["hex"]),
    ("White = Metal / Text", CLASS_INFO["metal"]["hex"]),
]

MODEL_METRICS = [
    ("mAP50", "0.649", True),
    ("Precision", "0.724 ★", False),
    ("Recall", "0.618", False),
    ("Speed", "0.04s", False),
    ("Beats SOTA", "✅ Yes", False),
]

RESEARCH_TEAM = ["Prathyusha Pentam", "Gowtham Kamle", "Dimple Alekya Basimi"]

# ── CSS polish ────────────────────────────────────────────────────────────
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
    .legend-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.85rem; }
    .legend-swatch { width: 13px; height: 13px; border-radius: 3px; flex-shrink: 0; }
    .empty-state {
        background: #10141C;
        border: 1px dashed #2A313C;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
    }
    .empty-state h3 { margin-top: 8px; }
    .step-card {
        background: #161B22;
        border: 1px solid #2A313C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }
    .class-card {
        background: #161B22;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .class-card .name { font-weight: 700; color: var(--accent); }
    .class-card .desc { color: #9CA3AF; font-size: 0.85rem; margin-top: 2px; }
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
        color = CLASS_INFO.get(label, {}).get("color", (240, 166, 58))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf:.0%}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1), text, fill=(10, 10, 10), font=font)

    return img


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦴 Fracture Detection AI")

    st.markdown("#### ⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

    st.markdown("#### 👥 Research Team")
    for name in RESEARCH_TEAM:
        st.write(name)
    st.caption(f"*Advisor: Dr. S. M. Mallikarjunaiah*")
    st.caption("Dept. of Mathematics & Statistics, Texas A&M University–Corpus Christi")

    st.markdown("#### 📊 Model: KAN-YOLOv8")
    rows = "".join(
        f"<tr><td>{label}</td><td><b>{value}</b></td></tr>" for label, value, _ in MODEL_METRICS
    )
    st.markdown(
        f'<table style="width:100%;font-size:0.9rem;">'
        f'<tr><th align="left">Metric</th><th align="left">Score</th></tr>{rows}</table>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎨 Color Guide")
    for label, hex_color in COLOR_GUIDE:
        st.markdown(
            f'<div class="legend-row"><span class="legend-swatch" '
            f'style="background:{hex_color}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "⚠️ For research use only. All detections must be confirmed by a "
        "qualified radiologist."
    )

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("## 🦴 Pediatric Wrist Fracture Detection")
st.caption(
    "KAN-YOLOv8 AI trained on GRAZPEDWRI-DX — 20,327 X-ray images | "
    "9 pathological classes | NVIDIA H100 GPU at TAMUCC CREST HPC | "
    "mAP50 = 0.649 — Surpassing published state-of-the-art!"
)

st.markdown(
    '<div class="kan-banner">★ KAN Innovation: Replaced MLP head with '
    'Kolmogorov-Arnold Network layers → Precision improved from 0.700 to 0.724 '
    '→ Best Precision among ALL models!</div>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
    st.success("✅ KAN-YOLOv8 model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.divider()

# ── Upload ───────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload X-ray Images")
st.write("Upload 1 to 10 pediatric wrist X-ray images at once — AI detects all findings simultaneously!")

uploaded_files = st.file_uploader(
    "Drag and drop or browse X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:2.5rem;">🦴</div>
            <h3>Upload X-ray Images to Start Detection</h3>
            <p style="color:#9CA3AF;">
                Upload 1 to 10 pediatric wrist X-ray images above.<br>
                The AI will detect fractures and all 9 pathological classes
                in just 0.04 seconds per image!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stat_cols = st.columns(4)
    empty_stats = [
        ("9", "Classes Detected", "#22D3EE"),
        ("0.04s", "Per Image", "#2ECC71"),
        ("0.649", "mAP50 Score", "#F0A63A"),
        ("10", "Max Images", "#A78BFA"),
    ]
    for col, (value, label, color) in zip(stat_cols, empty_stats):
        col.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.6rem;font-weight:800;color:{color};">{value}</div>'
            f'<div style="color:#9CA3AF;font-size:0.85rem;">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 How to Use")
    step_cols = st.columns(3)
    steps = [
        ("📤", "Step 1", "Upload 1 to 10 pediatric wrist X-ray images in JPG or PNG format including 16-bit PNG from medical scanners"),
        ("🔍", "Step 2", "Click the Detect button and wait 0.04 seconds per image for AI to analyze all 9 pathological classes simultaneously"),
        ("📊", "Step 3", "View results with colored bounding boxes, confidence scores, clinical alerts and detailed findings for each image"),
    ]
    for col, (icon, title, desc) in zip(step_cols, steps):
        col.markdown(
            f'<div class="step-card"><div style="font-size:1.8rem;">{icon}</div>'
            f'<b>{title}</b><p style="color:#9CA3AF;font-size:0.85rem;margin-top:6px;">{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🔬 9 Pathological Classes Detected")
    class_cols = st.columns(3)
    for i, (label, info) in enumerate(CLASS_INFO.items()):
        col = class_cols[i % 3]
        col.markdown(
            f'<div class="class-card" style="--accent:{info["hex"]};">'
            f'<div class="name">{label.upper()}</div>'
            f'<div class="desc">{info["desc"]}</div></div>',
            unsafe_allow_html=True,
        )

else:
    if len(uploaded_files) > 10:
        st.warning("Please upload no more than 10 images at a time. Using the first 10.")
        uploaded_files = uploaded_files[:10]

    st.info(f"{len(uploaded_files)} image(s) ready — click Detect!")
    run_detection = st.button("🔍 Detect with KAN-YOLOv8 — All Images", type="primary")

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
                    st.markdown("**📋 Detailed Findings:**")
                    for lbl, cf in p["detections"]:
                        icon = CLASS_ICONS.get(lbl, "🔎")
                        desc = CLASS_INFO.get(lbl, {}).get("desc", "")
                        accent = CLASS_INFO.get(lbl, {}).get("hex", "#F0A63A")
                        st.markdown(
                            f'<div class="class-card" style="--accent:{accent};">'
                            f'<div class="name">{icon} {lbl.upper()} — {cf:.0%}</div>'
                            f'<div class="desc">{desc}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No findings above the confidence threshold.")

st.divider()
st.caption(
    "⚠️ Research prototype for a capstone project — not a certified "
    "diagnostic tool. Do not use for real clinical decisions."
)
