import streamlit as st
import numpy as np
from PIL import Image
import time
import tempfile
import os
from pathlib import Path

try:
    import cv2
except ImportError:
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "pip",
        "install", "opencv-python-headless"
    ], check=True)
    import cv2

from ultralytics import YOLO

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="KAN-YOLOv8 Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
* { color: #FFFFFF !important; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="element-container"],
.main, .block-container {
    background-color: #0F172A !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebarContent"] * { color: #FFFFFF !important; }
.stButton > button {
    background-color: #F97316 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100% !important;
    padding: 16px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background-color: #EA580C !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploadDropzone"] {
    background-color: #1E293B !important;
    border: 2px dashed #F97316 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stFileUploadDropzone"] * { color: #FFFFFF !important; }
[data-testid="stExpander"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] * { color: #FFFFFF !important; }
details summary { color: #FFFFFF !important; }
[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border-radius: 12px !important;
    padding: 16px !important;
    border: 1px solid #334155 !important;
}
[data-testid="stMetric"] * { color: #FFFFFF !important; }
[data-testid="stMetricLabel"] * { color: #94A3B8 !important; }
[data-testid="stMetricValue"] * { color: #F8FAFC !important; font-size: 28px !important; }
[data-testid="stAlert"] { border-radius: 12px !important; }
[data-testid="stAlert"] * { color: #FFFFFF !important; }
.stProgress > div > div { background-color: #F97316 !important; }
[data-testid="stSlider"] * { color: #FFFFFF !important; }
[data-testid="stSlider"] > div > div > div {
    background-color: #F97316 !important;
}
label { color: #FFFFFF !important; }
p,h1,h2,h3,h4,h5,h6 { color: #FFFFFF !important; }
table * { color: #FFFFFF !important; }
.stImage { border-radius: 10px !important; overflow: hidden !important; }
hr { border-color: #334155 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
CLASS_NAMES = [
    'boneanomaly','bonelesion','foreignbody','fracture',
    'metal','periostealreaction','pronatorsign','softtissue','text'
]
CLASS_COLORS_BGR = {
    'fracture':           (0,200,0),
    'periostealreaction': (200,0,200),
    'boneanomaly':        (200,200,0),
    'bonelesion':         (0,200,200),
    'foreignbody':        (0,140,255),
    'metal':              (160,160,160),
    'pronatorsign':       (220,100,0),
    'softtissue':         (0,100,220),
    'text':               (140,140,140),
}
CLASS_EMOJI = {
    'fracture':'🦴','periostealreaction':'⚡',
    'boneanomaly':'🔍','bonelesion':'⚠️',
    'foreignbody':'🔩','metal':'🔧',
    'pronatorsign':'📍','softtissue':'🩹','text':'📝',
}
CLASS_COLOR_HEX = {
    'fracture':'#4ADE80','periostealreaction':'#F472B6',
    'boneanomaly':'#FBBF24','bonelesion':'#22D3EE',
    'foreignbody':'#FB923C','metal':'#CBD5E1',
    'pronatorsign':'#F97316','softtissue':'#60A5FA',
    'text':'#94A3B8',
}
CLASS_DESC = {
    'fracture':           'Broken bone — immediate clinical attention required',
    'periostealreaction': 'Bone healing response — may indicate hidden fracture nearby',
    'boneanomaly':        'Abnormal bone shape or structure detected',
    'bonelesion':         'Damaged bone area — requires further investigation',
    'foreignbody':        'Foreign object — must locate before any treatment',
    'metal':              'Metal implant from previous surgery detected',
    'pronatorsign':       'Indirect sign of hidden fracture — fat pad displaced',
    'softtissue':         'Soft tissue swelling or injury around bone',
    'text':               'Text label or marker on X-ray image',
}

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading KAN-YOLOv8 model...")
def load_model():
    for p in [
        Path(__file__).parent / "best.pt",
        Path("best.pt"),
    ]:
        if p.exists():
            return YOLO(str(p))
    return None

# ── Image Processing ─────────────────────────────────────────────
def load_image(uf):
    b = np.frombuffer(uf.read(), np.uint8)
    img = cv2.imdecode(b, cv2.IMREAD_ANYDEPTH)
    uf.seek(0)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 65535.0 * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect(model, img, conf):
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    tmp.close()
    t0 = time.time()
    res = model.predict(tmp.name, imgsz=640, conf=conf, verbose=False)
    el = time.time() - t0
    os.unlink(tmp.name)
    return res[0], el

def draw_boxes(img, res):
    out = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    dets = []
    if res.boxes is not None:
        for box in res.boxes:
            cid = int(box.cls[0])
            cf  = float(box.conf[0])
            nm  = CLASS_NAMES[cid]
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            c  = CLASS_COLORS_BGR.get(nm, (150,150,150))
            th = 4 if nm == 'fracture' else 2
            cv2.rectangle(out,(x1,y1),(x2,y2),c,th)
            lbl = f"{nm} {cf:.0%}"
            sz,_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.60,2)
            cv2.rectangle(out,(x1,max(y1-26,0)),(x1+sz[0]+10,y1),c,-1)
            cv2.putText(out,lbl,(x1+5,max(y1-6,12)),
                       cv2.FONT_HERSHEY_SIMPLEX,0.60,(0,0,0),2)
            dets.append({'class':nm,'conf':cf})
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), dets

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 5px;">
    <span style="font-size:32px;">🦴</span>
    <div style="font-size:18px;font-weight:700;color:#F97316;
    margin-top:4px;">KAN-YOLOv8</div>
    <div style="font-size:13px;color:#94A3B8;">
    Fracture Detection AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### ⚙️ Settings")
    conf = st.slider(
        "Confidence Threshold",
        0.10, 0.90, 0.20, 0.05,
        help="Lower = more detections | Higher = more certain"
    )
    st.caption(f"Current: {conf:.0%} confidence minimum")

    st.markdown("---")

    st.markdown("#### 👥 Research Team")
    st.markdown("""
<div style="font-size:13px;line-height:1.8;">
<b>Prathyusha Pentam</b><br>
<b>Dimple Alekhya Basimi</b><br>
<b>Gowtham Kamle</b><br>
<br>
<span style="color:#94A3B8;font-size:12px;">Contributors:</span><br>
Chandana Pati<br>
B. Veena S. N. Rao<br>
Ravi Samraj<br>
<br>
<i style="color:#94A3B8;">Advisor:</i><br>
<b>Dr. S. M. Mallikarjunaiah</b><br>
<br>
<span style="color:#64748B;font-size:11px;">
Dept. of Mathematics & Statistics<br>
Texas A&M University – Corpus Christi
</span>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 📊 KAN-YOLOv8 Performance")
    st.markdown("""
<div style="background:#0F172A;border-radius:10px;
padding:12px;font-size:12px;">
<table style="width:100%;border-collapse:collapse;">
<tr><td style="color:#94A3B8;padding:3px 0;">mAP50</td>
<td style="text-align:right;color:#F8FAFC;font-weight:600;">0.649</td></tr>
<tr><td style="color:#94A3B8;padding:3px 0;">Precision</td>
<td style="text-align:right;color:#F97316;font-weight:700;">0.724 ★</td></tr>
<tr><td style="color:#94A3B8;padding:3px 0;">Recall</td>
<td style="text-align:right;color:#F8FAFC;font-weight:600;">0.618</td></tr>
<tr><td style="color:#94A3B8;padding:3px 0;">Speed</td>
<td style="text-align:right;color:#4ADE80;font-weight:600;">0.04s</td></tr>
<tr><td style="color:#94A3B8;padding:3px 0;">vs YOLOv8</td>
<td style="text-align:right;color:#4ADE80;font-weight:600;">+2.4% Precision</td></tr>
</table>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 💡 What is KAN?")
    st.markdown("""
<div style="font-size:12px;color:#CBD5E1;line-height:1.7;
background:#0F172A;border-radius:10px;padding:12px;">
<b style="color:#F97316;">KAN</b> = Kolmogorov-Arnold Network<br><br>
Replaces standard MLP detection head
with <b>learnable B-spline</b> activation
functions on each network edge.<br><br>
✅ Higher Precision<br>
✅ Fewer false alarms<br>
✅ Better generalization
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 🎨 Color Guide")
    st.markdown("""
<div style="font-size:12px;line-height:2.0;">
🟢 <span style="color:#4ADE80;">Green</span> = Fracture<br>
🟣 <span style="color:#F472B6;">Magenta</span> = Periosteal Reaction<br>
🟡 <span style="color:#FBBF24;">Yellow</span> = Bone Anomaly<br>
🩵 <span style="color:#22D3EE;">Cyan</span> = Bone Lesion<br>
🟠 <span style="color:#F97316;">Orange</span> = Pronator Sign<br>
🔵 <span style="color:#60A5FA;">Blue</span> = Soft Tissue<br>
⬜ <span style="color:#CBD5E1;">Gray</span> = Metal / Text
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ For research use only. Not a diagnostic tool.")

# ════════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:10px 0 4px;">
<h1 style="font-size:28px;font-weight:800;margin:0;
color:#F8FAFC;letter-spacing:-0.5px;">
🦴 KAN-YOLOv8 Pediatric Wrist Fracture Detection
</h1>
<p style="color:#94A3B8;margin:6px 0 0;font-size:14px;">
Kolmogorov-Arnold Network enhanced YOLOv8 &nbsp;·&nbsp;
Trained on GRAZPEDWRI-DX — 20,327 X-ray images &nbsp;·&nbsp;
9 pathological classes &nbsp;·&nbsp;
NVIDIA H100 GPU at TAMUCC CREST HPC
</p>
</div>
""", unsafe_allow_html=True)

# KAN banner
st.markdown("""
<div style="background:linear-gradient(135deg,#1C1A12,#2D1F00);
border:2px solid #F97316;border-radius:12px;
padding:14px 20px;margin:14px 0 8px;">
<span style="color:#F97316;font-weight:700;font-size:15px;">
⭐ KAN Innovation:
</span>
<span style="color:#FED7AA;font-size:14px;">
Replaced MLP detection head with Kolmogorov-Arnold Network layers →
Precision improved from 0.700 to 0.724 — Best Precision among ALL models!
</span>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
if model is None:
    st.markdown("""
    <div style="background:#1A0505;border:2px solid #EF4444;
    border-radius:12px;padding:14px 20px;margin:8px 0;">
    <span style="color:#EF4444;font-weight:700;">
    ❌ Model file best.pt not found!
    Please ensure best.pt is in the repository.
    </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="background:#051A0A;border:2px solid #22C55E;
border-radius:12px;padding:12px 20px;margin:8px 0;">
<span style="color:#4ADE80;font-weight:700;">
✅ KAN-YOLOv8 model loaded successfully and ready for detection!
</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload Section ───────────────────────────────────────────────
st.markdown("### 📤 Upload X-ray Images")
st.markdown(
    "Upload **1 to 10** pediatric wrist X-ray images (JPG, JPEG, PNG including 16-bit clinical PNG). "
    "KAN-YOLOv8 detects all **9 pathological classes** in just **0.04 seconds** per image."
)

uploaded_files = st.file_uploader(
    "Drag and drop X-ray images here or click to browse",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="Supports standard and 16-bit PNG from clinical X-ray scanners"
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Maximum 10 images allowed. Using first 10.")
        uploaded_files = uploaded_files[:10]

    st.markdown(f"""
    <div style="background:#1E293B;border:1px solid #334155;
    border-radius:10px;padding:12px 16px;margin:10px 0;">
    <span style="color:#4ADE80;font-weight:700;font-size:15px;">
    📁 {len(uploaded_files)} image(s) ready — click Detect!
    </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        f"🔍  Detect with KAN-YOLOv8 — All {len(uploaded_files)} Image(s)",
        type="primary"
    ):
        st.markdown("---")
        st.markdown("### 🤖 KAN-YOLOv8 Detection Results")

        prog = st.progress(0, text="Initializing KAN-YOLOv8...")
        results = []

        for idx, uf in enumerate(uploaded_files):
            prog.progress(
                (idx+1)/len(uploaded_files),
                text=f"🔍 Analyzing image {idx+1} of {len(uploaded_files)}: {uf.name}"
            )
            img = load_image(uf)
            if img is None:
                st.error(f"Could not load {uf.name}")
                continue
            res, el = detect(model, img, conf)
            img_out, dets = draw_boxes(img, res)
            fracs = [d for d in dets if d['class']=='fracture']
            results.append({
                'name':     uf.name,
                'original': img,
                'result':   img_out,
                'dets':     dets,
                'fractures':fracs,
                'elapsed':  el,
            })

        prog.empty()

        if not results:
            st.error("No images could be processed.")
            st.stop()

        # ── Batch Summary ────────────────────────────────────────
        tot_frac  = sum(len(r['fractures']) for r in results)
        imgs_frac = sum(1 for r in results if r['fractures'])
        tot_finds = sum(len(r['dets']) for r in results)
        avg_t     = sum(r['elapsed'] for r in results) / len(results)

        st.markdown("#### 📊 Batch Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Images Processed",    len(results))
        c2.metric("Images with Fractures", imgs_frac,
                  delta=f"{imgs_frac} flagged" if imgs_frac > 0 else None)
        c3.metric("Total Findings",      tot_finds)
        c4.metric("Avg Detection Time",  f"{avg_t:.3f}s")

        # ── Alert Banner ─────────────────────────────────────────
        if tot_frac > 0:
            st.markdown(f"""
            <div style="background:#1A0505;border:2px solid #EF4444;
            border-radius:12px;padding:14px 20px;margin:12px 0;">
            <span style="color:#EF4444;font-weight:700;font-size:15px;">
            🚨 FRACTURES DETECTED!
            </span>
            <span style="color:#FCA5A5;font-size:14px;">
            &nbsp; {tot_frac} fracture(s) across {imgs_frac} image(s).
            KAN-YOLOv8 Precision = 0.724 —
            Please consult a qualified radiologist immediately!
            </span>
            </div>
            """, unsafe_allow_html=True)
        elif tot_finds > 0:
            st.markdown(f"""
            <div style="background:#1A1200;border:2px solid #F59E0B;
            border-radius:12px;padding:14px 20px;margin:12px 0;">
            <span style="color:#F59E0B;font-weight:700;">
            ⚠️ {tot_finds} finding(s) detected — no fractures above threshold.
            Clinical review recommended.
            </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#051A0A;border:2px solid #22C55E;
            border-radius:12px;padding:14px 20px;margin:12px 0;">
            <span style="color:#4ADE80;font-weight:700;">
            ✅ No findings detected at {conf:.0%} confidence threshold.
            Try lowering the threshold if needed.
            </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Individual Results ───────────────────────────────────
        st.markdown("#### 🖼️ Individual Detection Results")

        for i, r in enumerate(results):
            has_frac = len(r['fractures']) > 0
            has_find = len(r['dets']) > 0

            if has_frac:
                badge = f"🚨 {len(r['fractures'])} FRACTURE(S) DETECTED"
                border = "#EF4444"
            elif has_find:
                badge = f"⚠️ {len(r['dets'])} FINDING(S)"
                border = "#F59E0B"
            else:
                badge = "✅ CLEAR"
                border = "#22C55E"

            with st.expander(
                f"Image {i+1} — {r['name']} — {badge} — {r['elapsed']:.3f}s",
                expanded=True
            ):
                # Side by side images
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background:#0F172A;border-radius:8px;
                    padding:8px 12px;margin-bottom:8px;
                    border:1px solid #334155;">
                    <span style="color:#94A3B8;font-size:12px;
                    font-weight:600;">📷 ORIGINAL X-RAY</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(r['original'], use_container_width=True)

                with col2:
                    st.markdown("""
                    <div style="background:#0F172A;border-radius:8px;
                    padding:8px 12px;margin-bottom:8px;
                    border:1px solid #F97316;">
                    <span style="color:#F97316;font-size:12px;
                    font-weight:600;">🤖 KAN-YOLOv8 DETECTION</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(r['result'], use_container_width=True)

                # Result alert
                if has_frac:
                    st.markdown(f"""
                    <div style="background:#1A0505;border-left:4px solid #EF4444;
                    border-radius:6px;padding:10px 14px;margin:10px 0;">
                    <span style="color:#EF4444;font-weight:700;">
                    🚨 FRACTURE DETECTED — {len(r['fractures'])} fracture(s) found!
                    </span>
                    <span style="color:#FCA5A5;font-size:12px;">
                    &nbsp; Processed in {r['elapsed']:.3f}s —
                    Please consult a radiologist!
                    </span>
                    </div>
                    """, unsafe_allow_html=True)
                elif has_find:
                    st.markdown(f"""
                    <div style="background:#1A1200;border-left:4px solid #F59E0B;
                    border-radius:6px;padding:10px 14px;margin:10px 0;">
                    <span style="color:#F59E0B;font-weight:700;">
                    ⚠️ {len(r['dets'])} finding(s) — no fractures detected
                    </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#051A0A;border-left:4px solid #22C55E;
                    border-radius:6px;padding:10px 14px;margin:10px 0;">
                    <span style="color:#4ADE80;font-weight:700;">
                    ✅ No findings at this confidence level
                    </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Detailed findings
                if r['dets']:
                    st.markdown("**📋 Detailed Findings:**")
                    cols = st.columns(min(len(r['dets']), 3))
                    for j, det in enumerate(r['dets']):
                        cls  = det['class']
                        cf   = det['conf']
                        em   = CLASS_EMOJI.get(cls, '🔍')
                        desc = CLASS_DESC.get(cls, '')
                        col  = CLASS_COLOR_HEX.get(cls, '#FFFFFF')
                        with cols[j % 3]:
                            st.markdown(f"""
                            <div style="background:#0F172A;border:1px solid #334155;
                            border-left:3px solid {col};border-radius:8px;
                            padding:10px 12px;margin:4px 0;">
                            <div style="color:{col};font-weight:700;font-size:13px;">
                            {em} {cls.upper()}
                            </div>
                            <div style="color:{col};font-size:20px;
                            font-weight:800;margin:4px 0;">
                            {cf:.0%}
                            </div>
                            <div style="color:#64748B;font-size:11px;
                            line-height:1.4;">
                            {desc}
                            </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Speed
                spd = int(600/r['elapsed']) if r['elapsed'] > 0 else 15000
                st.markdown(f"""
                <div style="color:#475569;font-size:12px;
                margin-top:10px;text-align:right;">
                ⚡ KAN-YOLOv8: <b style="color:#94A3B8;">{r['elapsed']:.3f}s</b>
                &nbsp;|&nbsp; Manual review: <b style="color:#94A3B8;">10–15 min</b>
                &nbsp;|&nbsp;
                <b style="color:#F97316;">{spd:,}× faster</b>
                </div>
                """, unsafe_allow_html=True)

        # ── Speed Summary ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### ⚡ Speed Comparison")
        s1,s2,s3 = st.columns(3)
        s1.metric("KAN-YOLOv8 Avg", f"{avg_t:.3f}s")
        s2.metric("Manual Radiologist", "10–15 min")
        spd_total = int(600/avg_t) if avg_t > 0 else 15000
        s3.metric("Speed Improvement", f"{spd_total:,}×")

        # ── Disclaimer ───────────────────────────────────────────
        st.markdown("""
        <div style="background:#0F172A;border:1px solid #334155;
        border-radius:10px;padding:14px 18px;margin-top:20px;">
        <span style="color:#64748B;font-size:12px;">
        ⚠️ <b style="color:#94A3B8;">DISCLAIMER:</b>
        This AI tool is for research and educational purposes only.
        All detections must be reviewed and confirmed by a qualified
        radiologist before any clinical decisions are made.
        Not approved for diagnostic use.
        </span>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# WELCOME SCREEN (no files uploaded)
# ════════════════════════════════════════════════════════════════
else:
    st.markdown("---")

    # Model comparison cards
    st.markdown("### 📊 Model Comparison")
    ca, cb, cc, cd = st.columns(4)

    with ca:
        st.markdown("""
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:16px;text-align:center;">
        <div style="color:#60A5FA;font-weight:700;font-size:13px;
        margin-bottom:10px;">YOLOv8</div>
        <div style="color:#94A3B8;font-size:12px;line-height:2.0;">
        mAP50: 0.649<br>Precision: 0.700<br>
        Recall: 0.635<br>Speed: 0.04s
        </div>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:16px;text-align:center;">
        <div style="color:#A78BFA;font-weight:700;font-size:13px;
        margin-bottom:10px;">YOLOv12</div>
        <div style="color:#94A3B8;font-size:12px;line-height:2.0;">
        mAP50: 0.623<br>Precision: 0.639<br>
        Recall: 0.605<br>Speed: 0.04s
        </div>
        </div>
        """, unsafe_allow_html=True)

    with cc:
        st.markdown("""
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:16px;text-align:center;">
        <div style="color:#FB923C;font-weight:700;font-size:13px;
        margin-bottom:10px;">RT-DETR</div>
        <div style="color:#94A3B8;font-size:12px;line-height:2.0;">
        mAP50: 0.641<br>Precision: 0.673<br>
        Recall: 0.653<br>Speed: 0.04s
        </div>
        </div>
        """, unsafe_allow_html=True)

    with cd:
        st.markdown("""
        <div style="background:#1C1A12;border:2px solid #F97316;
        border-radius:12px;padding:16px;text-align:center;">
        <div style="color:#F97316;font-weight:700;font-size:13px;
        margin-bottom:10px;">⭐ KAN-YOLOv8</div>
        <div style="color:#94A3B8;font-size:12px;line-height:2.0;">
        mAP50: 0.649<br>
        <span style="color:#F97316;font-weight:700;">
        Precision: 0.724 ★</span><br>
        Recall: 0.618<br>Speed: 0.04s
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats row
    st.markdown("### 🔢 Quick Stats")
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Dataset Images",   "20,327")
    q2.metric("Classes Detected", "9")
    q3.metric("Best Precision",   "0.724 ★")
    q4.metric("Detection Speed",  "0.04s")

    st.markdown("---")

    # How to use
    st.markdown("### 📋 How to Use")
    h1,h2,h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div style="background:#1E293B;border-radius:12px;
        padding:20px;text-align:center;height:130px;">
        <div style="font-size:28px;margin-bottom:8px;">📤</div>
        <div style="font-weight:700;margin-bottom:6px;">Step 1 — Upload</div>
        <div style="color:#94A3B8;font-size:12px;">
        Upload 1–10 pediatric wrist X-ray images
        (JPG, PNG, 16-bit clinical PNG)
        </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div style="background:#1E293B;border-radius:12px;
        padding:20px;text-align:center;height:130px;">
        <div style="font-size:28px;margin-bottom:8px;">🔍</div>
        <div style="font-weight:700;margin-bottom:6px;">Step 2 — Detect</div>
        <div style="color:#94A3B8;font-size:12px;">
        Click Detect button — KAN-YOLOv8
        analyzes all images in 0.04s each
        </div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div style="background:#1E293B;border-radius:12px;
        padding:20px;text-align:center;height:130px;">
        <div style="font-size:28px;margin-bottom:8px;">📊</div>
        <div style="font-weight:700;margin-bottom:6px;">Step 3 — Review</div>
        <div style="color:#94A3B8;font-size:12px;">
        View colored bounding boxes,
        confidence scores and clinical alerts
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 9 Classes
    st.markdown("### 🔬 9 Pathological Classes Detected")
    g1,g2,g3 = st.columns(3)
    cols_cycle = [g1,g2,g3]
    for i,(name,desc) in enumerate(CLASS_DESC.items()):
        em  = CLASS_EMOJI.get(name,'🔍')
        col = CLASS_COLOR_HEX.get(name,'#FFFFFF')
        with cols_cycle[i % 3]:
            st.markdown(f"""
            <div style="background:#1E293B;border-left:3px solid {col};
            border-radius:8px;padding:10px 14px;margin:5px 0;">
            <div style="color:{col};font-weight:700;font-size:13px;">
            {em} {name.upper()}
            </div>
            <div style="color:#64748B;font-size:11px;margin-top:3px;">
            {desc}
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Research note
    st.markdown("""
    <div style="background:#0F172A;border:1px solid #334155;
    border-radius:12px;padding:16px 20px;margin-top:10px;">
    <div style="font-size:13px;color:#94A3B8;line-height:1.8;">
    <b style="color:#F8FAFC;">Research Context:</b>
    This application is part of a study comparing YOLOv8, YOLOv12, and RT-DETR on the
    GRAZPEDWRI-DX dataset (20,327 pediatric wrist radiographs, 9 pathology classes).
    The deployed model (extended fine-tune of YOLOv8) achieves
    <b style="color:#F97316;">Precision = 0.724</b> — the highest among all configurations tested.
    External qualitative validation was performed on de-identified radiographs from
    <b style="color:#60A5FA;">Driscoll Children's Hospital</b>, Corpus Christi, Texas.
    </div>
    </div>
    """, unsafe_allow_html=True)
