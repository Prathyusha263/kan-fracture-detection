import streamlit as st
import numpy as np
from PIL import Image
import time
import tempfile
import os
from pathlib import Path

# Fix cv2 import for Streamlit Cloud
try:
    import cv2
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", 
                   "opencv-python-headless"], check=True)
    import cv2

from ultralytics import YOLO

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="KAN-YOLOv8 Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
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
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebarContent"] * { color: #FFFFFF !important; }
.stButton > button {
    background-color: #F97316 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100% !important;
    padding: 14px !important;
}
.stButton > button:hover {
    background-color: #EA580C !important;
}
[data-testid="stFileUploadDropzone"] {
    background-color: #1E293B !important;
    border: 2px dashed #F97316 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] * { color: #FFFFFF !important; }
[data-testid="stExpander"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] * { color: #FFFFFF !important; }
details summary { color: #FFFFFF !important; }
[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border-radius: 10px !important;
    padding: 12px !important;
    border: 1px solid #334155 !important;
}
[data-testid="stMetric"] * { color: #FFFFFF !important; }
[data-testid="stMetricLabel"] * { color: #94A3B8 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stAlert"] * { color: #FFFFFF !important; }
.stProgress > div > div { background-color: #F97316 !important; }
[data-testid="stSlider"] * { color: #FFFFFF !important; }
label { color: #FFFFFF !important; }
p,h1,h2,h3,h4,h5,h6 { color: #FFFFFF !important; }
table * { color: #FFFFFF !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
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

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading KAN-YOLOv8 AI model...")
def load_model():
    for p in [
        Path(__file__).parent / "best.pt",
        Path("best.pt"),
    ]:
        if p.exists():
            return YOLO(str(p))
    return None

# ── Image Processing ──────────────────────────────────────────
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
            c = CLASS_COLORS_BGR.get(nm, (150,150,150))
            th = 4 if nm == 'fracture' else 3
            cv2.rectangle(out,(x1,y1),(x2,y2),c,th)
            lbl = f"{nm} {cf:.0%}"
            sz,_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
            cv2.rectangle(out,(x1,max(y1-28,0)),(x1+sz[0]+10,y1),c,-1)
            cv2.putText(out,lbl,(x1+5,max(y1-7,12)),
                       cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)
            dets.append({'class':nm,'conf':cf})
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), dets

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦴 KAN-YOLOv8")
    st.markdown("### Fracture Detection AI")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    conf = st.slider(
        "Confidence Threshold",
        0.10, 0.90, 0.30, 0.05,
        help="Lower = more detections | Higher = more certain"
    )
    st.markdown("---")
    st.markdown("### 👥 Research Team")
    st.markdown("""
**Prathyusha Pentam,**

**Dimple Alekya Basimi,**

**Gowtham Kamle**

*Advisor: Dr. S. M. Mallikarjunaiah*

Dept. of Mathematics & Statistics
Texas A&M University – Corpus Christi
    """)
    st.markdown("---")
    st.markdown("### 📊 KAN-YOLOv8 Performance")
    st.markdown("""
| Metric | Score |
|--------|-------|
| mAP50 | 0.649 |
| **Precision** | **0.724 ★** |
| Recall | 0.618 |
| Speed | 0.04s |
| vs YOLOv8 | +2.4% Precision |
    """)
    st.markdown("---")
    st.markdown("### 💡 What is KAN?")
    st.markdown("""
KAN = Kolmogorov-Arnold Network

Replaces standard MLP detection
head with learnable B-spline
activation functions.

Result: **Higher Precision!**
Fewer false alarms!
Better for clinical use!
    """)
    st.markdown("---")
    st.markdown("### 🎨 Color Guide")
    st.markdown("""
🟩 Green = Fracture

🟪 Magenta = Periosteal Reaction

🟨 Yellow = Bone Anomaly

🩵 Cyan = Bone Lesion

🟧 Orange = Pronator Sign

🟦 Blue = Soft Tissue

⬜ Gray = Metal or Text
    """)
    st.markdown("---")
    st.caption("⚠️ For research use only.")

# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════
st.markdown("# 🦴 KAN-YOLOv8 Pediatric Wrist Fracture Detection")
st.markdown(
    "**KAN-YOLOv8** — Kolmogorov-Arnold Network enhanced YOLOv8 | "
    "Trained on GRAZPEDWRI-DX — 20,327 X-ray images | "
    "9 pathological classes | NVIDIA H100 GPU at TAMUCC CREST HPC"
)

st.markdown("""
<div style="background:#1C1A12;border:2px solid #F97316;border-radius:10px;
padding:14px 20px;margin:10px 0;">
<span style="color:#F97316;font-weight:700;font-size:16px;">
⭐ KAN Innovation: Replaced MLP head with Kolmogorov-Arnold Network layers →
Precision improved from 0.700 to 0.724 — Best Precision among ALL models!
</span>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
if model is None:
    st.error("❌ KAN-YOLOv8 model file best.pt not found!")
    st.stop()

st.success("✅ KAN-YOLOv8 model loaded successfully and ready for detection!")
st.markdown("---")

st.markdown("### 📤 Upload X-ray Images")
st.markdown(
    "Upload **1 to 10** pediatric wrist X-ray images — "
    "KAN-YOLOv8 detects all 9 pathological classes with "
    "**highest Precision of 0.724** in just 0.04 seconds!"
)

uploaded_files = st.file_uploader(
    "Drag and drop or browse X-ray images here",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="Supports JPG JPEG PNG including 16-bit PNG from medical scanners"
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning(f"Maximum 10 images! Using first 10.")
        uploaded_files = uploaded_files[:10]

    st.info(f"📁 {len(uploaded_files)} image(s) ready — click Detect!")

    if st.button(
        f"🔍 Detect with KAN-YOLOv8 — All {len(uploaded_files)} Image(s)",
        type="primary"
    ):
        st.markdown("---")
        st.markdown("### 🤖 KAN-YOLOv8 Detection Results")

        prog = st.progress(0, text="Starting KAN-YOLOv8 detection...")
        results = []

        for idx, uf in enumerate(uploaded_files):
            prog.progress(
                (idx+1)/len(uploaded_files),
                text=f"Analyzing {idx+1}/{len(uploaded_files)}: {uf.name}"
            )
            img = load_image(uf)
            if img is None:
                st.error(f"Failed to load {uf.name}")
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

        tot_frac  = sum(len(r['fractures']) for r in results)
        imgs_frac = sum(1 for r in results if r['fractures'])
        tot_finds = sum(len(r['dets']) for r in results)
        avg_t     = sum(r['elapsed'] for r in results) / len(results)

        st.markdown("#### 📊 Batch Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Images Processed", len(results))
        c2.metric("Images with Fractures", imgs_frac)
        c3.metric("Total Findings", tot_finds)
        c4.metric("Avg Detection Time", f"{avg_t:.3f}s")

        if tot_frac > 0:
            st.error(
                f"🚨 FRACTURES DETECTED! "
                f"{tot_frac} fracture(s) across {imgs_frac} image(s). "
                f"Please consult a radiologist immediately!"
            )
        elif tot_finds > 0:
            st.warning(
                f"⚠️ {tot_finds} finding(s) — no fractures. "
                f"Clinical review recommended."
            )
        else:
            st.info(f"✅ No findings at {conf:.0%} confidence.")

        st.markdown("---")
        st.markdown("#### 🖼️ Individual Results")

        for i, r in enumerate(results):
            tag = (
                f"🚨 {len(r['fractures'])} FRACTURE(S)" if r['fractures']
                else f"⚠️ {len(r['dets'])} FINDING(S)" if r['dets']
                else "✅ CLEAR"
            )
            with st.expander(
                f"Image {i+1} — {r['name']} — {tag} — {r['elapsed']:.3f}s",
                expanded=True
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📷 Original X-ray**")
                    st.image(r['original'], use_container_width=True)
                with col2:
                    st.markdown("**🤖 KAN-YOLOv8 Detection**")
                    st.image(r['result'], use_container_width=True)

                if r['fractures']:
                    st.error(
                        f"🚨 FRACTURE DETECTED — "
                        f"{len(r['fractures'])} fracture(s)! "
                        f"Time: {r['elapsed']:.3f}s"
                    )
                elif r['dets']:
                    st.warning(f"⚠️ {len(r['dets'])} finding(s) — no fractures")
                else:
                    st.info("✅ No findings at this confidence level")

                if r['dets']:
                    st.markdown("**📋 Detailed Findings:**")
                    for det in r['dets']:
                        cls  = det['class']
                        cf   = det['conf']
                        em   = CLASS_EMOJI.get(cls, '🔍')
                        desc = CLASS_DESC.get(cls, '')
                        col  = CLASS_COLOR_HEX.get(cls, '#FFFFFF')
                        st.markdown(
                            f"<span style='color:{col};font-weight:700;"
                            f"font-size:15px'>{em} {cls.upper()}</span>"
                            f" — <span style='color:{col};font-weight:700'>"
                            f"{cf:.0%}</span><br>"
                            f"<span style='color:#94A3B8;font-size:12px'>"
                            f"{desc}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("")

                spd = int(600/r['elapsed']) if r['elapsed'] > 0 else 15000
                st.markdown(
                    f"⚡ AI: **{r['elapsed']:.3f}s** | "
                    f"Manual: **10–15 min** | "
                    f"**{spd:,}× faster!**"
                )

        st.markdown("---")
        st.markdown("#### ⚡ Speed Comparison")
        s1,s2,s3 = st.columns(3)
        s1.metric("KAN-YOLOv8 Per Image", f"{avg_t:.3f}s")
        s2.metric("Manual Radiologist", "10–15 min")
        spd = int(600/avg_t) if avg_t > 0 else 15000
        s3.metric("Speed Improvement", f"{spd:,}×")

        st.warning(
            "⚠️ DISCLAIMER: This AI tool is for research and educational "
            "purposes only. All detections must be confirmed by a qualified "
            "radiologist before any clinical decisions are made."
        )

else:
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#1E293B;border:2px solid #2563EB;
        border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:18px;font-weight:700;color:#60A5FA;
        margin-bottom:12px;">Standard YOLOv8</div>
        <div style="font-size:14px;color:#94A3B8;line-height:2">
        mAP50: 0.649<br>Precision: 0.700<br>
        Recall: 0.635<br>Speed: 0.04s
        </div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#1C1A12;border:2px solid #F97316;
        border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:18px;font-weight:700;color:#F97316;
        margin-bottom:12px;">⭐ KAN-YOLOv8</div>
        <div style="font-size:14px;color:#94A3B8;line-height:2">
        mAP50: 0.649<br>
        <span style="color:#F97316;font-weight:700">
        Precision: 0.724 ★ BEST!</span><br>
        Recall: 0.618<br>Speed: 0.04s
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Classes Detected","9")
    col2.metric("Speed Per Image","0.04s")
    col3.metric("Best Precision","0.724 ★")
    col4.metric("Max Images","10")

    st.markdown("---")
    st.markdown("### 📋 How to Use")
    h1,h2,h3 = st.columns(3)
    with h1:
        st.info("**📤 Step 1 — Upload**\n\nUpload 1 to 10 wrist X-ray images")
    with h2:
        st.info("**🔍 Step 2 — Detect**\n\nClick Detect — results in 0.04s!")
    with h3:
        st.info("**📊 Step 3 — Results**\n\nView bounding boxes and findings")

    st.markdown("---")
    st.markdown("### 🔬 9 Pathological Classes")
    c1,c2,c3 = st.columns(3)
    for i,(name,desc) in enumerate(CLASS_DESC.items()):
        em  = CLASS_EMOJI.get(name,'🔍')
        col = CLASS_COLOR_HEX.get(name,'#FFFFFF')
        with [c1,c2,c3][i%3]:
            st.markdown(
                f"<span style='color:{col};font-weight:700;"
                f"font-size:14px'>{em} {name.upper()}</span><br>"
                f"<span style='color:#94A3B8;font-size:12px'>{desc}</span>",
                unsafe_allow_html=True
            )
            st.markdown("---")
