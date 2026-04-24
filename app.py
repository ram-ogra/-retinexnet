"""
app.py – RetinexNet Streamlit UI
Run with: streamlit run app.py
"""

import os, sys, io, time, warnings
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from models.decom_net   import DecomNet
from models.enhance_net import EnhanceNet
from utils.image_utils  import (
    numpy_to_tensor, tensor_to_numpy, tensor_to_numpy_single,
    pad_to_multiple, unpad, reconstruct,
    illumination_to_rgb, get_device,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinexNet · Low-Light Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }

/* ── Header banner ── */
.banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    border: 1px solid rgba(99,102,241,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.banner h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg,#818cf8,#c084fc,#38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; padding: 0;
}
.banner p { color:#94a3b8; margin:0.4rem 0 0; font-size:0.95rem; }

/* ── Metric cards ── */
.metric-card {
    background: #1e2233;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value { font-size:1.6rem; font-weight:700; color:#818cf8; }
.metric-card .label { font-size:0.75rem; color:#64748b; margin-top:2px; }

/* ── Image cards ── */
.img-card {
    background: #1e2233;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
}
.img-title {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: #818cf8;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #13151f !important; }

/* ── Buttons ── */
.stDownloadButton > button {
    width: 100%;
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}

/* ── Progress / status ── */
.status-ok  { color:#34d399; font-weight:600; }
.status-warn{ color:#fbbf24; font-weight:600; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e2233; border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #6366f1 !important; color: white !important;
}

/* ── Dividers ── */
hr { border-color: rgba(99,102,241,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS_DIR     = os.path.join(os.path.dirname(__file__), "weights")
DECOM_WEIGHTS   = os.path.join(WEIGHTS_DIR, "pretrained_decom.pth")
ENHANCE_WEIGHTS = os.path.join(WEIGHTS_DIR, "pretrained_enhance.pth")
MAX_DIM         = 1024   # resize large images to keep inference fast


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models_cached():
    device    = get_device()
    decom     = DecomNet().to(device)
    enhance   = EnhanceNet().to(device)

    weights_ok = {"decom": False, "enhance": False}

    if os.path.isfile(DECOM_WEIGHTS):
        state = torch.load(DECOM_WEIGHTS, map_location=device)
        decom.load_state_dict(state, strict=False)
        weights_ok["decom"] = True

    if os.path.isfile(ENHANCE_WEIGHTS):
        state = torch.load(ENHANCE_WEIGHTS, map_location=device)
        enhance.load_state_dict(state, strict=False)
        weights_ok["enhance"] = True

    decom.eval()
    enhance.eval()
    return decom, enhance, device, weights_ok


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline (same logic as inference.py, wrapped for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_pipeline(image_np, decom, enhance, device, gamma=1.0):
    """Returns dict of all intermediate + final arrays (float32, [0,1])."""
    t0 = time.time()

    tensor        = numpy_to_tensor(image_np, device)
    padded, pads  = pad_to_multiple(tensor, multiple=8)

    refl, illu    = decom(padded)
    enh_illu      = enhance(illu, refl)

    if gamma != 1.0:
        enh_illu = torch.clamp(enh_illu ** gamma, 0.0, 1.0)

    enhanced      = reconstruct(refl, enh_illu)

    # Unpad
    refl          = unpad(refl,     pads)
    illu          = unpad(illu,     pads)
    enh_illu      = unpad(enh_illu, pads)
    enhanced      = unpad(enhanced, pads)

    elapsed_ms    = (time.time() - t0) * 1000

    return {
        "original":              image_np,
        "reflectance":           np.clip(tensor_to_numpy(refl),     0, 1),
        "illumination":          np.clip(tensor_to_numpy_single(illu),      0, 1),
        "enhanced_illumination": np.clip(tensor_to_numpy_single(enh_illu),  0, 1),
        "enhanced":              np.clip(tensor_to_numpy(enhanced),  0, 1),
        "elapsed_ms":            elapsed_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pil_from_np(arr, is_gray=False):
    """Convert float32 [0,1] numpy array → PIL Image."""
    uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if is_gray:
        return Image.fromarray(uint8, mode="L")
    return Image.fromarray(uint8, mode="RGB")


def np_to_bytes(arr, fmt="PNG", is_gray=False):
    """Encode numpy array to PNG/JPEG bytes for download."""
    pil = pil_from_np(arr, is_gray=is_gray)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=95)
    return buf.getvalue()


def resize_if_large(image_np, max_dim=MAX_DIM):
    h, w = image_np.shape[:2]
    if max(h, w) > max_dim:
        scale  = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_np


def brightness_score(arr):
    """Mean luminance in [0,100]."""
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    return float(np.mean(gray) * 100)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def render_image_card(title, arr, caption="", is_gray=False, use_container_width=True):
    st.markdown(f'<div class="img-title">{title}</div>', unsafe_allow_html=True)
    if is_gray:
        arr_disp = illumination_to_rgb(arr)
    else:
        arr_disp = arr
    st.image(arr_disp, caption=caption, use_container_width=use_container_width, clamp=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar(weights_ok, device):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        gamma = st.slider(
            "🌟 Gamma Boost",
            min_value=0.2, max_value=2.0, value=0.8, step=0.05,
            help="< 1 = brighter output  |  > 1 = darker output",
        )

        max_dim = st.select_slider(
            "📐 Max Image Dimension (px)",
            options=[256, 512, 768, 1024, 1280, 1536],
            value=1024,
            help="Resize input image to this size before inference (faster for large images).",
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Status")

        device_str = str(device).upper()
        icon = "🟢" if "cuda" in device_str.lower() else "🔵"
        st.markdown(f"{icon} **Device:** `{device_str}`")

        d_status = "✅ Loaded" if weights_ok["decom"]   else "⚠️ Demo (random)"
        e_status = "✅ Loaded" if weights_ok["enhance"] else "⚠️ Demo (random)"
        st.markdown(f"**DecomNet:**   {d_status}")
        st.markdown(f"**EnhanceNet:** {e_status}")

        if not weights_ok["decom"] or not weights_ok["enhance"]:
            st.warning(
                "Pretrained weights not found.\n\n"
                "Run in terminal:\n```\npython download_weights.py\n```\n"
                "Results will look noisy until real weights are loaded."
            )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "**RetinexNet** decomposes images into **Reflectance** (colour) "
            "and **Illumination** (light) using Retinex theory, then enhances "
            "only the illumination layer to brighten low-light photos."
        )
        st.markdown("---")
        st.markdown(
            "🛠️ Built by **[ram-ogra](https://github.com/ram-ogra)**"
        )

    return gamma, max_dim


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="banner">
        <h1>✨ RetinexNet · Low-Light Image Enhancer</h1>
        <p>Deep Retinex Decomposition · Reflectance & Illumination Analysis · GPU Accelerated</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        decom, enhance, device, weights_ok = load_models_cached()

    # ── Sidebar ───────────────────────────────────────────────────────────
    gamma, max_dim = build_sidebar(weights_ok, device)

    # ── Upload section ────────────────────────────────────────────────────
    st.markdown("### 📤 Upload a Low-Light Image")
    uploaded = st.file_uploader(
        "Supported: JPG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        _show_placeholder()
        return

    # ── Decode uploaded image ─────────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("❌ Could not decode image. Please upload a valid image file.")
        return
    rgb_np   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_np   = resize_if_large(rgb_np, max_dim=max_dim)
    H, W, _  = rgb_np.shape

    # ── Run inference ─────────────────────────────────────────────────────
    with st.spinner("🔍 Running RetinexNet inference…"):
        results = run_pipeline(rgb_np, decom, enhance, device, gamma=gamma)

    elapsed   = results["elapsed_ms"]
    bright_in = brightness_score(results["original"])
    bright_out= brightness_score(results["enhanced"])
    improvement = bright_out - bright_in

    # ── Metric cards ──────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, f"{W}×{H}",           "Image Size"),
        (c2, f"{elapsed:.0f} ms",  "Inference Time"),
        (c3, f"{bright_in:.1f}",   "Input Brightness"),
        (c4, f"{bright_out:.1f}",  "Output Brightness"),
        (c5, f"+{improvement:.1f}","Brightness Gain"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="value">{val}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Tabs: Results / Decomposition / Comparison / Analysis ─────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Results",
        "🔬 Decomposition Maps",
        "↔️ Side-by-Side",
        "📊 Analysis",
    ])

    # ══ TAB 1 – Results ═══════════════════════════════════════════════════
    with tab1:
        col_orig, col_enh = st.columns(2)
        with col_orig:
            render_image_card("Original (Low-Light)", results["original"])
        with col_enh:
            render_image_card("Enhanced Output", results["enhanced"])

        st.markdown("#### 💾 Download")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "⬇️ Enhanced Image (PNG)",
                data=np_to_bytes(results["enhanced"], "PNG"),
                file_name="enhanced.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Enhanced Image (JPEG)",
                data=np_to_bytes(results["enhanced"], "JPEG"),
                file_name="enhanced.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "⬇️ Reflectance Map",
                data=np_to_bytes(results["reflectance"], "PNG"),
                file_name="reflectance.png",
                mime="image/png",
                use_container_width=True,
            )

    # ══ TAB 2 – Decomposition ═════════════════════════════════════════════
    with tab2:
        st.markdown(
            "The Retinex model states: **Image = Reflectance × Illumination**. "
            "DecomNet splits your image into these two components."
        )
        col_r, col_i, col_ei = st.columns(3)
        with col_r:
            render_image_card(
                "Reflectance (R)",
                results["reflectance"],
                caption="Intrinsic colour / material properties",
            )
        with col_i:
            render_image_card(
                "Raw Illumination (L)",
                results["illumination"],
                caption="Input lighting map (heatmap)",
                is_gray=True,
            )
        with col_ei:
            render_image_card(
                "Enhanced Illumination (L̂)",
                results["enhanced_illumination"],
                caption="EnhanceNet output (heatmap)",
                is_gray=True,
            )

        dl_r, dl_i, dl_ei = st.columns(3)
        with dl_r:
            st.download_button(
                "⬇️ Reflectance",
                np_to_bytes(results["reflectance"]),
                "reflectance.png", "image/png",
                use_container_width=True,
            )
        with dl_i:
            st.download_button(
                "⬇️ Illumination",
                np_to_bytes(illumination_to_rgb(results["illumination"])),
                "illumination.png", "image/png",
                use_container_width=True,
            )
        with dl_ei:
            st.download_button(
                "⬇️ Enhanced Illumination",
                np_to_bytes(illumination_to_rgb(results["enhanced_illumination"])),
                "enhanced_illumination.png", "image/png",
                use_container_width=True,
            )

    # ══ TAB 3 – Side-by-Side ══════════════════════════════════════════════
    with tab3:
        st.markdown("##### 🔎 Drag the slider below to compare (visual diff)")

        # Build a concatenated comparison image
        orig_u8  = (np.clip(results["original"], 0, 1) * 255).astype(np.uint8)
        enh_u8   = (np.clip(results["enhanced"], 0, 1) * 255).astype(np.uint8)
        gap      = np.ones((H, 6, 3), dtype=np.uint8) * 80
        comp     = np.concatenate([orig_u8, gap, enh_u8], axis=1)

        st.image(comp, caption="← Original  |  Enhanced →", use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📏 All Four Maps Together")

        illu_rgb  = (illumination_to_rgb(results["illumination"]) * 255).astype(np.uint8)
        eillu_rgb = (illumination_to_rgb(results["enhanced_illumination"]) * 255).astype(np.uint8)
        refl_u8   = (np.clip(results["reflectance"], 0, 1) * 255).astype(np.uint8)
        gap2      = np.ones((H, 4, 3), dtype=np.uint8) * 80
        quad      = np.concatenate([orig_u8, gap2, refl_u8, gap2, illu_rgb, gap2, enh_u8], axis=1)

        st.image(quad,
                 caption="Original  |  Reflectance  |  Illumination (heatmap)  |  Enhanced",
                 use_container_width=True)

    # ══ TAB 4 – Analysis ══════════════════════════════════════════════════
    with tab4:
        st.markdown("##### 📈 Pixel Brightness Distribution")

        import json

        def build_histogram_html(orig_np, enh_np):
            def hist_data(arr):
                gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
                counts, _ = np.histogram(gray, bins=64, range=(0,1))
                return (counts / counts.max()).tolist()

            orig_hist = hist_data(orig_np)
            enh_hist  = hist_data(enh_np)

            html = f"""
            <div style="background:#1e2233;border-radius:12px;padding:20px;">
              <canvas id="histChart" height="200"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <script>
            const labels = Array.from({{length:64}}, (_,i) => (i/64).toFixed(2));
            new Chart(document.getElementById('histChart'), {{
              type: 'line',
              data: {{
                labels,
                datasets: [
                  {{label:'Original', data:{json.dumps(orig_hist)},
                   borderColor:'#64748b',backgroundColor:'rgba(100,116,139,0.15)',
                   borderWidth:2,fill:true,tension:0.4,pointRadius:0}},
                  {{label:'Enhanced', data:{json.dumps(enh_hist)},
                   borderColor:'#818cf8',backgroundColor:'rgba(129,140,248,0.2)',
                   borderWidth:2,fill:true,tension:0.4,pointRadius:0}},
                ]
              }},
              options:{{
                responsive:true,
                plugins:{{
                  legend:{{labels:{{color:'#94a3b8'}}}},
                  title:{{display:true,text:'Normalised Brightness Histogram',color:'#e2e8f0',font:{{size:14}}}}
                }},
                scales:{{
                  x:{{ticks:{{color:'#64748b',maxTicksLimit:10}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
                  y:{{ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.05)'}}}}
                }}
              }}
            }});
            </script>
            """
            return html

        hist_html = build_histogram_html(results["original"], results["enhanced"])
        st.components.v1.html(hist_html, height=280)

        st.markdown("---")
        st.markdown("##### 🔢 Quantitative Metrics")

        orig_u8_  = (np.clip(results["original"],  0,1)*255).astype(np.float32)
        refl_u8_  = (np.clip(results["reflectance"],0,1)*255).astype(np.float32)
        enh_u8_   = (np.clip(results["enhanced"],  0,1)*255).astype(np.float32)

        def contrast(arr):
            gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
            return float(gray.std())

        def sharpness(arr):
            gray = cv2.cvtColor((arr).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        orig_arr = (np.clip(results["original"], 0,1)*255).astype(np.uint8)
        enh_arr  = (np.clip(results["enhanced"], 0,1)*255).astype(np.uint8)

        ma, mb, mc, md = st.columns(4)
        with ma:
            st.metric("Input Brightness",  f"{bright_in:.2f}")
        with mb:
            st.metric("Output Brightness", f"{bright_out:.2f}", delta=f"{improvement:+.2f}")
        with mc:
            st.metric("Input Contrast (σ)",  f"{contrast(orig_u8_):.2f}")
        with md:
            st.metric("Output Contrast (σ)", f"{contrast(enh_u8_):.2f}",
                      delta=f"{contrast(enh_u8_)-contrast(orig_u8_):+.2f}")

        me, mf, mg, mh = st.columns(4)
        with me:
            st.metric("Input Sharpness",  f"{sharpness(orig_arr):.1f}")
        with mf:
            st.metric("Output Sharpness", f"{sharpness(enh_arr):.1f}",
                      delta=f"{sharpness(enh_arr)-sharpness(orig_arr):+.1f}")
        with mg:
            mean_illu  = float(np.mean(results["illumination"])  * 100)
            mean_eillu = float(np.mean(results["enhanced_illumination"]) * 100)
            st.metric("Avg Raw Illu (%)",  f"{mean_illu:.1f}")
        with mh:
            st.metric("Avg Enh Illu (%)", f"{mean_eillu:.1f}",
                      delta=f"{mean_eillu-mean_illu:+.1f}")

        st.markdown("---")
        st.markdown("##### 🎨 Per-Channel Statistics")
        ch_names = ["Red", "Green", "Blue"]
        ch_cols  = st.columns(3)
        for ci, (ch, col) in enumerate(zip(ch_names, ch_cols)):
            with col:
                i_mean = float(results["original"][:,:,ci].mean()*100)
                o_mean = float(results["enhanced"][:,:,ci].mean()*100)
                st.markdown(f"**{ch} channel**")
                st.metric(f"Input mean", f"{i_mean:.1f}")
                st.metric(f"Output mean", f"{o_mean:.1f}", delta=f"{o_mean-i_mean:+.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder shown before upload
# ─────────────────────────────────────────────────────────────────────────────

def _show_placeholder():
    st.markdown("""
    <div style="
        background: #1e2233;
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin-top: 1rem;
    ">
        <div style="font-size:3.5rem;margin-bottom:1rem;">🌙</div>
        <h3 style="color:#818cf8;margin-bottom:0.5rem;">
            Upload a Low-Light Image to Get Started
        </h3>
        <p style="color:#64748b;max-width:520px;margin:0 auto 1.5rem;">
            RetinexNet will decompose your image into <strong style="color:#c084fc">Reflectance</strong>
            and <strong style="color:#38bdf8">Illumination</strong> components,
            enhance the lighting, and reconstruct a brighter output.
        </p>
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;color:#475569;font-size:0.85rem;">
            <span>📸 JPG / PNG / BMP / WEBP</span>
            <span>⚡ GPU Accelerated</span>
            <span>🔬 Intermediate Maps</span>
            <span>📊 Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 How It Works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "Drop any dark / low-light photo"),
        ("2️⃣", "Decompose", "DecomNet splits R & L layers"),
        ("3️⃣", "Enhance", "EnhanceNet brightens L layer"),
        ("4️⃣", "Reconstruct", "Final image = R × L_enhanced"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(
                f"""<div style="background:#1e2233;border-radius:12px;padding:1.2rem;text-align:center;border:1px solid rgba(99,102,241,0.15);">
                <div style="font-size:2rem">{icon}</div>
                <div style="color:#818cf8;font-weight:700;margin:0.5rem 0 0.3rem">{title}</div>
                <div style="color:#64748b;font-size:0.82rem">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()