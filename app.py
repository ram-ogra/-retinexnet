"""
app.py - RetinexNet Streamlit UI
- Uses neural pipeline when real pretrained weights are loaded
- Falls back to traditional CLAHE+Gamma enhancement for demo weights
"""

import os, sys, io, time, warnings
import numpy as np
import torch
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.decom_net   import DecomNet
from models.enhance_net import EnhanceNet
from utils.image_utils  import (
    numpy_to_tensor, tensor_to_numpy, tensor_to_numpy_single,
    pad_to_multiple, unpad, reconstruct,
    illumination_to_rgb, resize_image, compute_sharpness, get_device,
)
from utils.fallback_enhance import enhance_traditional

st.set_page_config(
    page_title="RetinexNet · Low-Light Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.banner {
    background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    border-radius:16px; padding:2rem 2.5rem 1.6rem;
    margin-bottom:1.8rem; border:1px solid rgba(99,102,241,0.25);
    box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
.banner h1 {
    font-size:2.2rem; font-weight:800;
    background:linear-gradient(90deg,#818cf8,#c084fc,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.banner p { color:#94a3b8; margin:0.4rem 0 0; font-size:0.95rem; }
.metric-card {
    background:#1e2233; border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:1rem 1.2rem; text-align:center;
}
.metric-card .value { font-size:1.6rem; font-weight:700; color:#818cf8; }
.metric-card .label { font-size:0.75rem; color:#64748b; margin-top:2px; }
.img-title {
    font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase; color:#818cf8; margin-bottom:0.5rem;
}
section[data-testid="stSidebar"] { background:#13151f !important; }
.stDownloadButton > button {
    width:100%; background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important;
}
.stTabs [data-baseweb="tab-list"] { background:#1e2233; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#64748b; border-radius:8px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:#6366f1 !important; color:white !important;
}
</style>
""", unsafe_allow_html=True)

WEIGHTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DECOM_WEIGHTS   = os.path.join(WEIGHTS_DIR, "pretrained_decom.pth")
ENHANCE_WEIGHTS = os.path.join(WEIGHTS_DIR, "pretrained_enhance.pth")
MIN_WEIGHT_SIZE = 500_000   # real weights > 500KB; demo weights < 10KB


def _is_real_weights(path: str) -> bool:
    """Check if the weights file is real pretrained (not random demo)."""
    return os.path.isfile(path) and os.path.getsize(path) > MIN_WEIGHT_SIZE


@st.cache_resource(show_spinner=False)
def load_models_cached():
    device  = get_device()
    decom   = DecomNet().to(device)
    enhance = EnhanceNet().to(device)

    decom_real   = _is_real_weights(DECOM_WEIGHTS)
    enhance_real = _is_real_weights(ENHANCE_WEIGHTS)

    if os.path.isfile(DECOM_WEIGHTS):
        state = torch.load(DECOM_WEIGHTS, map_location=device)
        decom.load_state_dict(state, strict=False)

    if os.path.isfile(ENHANCE_WEIGHTS):
        state = torch.load(ENHANCE_WEIGHTS, map_location=device)
        enhance.load_state_dict(state, strict=False)

    decom.eval(); enhance.eval()

    weights_ok = {
        "decom":   decom_real,
        "enhance": enhance_real,
        "use_neural": decom_real and enhance_real,  # only use NN if BOTH are real
    }
    return decom, enhance, device, weights_ok


@torch.no_grad()
def run_neural_pipeline(image_np, decom, enhance, device, gamma=1.0):
    """Full RetinexNet neural inference."""
    t0           = time.time()
    tensor       = numpy_to_tensor(image_np, device)
    padded, pads = pad_to_multiple(tensor, multiple=8)
    refl, illu   = decom(padded)
    enh_illu     = enhance(illu, refl)
    if gamma != 1.0:
        enh_illu = torch.clamp(enh_illu ** gamma, 0.0, 1.0)
    enhanced     = reconstruct(refl, enh_illu)
    refl         = unpad(refl,     pads)
    illu         = unpad(illu,     pads)
    enh_illu     = unpad(enh_illu, pads)
    enhanced     = unpad(enhanced, pads)
    return {
        "original":              image_np,
        "reflectance":           np.clip(tensor_to_numpy(refl),    0, 1),
        "illumination":          np.clip(tensor_to_numpy_single(illu),     0, 1),
        "enhanced_illumination": np.clip(tensor_to_numpy_single(enh_illu), 0, 1),
        "enhanced":              np.clip(tensor_to_numpy(enhanced), 0, 1),
        "elapsed_ms":            (time.time() - t0) * 1000,
        "mode":                  "neural",
    }


def run_fallback_pipeline(image_np, gamma=1.0):
    """Traditional CLAHE+Gamma enhancement — works without pretrained weights."""
    t0      = time.time()
    results = enhance_traditional(image_np, gamma=gamma)
    results["original"]   = image_np
    results["elapsed_ms"] = (time.time() - t0) * 1000
    results["mode"]       = "traditional"
    return results


def np_to_bytes(arr, fmt="PNG"):
    uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    # Handle grayscale
    if arr.ndim == 2:
        pil = Image.fromarray(uint8, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=95)
    return buf.getvalue()


def brightness_score(arr):
    return float((0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).mean() * 100)


def render_card(title, arr, is_gray=False, caption=""):
    st.markdown(f'<div class="img-title">{title}</div>', unsafe_allow_html=True)
    disp = illumination_to_rgb(arr) if is_gray else arr
    st.image(disp, use_container_width=True, clamp=True, caption=caption)


def build_sidebar(weights_ok, device):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")
        gamma = st.slider("🌟 Gamma Boost", 0.2, 2.0, 0.7, 0.05,
                          help="< 1 = brighter  |  > 1 = darker")
        max_dim = st.select_slider("📐 Max Dimension (px)",
                                   options=[256,512,768,1024,1280], value=1024)
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        icon = "🟢" if "cuda" in str(device) else "🔵"
        st.markdown(f"{icon} **Device:** `{str(device).upper()}`")

        if weights_ok["use_neural"]:
            st.markdown("**DecomNet:**   ✅ Pretrained")
            st.markdown("**EnhanceNet:** ✅ Pretrained")
            st.markdown("**Mode:** 🧠 Neural (RetinexNet)")
        else:
            st.markdown("**DecomNet:**   ⚠️ Demo weights")
            st.markdown("**EnhanceNet:** ⚠️ Demo weights")
            st.markdown("**Mode:** ⚡ Traditional (CLAHE+Gamma)")
            st.info(
                "Using classical enhancement.\n\n"
                "For neural RetinexNet output, add real pretrained weights to `weights/` folder."
            )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "Separates any image into **Reflectance** & **Illumination**, "
            "then enhances only the light layer for natural brightening."
        )
        st.markdown("---")
        st.markdown("🛠️ Built by **[ram-ogra](https://github.com/ram-ogra)**")
    return gamma, max_dim


def main():
    st.markdown("""
    <div class="banner">
        <h1>✨ RetinexNet · Low-Light Image Enhancer</h1>
        <p>Deep Retinex Decomposition · Reflectance & Illumination Analysis · GPU Accelerated</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        decom, enhance, device, weights_ok = load_models_cached()

    gamma, max_dim = build_sidebar(weights_ok, device)

    st.markdown("### 📤 Upload a Low-Light Image")
    uploaded = st.file_uploader(
        "JPG, PNG, BMP, WEBP",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        _placeholder()
        return

    pil_img  = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_img, dtype=np.float32) / 255.0
    image_np = resize_image(image_np, max_dim=max_dim)
    H, W, _  = image_np.shape

    with st.spinner("🔍 Enhancing image…"):
        if weights_ok["use_neural"]:
            results = run_neural_pipeline(image_np, decom, enhance, device, gamma=gamma)
        else:
            results = run_fallback_pipeline(image_np, gamma=gamma)

    bright_in  = brightness_score(results["original"])
    bright_out = brightness_score(results["enhanced"])

    # ── Mode badge ────────────────────────────────────────────────────────
    mode_label = "🧠 Neural (RetinexNet)" if results["mode"] == "neural" else "⚡ Traditional (CLAHE+Gamma)"
    st.markdown(f"**Enhancement Mode:** {mode_label}")
    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{W}×{H}",                      "Image Size"),
        (c2, f"{results['elapsed_ms']:.0f} ms","Inference Time"),
        (c3, f"{bright_in:.1f}",               "Input Brightness"),
        (c4, f"{bright_out:.1f}",              "Output Brightness"),
        (c5, f"+{bright_out-bright_in:.1f}",   "Brightness Gain"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                        f'<div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Results","🔬 Decomposition","↔️ Compare","📊 Analysis"])

    # ══ TAB 1 ════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns(2)
        with col1: render_card("Original (Low-Light)", results["original"])
        with col2: render_card("Enhanced Output",      results["enhanced"])

        st.markdown("#### 💾 Download")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇️ Enhanced PNG",
                np_to_bytes(results["enhanced"], "PNG"),
                "enhanced.png", "image/png",
                use_container_width=True, key="dl_t1_png")
        with d2:
            st.download_button("⬇️ Enhanced JPEG",
                np_to_bytes(results["enhanced"], "JPEG"),
                "enhanced.jpg", "image/jpeg",
                use_container_width=True, key="dl_t1_jpg")
        with d3:
            st.download_button("⬇️ Reflectance",
                np_to_bytes(results["reflectance"], "PNG"),
                "reflectance.png", "image/png",
                use_container_width=True, key="dl_t1_refl")

    # ══ TAB 2 ════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("**Image = Reflectance × Illumination** — the two intrinsic layers of any photo.")
        c1, c2, c3 = st.columns(3)
        with c1: render_card("Reflectance (R)", results["reflectance"],
                              caption="Material colour / texture")
        with c2: render_card("Raw Illumination (L)", results["illumination"], is_gray=True,
                              caption="Input lighting map")
        with c3: render_card("Enhanced Illumination (L̂)", results["enhanced_illumination"], is_gray=True,
                              caption="Brightened lighting map")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇️ Reflectance",
                np_to_bytes(results["reflectance"]),
                "reflectance.png", "image/png",
                use_container_width=True, key="dl_t2_refl")
        with d2:
            st.download_button("⬇️ Illumination",
                np_to_bytes(illumination_to_rgb(results["illumination"])),
                "illumination.png", "image/png",
                use_container_width=True, key="dl_t2_illu")
        with d3:
            st.download_button("⬇️ Enhanced Illumination",
                np_to_bytes(illumination_to_rgb(results["enhanced_illumination"])),
                "enh_illu.png", "image/png",
                use_container_width=True, key="dl_t2_eillu")

    # ══ TAB 3 ════════════════════════════════════════════════════════════
    with tab3:
        orig_u8 = (np.clip(results["original"], 0,1)*255).astype(np.uint8)
        enh_u8  = (np.clip(results["enhanced"], 0,1)*255).astype(np.uint8)
        gap     = np.ones((H,6,3), dtype=np.uint8) * 80
        st.image(np.concatenate([orig_u8, gap, enh_u8], axis=1),
                 caption="← Original  |  Enhanced →", use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📏 All Four Maps")
        refl_u8  = (np.clip(results["reflectance"],0,1)*255).astype(np.uint8)
        illu_rgb = (illumination_to_rgb(results["illumination"])*255).astype(np.uint8)
        gap2     = np.ones((H,4,3),dtype=np.uint8)*80
        quad     = np.concatenate([orig_u8,gap2,refl_u8,gap2,illu_rgb,gap2,enh_u8],axis=1)
        st.image(quad, caption="Original | Reflectance | Illumination | Enhanced",
                 use_container_width=True)

    # ══ TAB 4 ════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("##### 📈 Brightness Distribution")
        import json
        def hist_data(arr):
            gray = 0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]
            c, _ = np.histogram(gray, bins=64, range=(0,1))
            return (c / max(c.max(),1)).tolist()

        html = f"""
        <div style="background:#1e2233;border-radius:12px;padding:20px">
          <canvas id="hc" height="200"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script>
        const lbs=Array.from({{length:64}},(_,i)=>(i/64).toFixed(2));
        new Chart(document.getElementById('hc'),{{
          type:'line',
          data:{{labels:lbs,datasets:[
            {{label:'Original',data:{json.dumps(hist_data(results["original"]))},
             borderColor:'#64748b',backgroundColor:'rgba(100,116,139,0.15)',
             borderWidth:2,fill:true,tension:0.4,pointRadius:0}},
            {{label:'Enhanced',data:{json.dumps(hist_data(results["enhanced"]))},
             borderColor:'#818cf8',backgroundColor:'rgba(129,140,248,0.2)',
             borderWidth:2,fill:true,tension:0.4,pointRadius:0}}
          ]}},
          options:{{responsive:true,
            plugins:{{legend:{{labels:{{color:'#94a3b8'}}}},
              title:{{display:true,text:'Normalised Brightness Histogram',color:'#e2e8f0'}}}},
            scales:{{
              x:{{ticks:{{color:'#64748b',maxTicksLimit:10}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
              y:{{ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.05)'}}}}
            }}
          }}
        }});
        </script>"""
        st.components.v1.html(html, height=280)

        st.markdown("---")
        st.markdown("##### 🔢 Metrics")
        def contrast(arr):
            return float((0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]).std()*100)

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Input Brightness",  f"{bright_in:.1f}")
        with m2: st.metric("Output Brightness", f"{bright_out:.1f}",
                            delta=f"{bright_out-bright_in:+.1f}")
        with m3: st.metric("Input Contrast",    f"{contrast(results['original']):.1f}")
        with m4: st.metric("Output Contrast",   f"{contrast(results['enhanced']):.1f}",
                            delta=f"{contrast(results['enhanced'])-contrast(results['original']):+.1f}")

        m5,m6,m7,m8 = st.columns(4)
        mi   = float(np.mean(results["illumination"])*100)
        mei  = float(np.mean(results["enhanced_illumination"])*100)
        sh_i = compute_sharpness(results["original"])
        sh_o = compute_sharpness(results["enhanced"])
        with m5: st.metric("Input Sharpness",  f"{sh_i:.1f}")
        with m6: st.metric("Output Sharpness", f"{sh_o:.1f}", delta=f"{sh_o-sh_i:+.1f}")
        with m7: st.metric("Avg Raw Illu (%)", f"{mi:.1f}")
        with m8: st.metric("Avg Enh Illu (%)", f"{mei:.1f}", delta=f"{mei-mi:+.1f}")

        st.markdown("---")
        st.markdown("##### 🎨 Per-Channel Stats")
        ch_cols = st.columns(3)
        for ci, (ch, col) in enumerate(zip(["Red","Green","Blue"], ch_cols)):
            im = float(results["original"][:,:,ci].mean()*100)
            om = float(results["enhanced"][:,:,ci].mean()*100)
            with col:
                st.markdown(f"**{ch}**")
                ca, cb = st.columns(2)
                with ca: st.metric("Input",  f"{im:.1f}")
                with cb: st.metric("Output", f"{om:.1f}", delta=f"{om-im:+.1f}")


def _placeholder():
    st.markdown("""
    <div style="background:#1e2233;border:2px dashed rgba(99,102,241,0.3);
    border-radius:16px;padding:3rem;text-align:center;margin-top:1rem;">
        <div style="font-size:3.5rem;margin-bottom:1rem;">🌙</div>
        <h3 style="color:#818cf8;">Upload a Low-Light Image to Get Started</h3>
        <p style="color:#64748b;max-width:520px;margin:0 auto 1.5rem;">
            Separates your image into <strong style="color:#c084fc">Reflectance</strong>
            and <strong style="color:#38bdf8">Illumination</strong>, enhances the light,
            and reconstructs a naturally brighter output.
        </p>
        <div style="color:#475569;font-size:0.85rem;">
            📸 JPG / PNG / BMP / WEBP &nbsp;·&nbsp;
            ⚡ Always works &nbsp;·&nbsp;
            🔬 Decomposition Maps &nbsp;·&nbsp;
            📊 Analytics
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 How It Works")
    c1,c2,c3,c4 = st.columns(4)
    for col, (icon, title, desc) in zip([c1,c2,c3,c4],[
        ("1️⃣","Upload",      "Drop any dark / low-light photo"),
        ("2️⃣","Decompose",   "Split into Reflectance & Light"),
        ("3️⃣","Enhance",     "Brighten the light layer only"),
        ("4️⃣","Reconstruct", "Final = Reflectance × Light"),
    ]):
        with col:
            st.markdown(
                f'<div style="background:#1e2233;border-radius:12px;padding:1.2rem;'
                f'text-align:center;border:1px solid rgba(99,102,241,0.15);">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="color:#818cf8;font-weight:700;margin:0.5rem 0 0.3rem">{title}</div>'
                f'<div style="color:#64748b;font-size:0.82rem">{desc}</div></div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()