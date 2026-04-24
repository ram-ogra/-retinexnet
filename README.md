<div align="center">

# ✨ RetinexNet — Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Built by ram-ogra](https://img.shields.io/badge/Built%20by-ram--ogra-818cf8?style=flat-square&logo=github&logoColor=white)](https://github.com/ram-ogra)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

**A deep learning system that enhances dark, low-light images**  
*by decomposing light and colour — then fixing only the light.*

[🚀 Quick Start](#-quick-start) · [🖼️ How It Works](#-how-it-works) · [🏗️ Architecture](#-architecture) · [📊 Features](#-features)

</div>

---

## 💡 The Idea

Ever taken a photo at night and it came out too dark to use?

I built this project to solve exactly that. The core insight is simple:

> **Any image = Colour information × Lighting information**

Instead of blindly brightening the whole image (which destroys colour and adds noise), this system **separates the two layers**, enhances only the lighting, and puts them back together. The result is a naturally bright image that keeps its original colours intact.

---

## 🖼️ How It Works

```
Your Dark Photo
      │
      ▼
┌─────────────┐
│  DecomNet   │  ──→  Separates colour (Reflectance) & light (Illumination)
└─────────────┘
      │
      ▼
┌──────────────┐
│  EnhanceNet  │  ──→  Intelligently brightens the Illumination layer
└──────────────┘
      │
      ▼
 Reflectance × Enhanced Illumination
      │
      ▼
  ✨ Bright, Natural Output
```

---

## 🏗️ Architecture

### DecomNet — Image Decomposer
- 5-layer convolutional network
- Input: Your image (3 channels) + a brightness hint (1 channel) = **4 channels**
- Output: **Reflectance map** (colour) + **Illumination map** (light)
- Keeps all outputs in [0, 1] range via Sigmoid activation

### EnhanceNet — Light Enhancer
- **U-Net style** encoder-decoder with skip connections
- Input: Illumination + Reflectance together (so it understands the scene structure)
- Output: A brighter, cleaner illumination map
- Skip connections preserve fine spatial detail

---

## 📊 Features

| Feature | Details |
|---|---|
| 🔬 Decomposition viewer | See Reflectance + Illumination maps |
| 📊 Analytics dashboard | Brightness histogram, contrast, sharpness metrics |
| ↔️ Side-by-side compare | Original vs Enhanced quad view |
| 🎛️ Gamma slider | Fine-tune brightness in real-time |
| ⚡ GPU accelerated | Auto-detects CUDA, falls back to CPU |
| 📁 Batch processing | Enhance entire folders via CLI |
| 💾 One-click download | PNG + JPEG export buttons |

---

## 🚀 Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/ram-ogra/retinexnet.git
cd retinexnet
```

### 2 · Setup environment

```bash
# Windows (Git Bash)
python -m venv venv
source venv/Scripts/activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users** (recommended for speed):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4 · Download weights

```bash
python download_weights.py
```

### 5 · Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** — done! 🎉

---

## 🖥️ CLI Usage

```bash
# Single image
python inference.py --input my_dark_photo.jpg --output result.jpg

# Save decomposition maps too
python inference.py --input dark.jpg --output result.jpg --save_maps

# Batch process a folder
python inference.py --input_dir photos/dark/ --output_dir photos/enhanced/

# Adjust brightness (gamma < 1 = brighter)
python inference.py --input dark.jpg --gamma 0.7

# Force CPU
python inference.py --input dark.jpg --no_gpu
```

---

## 📁 Project Structure

```
retinexnet/
│
├── models/
│   ├── decom_net.py          ← Decomposes image into R & L
│   └── enhance_net.py        ← U-Net that brightens illumination
│
├── utils/
│   └── image_utils.py        ← Image I/O, tensor ops, helpers
│
├── weights/
│   ├── pretrained_decom.pth      ← DecomNet weights
│   ├── pretrained_enhance.pth    ← EnhanceNet weights
│   └── generate_demo_weights.py  ← Offline fallback
│
├── app.py                    ← Streamlit UI
├── inference.py              ← CLI script
├── download_weights.py       ← Weight downloader
├── generate_sample.py        ← Test image generator
└── requirements.txt
```

---

## ⚙️ Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python >= 4.8.0
numpy >= 1.24.0
Pillow >= 10.0.0
streamlit >= 1.32.0
```

---

## 📈 Performance

| Hardware | 512×512 | 1024×1024 |
|---|---|---|
| NVIDIA RTX 3060 | ~18 ms | ~65 ms |
| CPU (i7-12700) | ~320 ms | ~1200 ms |
---

<div align="center">

Designed & Built by **[ram-ogra](https://github.com/ram-ogra)**  
*If this helped you, drop a ⭐ on the repo!*

</div>
