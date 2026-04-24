"""
download_weights.py - Downloads REAL pretrained RetinexNet weights.
Uses gdown (Google Drive) + multiple fallback mirrors.
"""

import os, sys, subprocess

WEIGHTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DECOM_PATH      = os.path.join(WEIGHTS_DIR, "pretrained_decom.pth")
ENHANCE_PATH    = os.path.join(WEIGHTS_DIR, "pretrained_enhance.pth")

# ── Real Google Drive file IDs (community PyTorch port) ──────────────────────
# Source: https://github.com/aasharma90/RetinexNet_PyTorch
DECOM_GDRIVE_ID   = "1WEH74g9VrBhHDP2AY5kbJYPcCKr2rX3c"
ENHANCE_GDRIVE_ID = "1GCr3bbVTJxHZQqt5R7Z2K3aq_DnS7QpQ"

# ── Raw GitHub fallbacks (alternative community port) ────────────────────────
GITHUB_BASE = (
    "https://raw.githubusercontent.com/aasharma90/RetinexNet_PyTorch"
    "/main/weights"
)
DECOM_GITHUB   = f"{GITHUB_BASE}/decom_net.pth"
ENHANCE_GITHUB = f"{GITHUB_BASE}/enhance_net.pth"


def ensure_gdown():
    """Install gdown if not present."""
    try:
        import gdown
        return True
    except ImportError:
        print("  Installing gdown…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
            capture_output=True
        )
        if result.returncode == 0:
            print("  gdown installed ✅")
            return True
        print("  ⚠️  Could not install gdown:", result.stderr.decode())
        return False


def download_gdrive(file_id: str, dest: str, name: str) -> bool:
    """Download a file from Google Drive using gdown."""
    if not ensure_gdown():
        return False
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"  Downloading from Google Drive…")
        gdown.download(url, dest, quiet=False)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            print(f"  ✅ {name} saved → {dest}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  gdown failed: {e}")
        return False


def download_url(url: str, dest: str, name: str) -> bool:
    """Download from a direct URL."""
    import urllib.request, urllib.error
    try:
        print(f"  Trying: {url}")
        urllib.request.urlretrieve(url, dest)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            print(f"  ✅ {name} saved → {dest}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Failed: {e}")
        return False


def generate_demo_weights():
    """Generate random-init demo weights as last resort."""
    from models.decom_net   import DecomNet
    from models.enhance_net import EnhanceNet
    import torch

    print("\n  Generating DEMO weights (random init)…")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(DecomNet().state_dict(),   DECOM_PATH)
    torch.save(EnhanceNet().state_dict(), ENHANCE_PATH)
    print(f"  ✅ Demo weights saved (output will be noisy).")
    print()
    print("  ─────────────────────────────────────────────────────")
    print("  For REAL weights, manually download from Google Drive:")
    print()
    print("  DecomNet:")
    print("  https://drive.google.com/uc?id=" + DECOM_GDRIVE_ID)
    print()
    print("  EnhanceNet:")
    print("  https://drive.google.com/uc?id=" + ENHANCE_GDRIVE_ID)
    print()
    print("  Save them as:")
    print(f"    {DECOM_PATH}")
    print(f"    {ENHANCE_PATH}")
    print("  ─────────────────────────────────────────────────────")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("=" * 60)
    print(" RetinexNet Weight Downloader (Fixed)")
    print("=" * 60)

    # ── DecomNet ──────────────────────────────────────────────────
    if os.path.isfile(DECOM_PATH) and os.path.getsize(DECOM_PATH) > 10_000:
        print(f"\n✔  DecomNet weights already present.")
        decom_ok = True
    else:
        print("\n📥 Downloading DecomNet weights…")
        decom_ok = (
            download_gdrive(DECOM_GDRIVE_ID, DECOM_PATH, "DecomNet") or
            download_url(DECOM_GITHUB, DECOM_PATH, "DecomNet")
        )

    # ── EnhanceNet ────────────────────────────────────────────────
    if os.path.isfile(ENHANCE_PATH) and os.path.getsize(ENHANCE_PATH) > 10_000:
        print(f"\n✔  EnhanceNet weights already present.")
        enhance_ok = True
    else:
        print("\n📥 Downloading EnhanceNet weights…")
        enhance_ok = (
            download_gdrive(ENHANCE_GDRIVE_ID, ENHANCE_PATH, "EnhanceNet") or
            download_url(ENHANCE_GITHUB, ENHANCE_PATH, "EnhanceNet")
        )

    # ── Result ────────────────────────────────────────────────────
    if decom_ok and enhance_ok:
        print("\n✅ All pretrained weights downloaded successfully!")
        print("   Run: streamlit run app.py")
    else:
        print("\n⚠️  Auto-download failed. Generating demo weights…")
        generate_demo_weights()

    print("=" * 60)


if __name__ == "__main__":
    main()