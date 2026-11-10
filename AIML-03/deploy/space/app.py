"""Hugging Face Space entry point for the demo in github.com/ram15x/personalized-recommendation-engine.

At start-up it downloads the public code (branch main) and the trimmed model bundle (branch demo-bundle,
no per-user rating data), then runs demo/app.py from the repository unchanged.
"""
import io
import os
import runpy
import sys
import urllib.request
import zipfile
from pathlib import Path

try:                                    # ZeroGPU Spaces need one @spaces.GPU function; the models run on CPU
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return 0
except ImportError:
    pass

OWNER_REPO = "ram15x/personalized-recommendation-engine"
ROOT = Path(os.environ.get("APP_ROOT", "/tmp/rec_app"))


def fetch(branch: str) -> Path:
    dest = ROOT / branch
    if not dest.exists():
        url = f"https://github.com/{OWNER_REPO}/archive/refs/heads/{branch}.zip"
        with urllib.request.urlopen(url, timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(ROOT / f"_{branch}")
        (inner,) = (ROOT / f"_{branch}").iterdir()
        inner.rename(dest)
    return dest


code, bundle = fetch("main"), fetch("demo-bundle")
os.environ.update(BUNDLE=str(bundle), TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="2")
sys.path.insert(0, str(code / "src"))
demo = runpy.run_path(str(code / "demo" / "app.py"), run_name="space")["demo"]

if __name__ == "__main__":
    demo.launch()
