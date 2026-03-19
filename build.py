#!/usr/bin/env python3
"""
Final Build Script for keySearch CLI
Fixes: 
1. PaddleX OCR DependencyError (Metadata + Hidden Imports)
2. PyInstaller crash on missing metadata
3. macOS binary path resolution
"""

import os
import sys
import subprocess
import importlib.metadata
from pathlib import Path
import doclayout_yolo
import llama_cpp
import paddle

OUTPUT_NAME = "keySearch"
SEP = os.pathsep 

# 1️⃣ Verification
if not Path("app/main.py").exists():
    print("❌ main.py not found. Run from project root.")
    sys.exit(1)

# 2️⃣ Data Folders & Packages
data_folders = [("app/*", "app")]
cfg_source = Path(doclayout_yolo.__file__).parent / "cfg"
if cfg_source.exists():
    data_folders.append((str(cfg_source), "doclayout_yolo/cfg"))

collect_all = [
    "paddleocr", "paddle", "paddlex", "doclayout_yolo",
    "pypdfium2", "PyMuPDF", "langchain", "tqdm", "regex",
    "PyPDF2", "dill", "hf_xet", "tiktoken", "pillow",
     "cv2", "opencv-contrib-python"
]

# 3️⃣ Metadata & Hidden Imports (The OCR Fix)
# Package names for metadata (often have hyphens)
metadata_to_copy = [ "paddlex", "paddleocr", 
                    "opencv-contrib-python", "shapely", "pyclipper", "lanms-neo"]

# Module names for imports (must use underscores)
hidden_imports = [
    "cv2", "paddlex.inference.utils.io.readers",
    "huggingface_hub", "llama_cpp", "shapely", "pyclipper", "lanms_neo",
    "paddlex.inference.models.ocr", "paddlex.inference.pipelines.ocr",
    "paddle.base.libpaddle", "paddle.utils.cpp_extension"
]

# 4️⃣ Construct Command
cmd_parts = [
    "pyinstaller", "--onefile", "--console", "--clean",
    "--distpath=dist", "--workpath=build", "--specpath=.",
    "--name=" + OUTPUT_NAME
]

# --- Add Metadata Safely ---
for pkg in metadata_to_copy:
    try:
        importlib.metadata.distribution(pkg)
        cmd_parts.append(f"--copy-metadata={pkg}")
    except importlib.metadata.PackageNotFoundError:
        print(f"⚠️ Metadata for '{pkg}' not found. Skipping metadata copy.")

# --- Add Hidden Imports ---
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# --- Add Full Collections ---
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# --- Add Runtime Hooks ---
if Path("hooks/rthook_paddle.py").exists():
    cmd_parts.append("--runtime-hook=hooks/rthook_paddle.py")
if Path("hooks").exists():
    cmd_parts.append("--additional-hooks-dir=hooks")

# --- Add Data Paths ---
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{SEP}{dest}")

# Force include binary directories
cmd_parts.append(f"--add-data={os.path.dirname(paddle.__file__)}{SEP}paddle")
cmd_parts.append(f"--add-data={os.path.dirname(llama_cpp.__file__)}{SEP}llama_cpp")
llama_path = os.path.dirname(llama_cpp.__file__)

# Add this to your PyInstaller command parts
cmd_parts.append(f"--add-data={llama_path}{os.pathsep}llama_cpp")
cmd_parts.append("--collect-data=paddlex")


# Get the absolute path to the installed package
doclayout_path = os.path.dirname(doclayout_yolo.__file__)

# Add the 'cfg' folder to your data_folders
# We map it so it lands in 'doclayout_yolo/cfg' inside the bundle
data_folders = [
    ("app/*", "app"),
    (os.path.join(doclayout_path, "cfg", "*"), "doclayout_yolo/cfg"),
    # Add this if there are other yaml files in the root of the package
    (os.path.join(doclayout_path, "*.yaml"), "doclayout_yolo"), 
]

# Ensure you use these in your pyinstaller command loop
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{os.pathsep}{dest}")

import importlib.metadata

# Packages PaddleX checks for the "OCR" extra
metadata_to_bundle = ["paddlex", "paddleocr", "shapely", "pyclipper", "lanms-neo"]

for pkg in metadata_to_bundle:
    try:
        # Verify the package exists in the CI environment
        importlib.metadata.distribution(pkg)
        cmd_parts.append(f"--copy-metadata={pkg}")
    except importlib.metadata.PackageNotFoundError:
        print(f"⚠️ Metadata for '{pkg}' not found. Skipping.")

# Ensure modules are hidden imports (using underscores, not hyphens)
cmd_parts.extend([
    "--hidden-import=shapely",
    "--hidden-import=pyclipper",
    "--hidden-import=lanms_neo",
    "--hidden-import=paddlex.inference.models.ocr",
    "--hidden-import=paddlex.inference.pipelines.ocr"
])
# Final Script Path
cmd_parts.append("app/main.py")

# 5️⃣ Execution
full_cmd = " ".join(cmd_parts)
print("\n🚀 Executing PyInstaller Build...\n")
result = subprocess.run(full_cmd, shell=True)

if result.returncode == 0:
    print(f"\n✅ Build Successful! Check dist/{OUTPUT_NAME}")
else:
    print(f"\n❌ Build Failed with code {result.returncode}")