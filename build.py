#!/usr/bin/env python3
"""
Build script for keySearch CLI
Handles hidden imports, dynamic libraries, llama_cpp, and PaddleX OCR dependencies.
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
# macOS uses ':', Windows uses ';'
SEP = os.pathsep 

# Verify main script exists
if not Path("app/main.py").exists():
    print("❌ main.py not found. Run this script from project root.")
    sys.exit(1)

# 1️⃣ Data folders
data_folders = [("app/*", "app")]
cfg_source = Path(doclayout_yolo.__file__).parent / "cfg"
if cfg_source.exists():
    data_folders.append((str(cfg_source), "doclayout_yolo/cfg"))

# 2️⃣ Packages to collect fully
collect_all = [
    "paddleocr",
    "paddle",
    "paddlex",
    "doclayout_yolo",
    "pypdfium2",
    "PyMuPDF",
    "langchain",
    "tqdm",
    "regex",
    "PyPDF2",
    "dill",
    "hf_xet",
    "tiktoken",
    "pillow",
]

# 3️⃣ Hidden imports (Module names)
hidden_imports = [
    "huggingface_hub",
    "llama_cpp",
    "paddlex.inference.models.ocr",
    "paddlex.inference.pipelines.ocr",
    "shapely",
    "pyclipper",
    "lanms_neo", 
    "paddle.base.libpaddle",
    "paddle.utils.cpp_extension",
]

# 4️⃣ Metadata to copy (Package names)
# We check these safely so PyInstaller doesn't crash if one is missing.
metadata_to_copy = [
    "paddlex",
    "paddleocr",
    "shapely",
    "pyclipper",
    "lanms-neo"
]

# Initialize PyInstaller command
cmd_parts = [
    "pyinstaller",
    "--onefile",
    "--console",
    "--clean",
    "--distpath=dist",
    "--workpath=build",
    "--specpath=.",
    # --strip can break code signatures on macOS ARM64; remove if getting 'Killed: 9'
    "--strip", 
    f"--name={OUTPUT_NAME}",
]

# --- SAFE METADATA COLLECTION ---
for pkg in metadata_to_copy:
    try:
        importlib.metadata.distribution(pkg)
        cmd_parts.append(f"--copy-metadata={pkg}")
    except importlib.metadata.PackageNotFoundError:
        print(f"⚠️ Warning: Metadata for package '{pkg}' not found. Skipping.")

# Add runtime hooks and additional hooks
if Path("hooks/rthook_paddle.py").exists():
    cmd_parts.append("--runtime-hook=hooks/rthook_paddle.py")
if Path("hooks").exists():
    cmd_parts.append("--additional-hooks-dir=hooks")

# Add data folders
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{SEP}{dest}")

# Add full package collections
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Add hidden imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# Force include binary directories
paddle_dir = os.path.dirname(paddle.__file__)
cmd_parts.append(f"--add-data={paddle_dir}{SEP}paddle")

llama_dir = os.path.dirname(llama_cpp.__file__)
cmd_parts.append(f"--add-data={llama_dir}{SEP}llama_cpp")

cmd_parts.append("--collect-data=paddlex")

# 5️⃣ The Main Script (Last argument)
cmd_parts.append("app/main.py")

# Execute Build
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller:\n")
print(cmd)
print("\n" + "="*80 + "\n")

result = subprocess.run(cmd, shell=True)

# Check result
exe_path = Path(f"dist/{OUTPUT_NAME}")
if sys.platform == "win32":
    exe_path = exe_path.with_suffix(".exe")

if result.returncode == 0 and exe_path.exists():
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Build completed successfully! Size: {size_mb:.1f} MB")
else:
    print(f"❌ Build failed with exit code {result.returncode}.")