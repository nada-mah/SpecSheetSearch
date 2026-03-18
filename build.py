#!/usr/bin/env python3
"""
Build script for keySearch CLI
Handles hidden imports, dynamic libraries, llama_cpp, and PaddleX OCR dependencies.
"""

import os
import sys
import subprocess
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

# 2️⃣ Packages to collect fully (Data + Binaries + Py)
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

# 3️⃣ Hidden imports (Force include modules PyInstaller might miss)
hidden_imports = [
    "huggingface_hub",
    "llama_cpp",
    "paddlex.inference.models.ocr",
    "paddlex.inference.pipelines.ocr",
    "shapely",
    "pyclipper",
    "lanms_neo", # Critical for OCR text detection
    "paddle.base.libpaddle",
    "paddle.utils.cpp_extension",
]

# 4️⃣ Initialize PyInstaller command
cmd_parts = [
    "pyinstaller",
    "--onefile",
    "--console",
    "--clean",
    "--distpath=dist",
    "--workpath=build",
    "--specpath=.",
    "--copy-metadata=paddlex",
    "--copy-metadata=paddleocr",
    "--copy-metadata=shapely",
    "--copy-metadata=pyclipper",
    "--copy-metadata=lanms-neo",
    # Force include these hidden imports again to be safe
    "--hidden-import=shapely",
    "--hidden-import=pyclipper",
    "--hidden-import=lanms_neo",
    "--hidden-import=paddlex.inference.models.ocr",
    "--hidden-import=paddlex.inference.pipelines.ocr"
    # Note: --strip can break code signatures on macOS ARM64; 
    # remove it if you get 'Signature Invalid' errors.
    "--strip", 
    f"--name={OUTPUT_NAME}",
]

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

# 🔥 Force include paddle and llama_cpp directories to ensure .so/.dylib are caught
paddle_dir = os.path.dirname(paddle.__file__)
cmd_parts.append(f"--add-data={paddle_dir}{SEP}paddle")

llama_dir = os.path.dirname(llama_cpp.__file__)
cmd_parts.append(f"--add-data={llama_dir}{SEP}llama_cpp")

# Force include specific paddlex data
cmd_parts.append("--collect-data=paddlex")

# 5️⃣ The Main Script (Must be the last positional argument)
cmd_parts.append("app/main.py")

# Execute Build
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller:\n")
print(cmd)
print("\n" + "="*80 + "\n")

# Use subprocess for better error handling than os.system
result = subprocess.run(cmd, shell=True)

# Check executable
exe_path = Path(f"dist/{OUTPUT_NAME}")
if sys.platform == "win32":
    exe_path = exe_path.with_suffix(".exe")

if result.returncode == 0 and exe_path.exists():
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Build completed successfully! Size: {size_mb:.1f} MB")
    print(f"👉 Run: ./dist/{OUTPUT_NAME} --input ./data/new_pdfs --schema ./schema/test_schema.json")
else:
    print(f"❌ Build failed with exit code {result.returncode}.")