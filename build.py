#!/usr/bin/env python3
"""
Build script for Key Search CLI (Order Information Extractor)
✅ Handles hidden imports, dynamic libraries, and llama_cpp for PyInstaller on macOS ARM.
"""

import os
import sys
from pathlib import Path
import doclayout_yolo
import llama_cpp  # Import to get folder location

OUTPUT_NAME = "keySearch"

# 1️⃣ Ensure main.py exists
if not Path("app/main.py").exists():
    print("❌ main.py not found. Run this script from project root.")
    sys.exit(1)

print("🔍 Analyzing project structure...")

# 2️⃣ Data folders (configs only)
data_folders = [("app/*", "app")]
cfg_source = Path(doclayout_yolo.__file__).parent / "cfg"
if cfg_source.exists():
    data_folders.append((str(cfg_source), "doclayout_yolo/cfg"))

# 3️⃣ Packages to fully collect
collect_all = [
    "paddleocr",
    "paddle",
    "doclayout_yolo",
    "pypdfium2",
    "skimage",
    "docx",
    "tiktoken",
    "paddlex",
    "PyMuPDF",
    "python_Levenshtein",
]

# 4️⃣ Hidden imports
hidden_imports = [
    "huggingface_hub",
    "pyclipper",
    "regex",
    "fuzzywuzzy",
    "tiktoken",
    "dill",
    "numpy",
    "scipy",
    "PIL",
    "langchain",
    "hf_xet",
    # doclayout_yolo nested modules
    "doclayout_yolo.nn.modules.modeling",
    "doclayout_yolo.nn.modules.modeling.backbone",
    # paddlex optional serving plugin
    "paddlex.inference.serving",
    # llama_cpp
    "llama_cpp",
]

# 5️⃣ PyInstaller command
cmd_parts = [
    "pyinstaller",
    "--onefile",
    "--console",
    "--clean",
    "--distpath=dist",
    "--workpath=build",
    "--specpath=.",
    "--strip",
    "--upx-dir=/usr/bin",
    "--upx-exclude=libz.so",
    "--upx-exclude=liblzma.so",
    "--upx-exclude=libssl.so",
    "--upx-exclude=libcrypto.so",
    f"--name={OUTPUT_NAME}",
]

# Add data folders
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{os.pathsep}{dest}")

# Collect all submodules
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Hidden imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# 6️⃣ Include llama_cpp manually (binary module)
llama_dir = os.path.dirname(llama_cpp.__file__)
cmd_parts.append(f"--add-data={llama_dir}{os.pathsep}llama_cpp")

# 7️⃣ Main script
cmd_parts.append("app/main.py")

# 8️⃣ Execute build
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller command:\n")
print(cmd)
print("\n" + "="*80 + "\n")

result = os.system(cmd)

# 9️⃣ Check executable
exe_path = Path(f"dist/{OUTPUT_NAME}")
if not exe_path.exists():
    exe_path = Path(f"dist/{OUTPUT_NAME}.exe")

if result == 0 and exe_path.exists():
    size_mb = exe_path.stat().st_size / (1024*1024)
    print(f"\n✅ Build completed successfully! Size: {size_mb:.1f} MB")
    print(f"👉 Run: dist/{OUTPUT_NAME} --inputs ./data")
else:
    print(f"❌ Build failed with exit code {result} or executable not found.")
