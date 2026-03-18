#!/usr/bin/env python3
"""
Build script for keySearch CLI
Handles hidden imports, dynamic libraries, llama_cpp, and chardet mypyc issue.
"""

import os
import sys
from pathlib import Path
import doclayout_yolo
import llama_cpp  # needed for PyInstaller binary inclusion

OUTPUT_NAME = "keySearch"

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

# 3️⃣ Hidden imports
hidden_imports = [
    "huggingface_hub",
    "llama_cpp",
]

# 4️⃣ PyInstaller command
cmd_parts = [
    "pyinstaller",
    "--onefile",
    "--console",
    "--clean",
    "--distpath=dist",
    "--workpath=build",
    "--specpath=.",
    "--strip",

    # 🔥 CRITICAL FIXES
    "--runtime-hook=hooks/rthook_paddle.py",
    "--hidden-import=paddle.base.libpaddle",
    "--hidden-import=paddle.utils.cpp_extension",
    "--hidden-import=paddlex",

    "--additional-hooks-dir=hooks",

    f"--name={OUTPUT_NAME}",
]

# Add data folders
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{os.pathsep}{dest}")

# Add packages
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Add hidden imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# 🔥 Force include paddle libs directory
import paddle
paddle_dir = os.path.dirname(paddle.__file__)
cmd_parts.append(f"--add-data={paddle_dir}{os.pathsep}paddle")

# Add llama_cpp binaries manually
llama_dir = os.path.dirname(llama_cpp.__file__)
cmd_parts.append(f"--add-data={llama_dir}{os.pathsep}llama_cpp")

# Main script
cmd_parts.append("app/main.py")
cmd_parts.append("--collect-data=paddlex")
# Build
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller:\n")
print(cmd)
print("\n" + "="*80 + "\n")

result = os.system(cmd)

# Check executable
exe_path = Path(f"dist/{OUTPUT_NAME}")
if not exe_path.exists():
    exe_path = Path(f"dist/{OUTPUT_NAME}.exe")

if result == 0 and exe_path.exists():
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Build completed successfully! Size: {size_mb:.1f} MB")
    print(f"👉 Run: dist/{OUTPUT_NAME} --inputs ./data")
else:
    print(f"❌ Build failed with exit code {result} or executable not found.")
