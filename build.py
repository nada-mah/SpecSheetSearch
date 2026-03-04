#!/usr/bin/env python3
"""
Robust build script for Key Search (Order Information Extractor)
Automatically includes mypyc-compiled modules from chardet and paddlex.
"""

import os
import sys
from pathlib import Path
import glob
import doclayout_yolo

OUTPUT_NAME = "keySearch"

# Verify we're in the project root
if not Path("app/main.py").exists():
    print("❌ Error: main.py not found. Run from project root.")
    sys.exit(1)

print("🔍 Analyzing project structure...")

# Ensure models are NOT included in the build
model_files = list(Path("models").rglob("*.pt")) + list(Path("models").rglob("*.gguf"))
if model_files:
    print(f"⚠️ WARNING: {len(model_files)} model files detected in models/ directory. They will not be included.")
else:
    print("✅ No model files detected (good)")

# Data folders (configs only)
data_folders = [
    ('app/*', 'app')
]
cfg_source = Path(doclayout_yolo.__file__).parent / "cfg"
if cfg_source.exists():
    data_folders.append((str(cfg_source), "doclayout_yolo/cfg"))

print(f"📦 Including {len(data_folders)} data folders")

# Packages to collect fully
collect_all = [
    "paddleocr",
    "paddle",
    "Cython",
    "doclayout_yolo",
    "llama_cpp",
    "pypdfium2",
    "skimage",
    "docx",
    "tiktoken",
    "chardet",
    "paddlex",
]

# Minimal hidden imports
hidden_imports = [
    "huggingface_hub",
    "pyclipper",
    "regex",
    "fuzzywuzzy",
    "python_Levenshtein",
    "tiktoken",
    "dill",
    "numpy",
    "scipy",
    "PIL",
    "skimage",
]

# Build command
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

# --- Automatically detect mypyc .so files for chardet and paddlex ---
venv_site = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

def add_mypyc_binaries(package_name):
    pkg_path = venv_site / package_name
    if pkg_path.exists():
        so_files = glob.glob(str(pkg_path / "*__mypyc*.so"))
        for so_file in so_files:
            print(f"🔹 Adding mypyc binary: {so_file}")
            cmd_parts.append(f"--add-binary={so_file}:{package_name}")

for pkg in ["chardet", "paddlex"]:
    add_mypyc_binaries(pkg)

# Main script
cmd_parts.append("app/main.py")

# Execute
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller command:\n")
print(cmd)
print("\n" + "="*80 + "\n")

print("⏳ Building executable... (may take 5-10 minutes)")
result = os.system(cmd)

# Check result
exe_path = Path(f"dist/{OUTPUT_NAME}")
if not exe_path.exists():
    exe_path = Path(f"dist/{OUTPUT_NAME}.exe")

if result == 0 and exe_path.exists():
    size_mb = exe_path.stat().st_size / (1024*1024)
    print(f"\n✅ Build completed successfully! Size: {size_mb:.1f} MB")
    print(f"👉 Run: dist/{OUTPUT_NAME} --inputs ./data")
    print("💡 Remember: models must be downloaded separately to ~/.orderextractor/models/")
else:
    print(f"❌ Build failed with exit code {result} or executable not found.")
    print("💡 Try running with --log-level=DEBUG for details")
