#!/usr/bin/env python3
"""
Robust build script for Key Search (Order Information Extractor)
Includes proper hooks for mypyc-compiled modules (chardet, paddlex) and ensures dynamic libs are bundled.
"""

import os
import sys
from pathlib import Path
import doclayout_yolo

OUTPUT_NAME = "keySearch"

# Check that main.py exists
if not Path("app/main.py").exists():
    print("❌ Error: main.py not found. Run from project root.")
    sys.exit(1)

print("🔍 Analyzing project structure...")

# Ensure models are NOT included in the build
model_files = list(Path("models").rglob("*.pt")) + list(Path("models").rglob("*.gguf"))
if model_files:
    print(f"⚠️ WARNING: {len(model_files)} model files detected in models/")
    print("   These will NOT be included in the executable. Users must download separately.")
else:
    print("✅ No model files detected (good)")

# Data folders (config files only)
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

print(f"⚙️  Collecting {len(collect_all)} packages")
print(f"⚙️  Using {len(hidden_imports)} hidden imports")

# Ensure hooks folder exists for dynamic libs
hooks_dir = Path("hooks")
hooks_dir.mkdir(exist_ok=True)

# Hook for chardet
(chardet_hook := hooks_dir / "hook-chardet.py").write_text("""\
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs
hiddenimports = collect_submodules('chardet')
binaries = collect_dynamic_libs('chardet')
""")

# Hook for paddlex
(paddlex_hook := hooks_dir / "hook-paddlex.py").write_text("""\
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs
hiddenimports = collect_submodules('paddlex')
binaries = collect_dynamic_libs('paddlex')
""")

# Build command
cmd_parts = [
    "pyinstaller",
    "--onefile",
    "--console",
    "--clean",
    f"--distpath=dist",
    "--workpath=build",
    "--specpath=.",
    "--strip",
    "--upx-dir=/usr/bin",
    "--upx-exclude=libz.so",
    "--upx-exclude=liblzma.so",
    "--upx-exclude=libssl.so",
    "--upx-exclude=libcrypto.so",
    f"--additional-hooks-dir={hooks_dir}",
    f"--name={OUTPUT_NAME}",
]

# Add data folders
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{os.pathsep}{dest}")

# Collect all submodules for critical packages
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Add hidden imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# Main script
cmd_parts.append("app/main.py")

# Execute command
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller with the following command:\n")
print(cmd)
print("\n" + "="*80 + "\n")

print("⏳ Building executable... (may take 5-10 min)")
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
    print("💡 Try running with --log-level=DEBUG for details.")
