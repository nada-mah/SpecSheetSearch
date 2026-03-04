#!/usr/bin/env python3
"""
Build script for Key Search (Order Information Extractor)
Creates a small executable by filtering out unnecessary dependencies
"""

import os
import sys
from pathlib import Path
import doclayout_yolo

OUTPUT_NAME = "keySearch"  # <-- New executable name

# Verify we're in the right environment
if not Path("app/main.py").exists():
    print("❌ Error: main.py not found. Please run this script from project root.")
    sys.exit(1)

print("🔍 Analyzing project structure...")

# 1. Ensure models are NOT included in the build
model_files = list(Path("models").rglob("*.pt")) + list(Path("models").rglob("*.gguf"))
if model_files:
    print(f"⚠️ WARNING: {len(model_files)} model files detected in models/ directory.")
    print("   These will NOT be included in the executable (as intended).")
    print("   Users must download them separately and place in ~/.orderextractor/models/")
else:
    print("✅ No model files detected in project directory (good - they should be external)")

# 2. Define data folders to include (ONLY config files, NOT models)
data_folders = [
    ('app/*', 'app')
]
cfg_source = Path(doclayout_yolo.__file__).parent / "cfg"
if cfg_source.exists():
    data_folders.append((str(cfg_source), "doclayout_yolo/cfg"))

config_dir = Path("config")
if config_dir.exists():
    data_folders.append(("config", "config"))

print(f"📦 Including {len(data_folders)} data folders in the build")

# 3. Packages to collect fully
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
    "chardet",    # Added for mypyc modules
    "paddlex",    # Added for dynamic imports
]

# 4. Hidden imports (including mypyc submodules)
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
    "skimage.morphology",
    "skimage.util",
    "skimage.filters",
    "skimage.draw",
    "docx",
    "tiktoken_ext.openai_public",
    "tiktoken_ext",
    # Add chardet mypyc submodules
    "chardet.charsetprober",
    "chardet.universaldetector",
    "chardet.utf8prober",
    # Add paddlex utils if needed
    "paddlex.utils",
]

print(f"⚙️  Using {len(collect_all)} packages with targeted collection")
print(f"⚙️  Using {len(hidden_imports)} hidden imports (minimal set)")

# 5. Build command with aggressive filtering
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
    f"--name={OUTPUT_NAME}",  # <-- Sets executable name
]

# Add data folders
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{os.pathsep}{dest}")

# Add package collections (minimal set)
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Add hidden imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# Main script
cmd_parts.append("app/main.py")

# 6. Run the command
cmd = " ".join(cmd_parts)
print("\n" + "="*80)
print("🚀 Running PyInstaller with the following command:\n")
print(cmd)
print("\n" + "="*80 + "\n")

# 7. Execute the build
print("⏳ Building executable... (this may take 5-10 minutes)")
result = os.system(cmd)

if result == 0:
    exe_path = Path(f"dist/{OUTPUT_NAME}")
    if not exe_path.exists():
        exe_path = Path(f"dist/{OUTPUT_NAME}.exe")

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Build completed successfully!")
        print(f"📦 Executable size: {size_mb:.1f} MB")

        if size_mb > 400:
            print("⚠️  WARNING: Executable is larger than expected (>150MB)")
            print("   Run 'pyinstxtractor dist/{OUTPUT_NAME}' to analyze contents")
        else:
            print("✨ Executable size is optimal (<150MB)")

        print(f"\n👉 Run your app with: dist/{OUTPUT_NAME} --inputs ./data")
        print("💡 Remember: Users must download models separately to ~/.orderextractor/models/")
    else:
        print(f"❌ Build succeeded but executable {OUTPUT_NAME} not found in dist/")
else:
    print(f"❌ Build failed with exit code {result}")
    print("💡 Try running with --log-level=DEBUG for more details")
