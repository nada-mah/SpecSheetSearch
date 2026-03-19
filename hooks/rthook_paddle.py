import sys
import os

if hasattr(sys, '_MEIPASS'):
    bundle_dir = sys._MEIPASS
    
    # This is the directory PyInstaller creates for your data_folders
    # Mapping doclayout_path -> 'doclayout_yolo' in build.py results in this:
    yolo_cfg_dir = os.path.join(bundle_dir, "doclayout_yolo", "cfg")
    
    # Force the library to look HERE for its YAML files
    os.environ['YOLO_CONFIG_DIR'] = yolo_cfg_dir
    # Some versions use this specific env var for the default config
    os.environ['ULTRALYTICS_CONFIG_DIR'] = yolo_cfg_dir

# --- The Monkeypatch to bypass PaddleX checks ---
try:
    import paddlex.utils.deps as paddlex_deps
    def mock_true(*args, **kwargs): return True
    paddlex_deps.require_extra = mock_true
    paddlex_deps.require_deps = mock_true
    paddlex_deps.require_all_deps = mock_true
    print("🚀 PaddleX Dependency Checks Bypassed")
except Exception:
    pass