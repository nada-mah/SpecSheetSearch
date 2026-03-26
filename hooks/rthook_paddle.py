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
    # Add the bundle directory to the library search path for macOS
    os.environ['DYLD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', '')
    
    # Also add to PATH just in case
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LLAMA_CPP_LIB'] = os.path.join(bundle_dir, "llama_cpp")
    os.environ['LD_LIBRARY_PATH'] = sys._MEIPASS + ":" + os.environ.get('LD_LIBRARY_PATH', '')
    
    # CRITICAL: Disable oneDNN. This kills the buggy CPU path you saw in production.
    os.environ['FLAGS_use_onednn'] = '0'
    
    # Optional: Force Paddle to use GPU 0 if available
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
    

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