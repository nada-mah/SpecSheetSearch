import sys
import os

def run_runtime_patches():
    # 1. FIX PATHS FOR BUNDLED ENVIRONMENT
    if hasattr(sys, '_MEIPASS'):
        bundle_dir = sys._MEIPASS
        
        # Add bundle root to search paths for .dylib / .so files (fixes llama-cpp)
        os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', '')
        
        # Fix for doclayout_yolo config files
        os.environ['YOLO_CONFIG_DIR'] = os.path.join(bundle_dir, "doclayout_yolo/cfg")
        os.environ['DOCLAYOUT_YOLO_CFG'] = os.path.join(bundle_dir, "doclayout_yolo/cfg/default.yaml")

    # 2. FORCE-INJECT CRITICAL MODULES
    # This ensures PyInstaller's hidden imports are correctly mapped in sys.modules
    modules_to_ensure = ['shapely', 'pyclipper', 'lanms_neo', 'cv2']
    for mod_name in modules_to_ensure:
        try:
            mod = __import__(mod_name)
            sys.modules[mod_name] = mod
        except ImportError:
            pass

    # 3. MEGA MONKEYPATCH (Bypass PaddleX Dependency Checks)
    # We do this inside a function and catch exceptions to avoid "already initialized" errors
    try:
        import paddlex.utils.deps as paddlex_deps
        
        def mock_true(*args, **kwargs):
            return True
        
        # Override all checker functions
        paddlex_deps.require_extra = mock_true
        paddlex_deps.require_deps = mock_true
        paddlex_deps.require_all_deps = mock_true
        if hasattr(paddlex_deps, 'check_dep'):
            paddlex_deps.check_dep = mock_true
            
        print("🚀 PaddleX Dependency Checks Bypassed")
    except Exception:
        # If it fails, we keep going so the main app can try its own logic
        pass

# Execute the patches
run_runtime_patches()