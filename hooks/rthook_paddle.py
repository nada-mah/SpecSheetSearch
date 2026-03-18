import sys
import os

# --- THE MONKEYPATCH ---
# This overrides the internal PaddleX dependency checker to always return True
try:
    import paddlex.utils.deps as paddlex_deps
    def mock_require_extra(*args, **kwargs):
        return True
    paddlex_deps.require_extra = mock_require_extra
    print("🚀 PaddleX Dependency Check Bypassed")
except ImportError:
    pass

# Force-inject modules into sys.modules to be safe
try:
    import shapely, pyclipper, lanms_neo
    sys.modules['shapely'] = shapely
    sys.modules['pyclipper'] = pyclipper
    sys.modules['lanms_neo'] = lanms_neo
except ImportError:
    pass

# Standard search path fix
if hasattr(sys, '_MEIPASS'):
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')
import sys
import os

# --- THE MEGA MONKEYPATCH ---
try:
    import paddlex.utils.deps as paddlex_deps
    
    # Mock the 'require_extra' check (fixes the [ocr] error)
    def mock_require_extra(*args, **kwargs):
        return True
    
    # Mock the 'require_deps' check (fixes the opencv-contrib error)
    def mock_require_deps(*args, **kwargs):
        return True

    paddlex_deps.require_extra = mock_require_extra
    paddlex_deps.require_deps = mock_require_deps
    paddlex_deps.require_all_deps = mock_require_deps
    
    print("🚀 PaddleX Dependency Checks Fully Bypassed")
except Exception as e:
    print(f"⚠️ Failed to patch PaddleX: {e}")

# --- Ensure OpenCV is visible ---
try:
    import cv2
    sys.modules['cv2'] = cv2
except ImportError:
    pass

# Standard search path fix for PyInstaller
if hasattr(sys, '_MEIPASS'):
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')
    
    import sys
import os

# --- SAFER MONKEYPATCH ---
# We inject these into sys.modules BEFORE anything else imports them
# This avoids triggering the 'PDX has already been initialized' error.

def bypass_paddlex_checks():
    try:
        # Check if paddlex is in sys.modules; if not, we can safely mock the utils
        # without triggering the full package initialization.
        import paddlex.utils.deps as paddlex_deps
        
        def mock_true(*args, **kwargs):
            return True
        
        paddlex_deps.require_extra = mock_true
        paddlex_deps.require_deps = mock_true
        paddlex_deps.require_all_deps = mock_true
        # Also patch the internal check used by PDFReader
        if hasattr(paddlex_deps, 'check_dep'):
            paddlex_deps.check_dep = mock_true
            
        print("🚀 PaddleX Dependency Checks Safely Bypassed")
    except Exception:
        # If it fails here, the main app will try to import it anyway
        pass

bypass_paddlex_checks()

# Fix for pathing
if hasattr(sys, '_MEIPASS'):
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')