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