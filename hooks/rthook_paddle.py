import sys
import os

# Force-inject OCR dependencies into sys.modules
# This prevents PaddleX from raising DependencyError
try:
    import shapely
    import pyclipper
    import lanms_neo
    sys.modules['shapely'] = shapely
    sys.modules['pyclipper'] = pyclipper
    sys.modules['lanms_neo'] = lanms_neo
except ImportError:
    pass

# Ensure the bundled libraries are in the search path
if hasattr(sys, '_MEIPASS'):
    # Fix for macOS/Linux .so loading
    os.environ['LD_LIBRARY_PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    # Fix for Windows .dll loading
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')