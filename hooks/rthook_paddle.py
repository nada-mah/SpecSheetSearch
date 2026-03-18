import sys
import os

# 1. Force-inject OCR dependencies into sys.modules so PaddleX thinks they are installed
try:
    import shapely
    import pyclipper
    # paddleocr/paddlex often look for these specifically
    sys.modules['shapely'] = shapely
    sys.modules['pyclipper'] = pyclipper
except ImportError:
    pass

# 2. Fix the Paddle Search Path for the .so files
if hasattr(sys, '_MEIPASS'):
    paddle_libs = os.path.join(sys._MEIPASS, 'paddle', 'libs')
    if os.path.exists(paddle_libs):
        os.environ['PATH'] = paddle_libs + os.pathsep + os.environ.get('PATH', '')