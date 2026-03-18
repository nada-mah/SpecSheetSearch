import os
import sys

if hasattr(sys, "_MEIPASS"):
    base = sys._MEIPASS

    # Paddle libs
    paddle_lib = os.path.join(base, "paddle", "libs")
    if os.path.exists(paddle_lib):
        os.environ["DYLD_LIBRARY_PATH"] = paddle_lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = paddle_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")