"""Import sherpa-onnx with a compatible onnxruntime DLL loaded first (Windows).

The sherpa-onnx wheel's extension links against `onnxruntime.dll` but this environment does not install
`sherpa-onnx-core`, which normally ships it. Windows then resolves the name to
C:\\Windows\\System32\\onnxruntime.dll (ORT 1.17, part of Windows ML), which lacks the C API version
sherpa-onnx requests and the process segfaults on the first model load. Loading the onnxruntime
package's own DLL by full path first makes the loader reuse it for sherpa-onnx.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

_lock = threading.Lock()
_module: ModuleType | None = None


def _preload_onnxruntime_dll() -> None:
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise ImportError("onnxruntime is required to run sherpa-onnx models on Windows but is not installed")
    dll = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
    if not dll.is_file():
        raise ImportError(f"onnxruntime.dll not found at {dll}; reinstall the onnxruntime package")
    ctypes.WinDLL(str(dll))


def import_sherpa() -> ModuleType:
    global _module
    with _lock:
        if _module is None:
            if sys.platform == "win32":
                _preload_onnxruntime_dll()  # only effective if sherpa_onnx was not imported earlier
            import sherpa_onnx

            _module = sherpa_onnx
        return _module
