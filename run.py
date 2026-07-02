# ─── ONNX int32 → int64 Compatibility Patch ────────────────────────────────
#
# Problem:
#   pymupdf4llm uses ONNX models internally for image embedding and table
#   detection. These models expect int64 tensors, but on Windows, numpy
#   defaults to int32 — causing an ONNXRuntimeError: INVALID_ARGUMENT.
#
# Solution:
#   Monkey-patch numpy's array creation functions and onnxruntime's
#   InferenceSession.run to silently upcast int32 → int64 at two levels:
#
#   Level 1 — np.array / np.asarray:
#       Intercepts tensor creation before it reaches ONNX.
#
#   Level 2 — ort.InferenceSession.run:
#       Last line of defence — catches any int32 tensors that slipped
#       through Level 1 just before the model runs.
#
# Applied in run.py before all other imports so the patch is in place
# before pymupdf4llm or onnxruntime are imported anywhere in the app.
#
# Remove this patch if:
#   - pymupdf4llm fixes the dtype issue in a future release
#   - Migrating away from Windows to Linux (int64 is default there)
# ────────────────────────────────────────────────────────────────────────────
import numpy as np

# Patch int32 to int64 for ONNX compatibility
_original_asarray = np.asarray
_original_array = np.array


def _patched_asarray(a, dtype=None, **kwargs):
    result = _original_asarray(a, dtype=dtype, **kwargs)
    if result.dtype == np.int32:
        return result.astype(np.int64)
    return result


def _patched_array(a, dtype=None, **kwargs):
    result = _original_array(a, dtype=dtype, **kwargs)
    if result.dtype == np.int32:
        return result.astype(np.int64)
    return result


np.asarray = _patched_asarray
np.array = _patched_array

import onnxruntime as ort

_original_run = ort.InferenceSession.run


def _patched_run(self, output_names, input_feed, **kwargs):
    # cast all int32 inputs to int64
    patched_feed = {
        k: v.astype(np.int64) if hasattr(v, "dtype") and v.dtype == np.int32 else v
        for k, v in input_feed.items()
    }
    return _original_run(self, output_names, patched_feed, **kwargs)


ort.InferenceSession.run = _patched_run
"""
CyberRAG Entry Point.

Run the server with:
    python run.py

Or with uvicorn directly:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
