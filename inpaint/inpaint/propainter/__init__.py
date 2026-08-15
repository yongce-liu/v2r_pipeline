"""ProPainter (streaming) video-inpainting backend for masked frames.

Mirrors ``inpaint.lama`` and ``inpaint.qwen``: an inpainter driving the
third-party ``propainter`` pip package (osmr streaming reimplementation built on
pytorchcv), its argument dataclass (``args.py``), and the video-mode workflow
(``workflow.py``). Unlike the per-frame backends, ProPainter inpaits a whole
video at once (RAFT + flow completion + transformer), so it only supports
``--command video``.
"""

__version__ = "0.1.0"
