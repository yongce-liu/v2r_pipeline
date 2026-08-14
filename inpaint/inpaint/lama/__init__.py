"""LaMa (big-lama) inpainting backend for masked frames.

The ``lama_inpaint`` subpackage drives the third-party ``simple-lama-inpainting``
package (a TorchScript big-lama FFC wrapper) behind the same
``inpaint(image, output_path, args, mask=...)`` interface as
``inpaint.qwen.inpainter.QwenInpainter``, so the video workflow can drive either
backend. See ``inpaint.lama_inpaint.simple`` for the inpainter and
``inpaint.lama_inpaint.workflow`` for the video mode.
"""

__version__ = "0.1.0"
