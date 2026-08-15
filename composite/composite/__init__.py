"""Depth-aware compositing of robot render frames into the inpainted scene.

The composite step is the final stage of the v2r pipeline: it takes the
robot-arm RGB+depth renders from ``retarget`` and the inpainted background plus
its estimated depth from ``inpaint``/``depth``, and pastes the arm in wherever
it is closer to the camera than the scene surface behind it.
"""

__version__ = "0.1.0"
