"""
00: process
    process/.venv/bin/python -m process.cli --video-path inputs/0.mp4
01: segment
    segment/.venv/bin/python -m segment.cli --video.frames-json outputs/0/process/frames.json --video.vis
02: depth
    depth/.venv/bin/python -m depth.cli --command video --video.frames-json outputs/0/process/frames.json --video.vis
03: inpaint
04: retarget
"""
