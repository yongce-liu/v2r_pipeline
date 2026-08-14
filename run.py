"""
00: process
    process/.venv/bin/python -m process.cli --video-path inputs/0.mp4
01: segment
    segment/.venv/bin/python -m segment.cli --video.frames-json outputs/0/process/frames.json --video.vis
02: depth
    depth/.venv/bin/python -m depth.cli --video.frames-json outputs/0/process/frames.json --video.vis
03: inpaint
    inpaint/.venv/bin/python -m inpaint.cli --lama-video.masks-json outputs/0/segment/masks.json --lama-video.vis --lama-video.lama.model-path ckpts/big-lama/big-lama.pt
04: retarget
    retarget/.venv/bin/python -m retarget --input inputs/0.hdf5 --ik-config configs/egodex_g1_inspire_dfq.json --frames-json outputs/0/process/frames.json --vis
05: recamera
"""
