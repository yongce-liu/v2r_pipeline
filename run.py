"""
00: process
    process/.venv/bin/python -m process.cli --video-path inputs/0.mp4
01: segment
    segment/.venv/bin/python -m segment.cli --video.frames-json outputs/0/process/frames.json --video.vis
02: inpaint
    inpaint/.venv/bin/python -m inpaint.cli --video.masks-json outputs/0/segment/masks.json --video.propainter.resize-ratio 0.5 --video.vis
03: depth
    depth/.venv/bin/python -m depth.cli --video.frames-json outputs/0/inpaint/inpainted.json --video.vis
04: retarget
    retarget/.venv/bin/python -m retarget --input inputs/0.hdf5 --ik-config configs/egodex_UnitreeG1InspireDfq_camera.json --frames-json outputs/0/process/frames.json --third-person-vis
06: rerender (depth composite)

"""
