"""
00: process
    process/.venv/bin/python -m process.cli --video-path inputs/0.mp4
01: segment
    segment/.venv/bin/python -m segment.cli --video.frames-json outputs/0/process/frames.json --video.vis
02: inpaint
    inpaint/.venv/bin/python -m inpaint.cli --video.masks-json outputs/0/segment/masks.json --video.propainter.resize-ratio 0.5 --video.vis
03: depth
    depth/.venv/bin/python -m depth.cli --video.frames-json outputs/0/inpaint/inpainted.json --video.vis
04: depth-oringinal
    depth/.venv/bin/python -m depth.cli --video.frames-json outputs/0/process/frames.json --video.vis --output-root outputs/0/depth_orig
04: retarget
    retarget/.venv/bin/python -m retarget --input inputs/0.hdf5 --ik-config configs/egodex_UnitreeG1InspireDfq_camera.json --frames-json outputs/0/process/frames.json --third-person-vis
05: composite calibration depth (DA3 on the ORIGINAL frames, used by composite)
    depth/.venv/bin/python -m depth.cli --video.frames-json outputs/0/process/frames.json --video.output-subdir depth_orig --video.vis
06: composite (depth-aware compositing -> final robot operation video frames)
    composite/.venv/bin/python -m composite.cli \
        --inpainted-json outputs/0/inpaint/inpainted.json \
        --depth-json outputs/0/depth/depth.json \
        --calibration-depth-json outputs/0/depth_orig/depth.json \
        --masks-json outputs/0/segment/masks.json \
        --camera-json outputs/0/retarget/camera.json \
        --video

"""
