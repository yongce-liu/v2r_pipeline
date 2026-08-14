# recamera

First-person depth-matched robot arm rendering.

Reads the per-episode outputs of the earlier pipeline stages (segment masks,
depth + intrinsics, process frame manifest, retarget trajectory), solves a
first-person camera pose per frame so the retargeted robot arm covers the human
arm from the original video, and writes transparent-background RGBA frames.

## Usage

From the repo root:

```bash
uv run recamera --input outputs/0
```

Outputs per-frame transparent PNGs to `outputs/<stem>/first_person/frames/`
plus a `config.json` manifest.

## How it works

* Human arm cloud = `mask × depth` backprojection (camera frame).
* Robot skeleton anchors (arm + finger links) in the retarget world frame.
* Frame 0: absolute camera pose via Open3D ICP seeded from skeleton/cloud PCA
  long-axis alignment (no robot camera-mount prior).
* Later frames: camera pose propagated by an ICP of consecutive human arm
  clouds.
* Render the robot arm under the solved pose; segmentation pass keeps only arm
  pixels opaque (alpha=255), everything else alpha=0.
