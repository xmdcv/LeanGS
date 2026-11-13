# Data preparation

The batch configuration uses 13 scenes:

| Dataset | Scenes | Default directory |
|---|---|---|
| Mip-NeRF 360 | bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai | `datasets/mipnerf360` |
| Tanks & Temples | truck, train | `datasets/tanksandtemples` |
| Deep Blending | playroom, drjohnson | `datasets/deepblending` |

Obtain Mip-NeRF 360 from [the dataset authors](https://jonbarron.info/mipnerf360/). The original Gaussian Splatting project provides [prepared Tanks & Temples and Deep Blending inputs](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

For each prepared COLMAP scene, the loader expects images and camera reconstruction data:

```text
scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Equivalent COLMAP text reconstruction files are supported by the scene loader. `convert.py` is the inherited conversion tool for preparing your own images with COLMAP.

Set `NERF360_ROOT`, `TANDT_ROOT`, and `DB_ROOT` when your dataset locations differ from these defaults. Keep original scene names when using the batch launchers. Use the image directory and resolution corresponding to your experiment configuration.
