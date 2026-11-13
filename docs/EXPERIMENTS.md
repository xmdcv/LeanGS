# Experiment guide

## Training configuration

The provided launcher uses rank 5, contribution aggregation over 4 visible views, a fixed pruning ratio of 0.25, 30,000 training iterations and a densification interval of 500.

## Per-scene settings

All scenes use the common settings above with the following scene-specific options:

| Scene | Additional settings |
|---|---|
| bicycle | `--grad_abs_thresh 0.0012` |
| flowers | `--dense 0.005 --grad_abs_thresh 0.0015` |
| garden | `--highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008` |
| stump | `--dense 0.004 --grad_abs_thresh 0.0015` |
| treehill | `--dense 0.01 --grad_abs_thresh 0.002` |
| room | `--highfeature_lr 0.02 --grad_abs_thresh 0.0008` |
| counter | `--highfeature_lr 0.02 --grad_abs_thresh 0.0008` |
| kitchen | `--highfeature_lr 0.02 --grad_abs_thresh 0.0006` |
| bonsai | `--highfeature_lr 0.02 --grad_abs_thresh 0.0006` |
| truck | `--highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7` |
| train | `--highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7` |
| playroom | `--highfeature_lr 0.0015 --dense 0.003 --mult 0.7` |
| drjohnson | `--highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7` |

The evaluation launcher explicitly uses `--mult 0.5` for Mip-NeRF 360 and `--mult 0.7` for Tanks & Temples and Deep Blending, matching training. It also passes rank 5 and the factored-color switch explicitly.

## Outputs

With `OUT_BASE=./output/leangs`, each scene produces:

```text
output/leangs/<scene>/
├── cfg_args
├── point_cloud/
│   └── iteration_30000/
│       ├── point_cloud.ply
│       └── point_cloud_basis_network.pth
├── test/
│   └── ours_30000/
│       ├── renders/
│       └── gt/
├── results.json
└── per_view.json
```

The point cloud stores per-Gaussian attributes, while `point_cloud_basis_network.pth` stores the shared learned color basis. Distributing only the point cloud omits part of the model.

Batch logs are written to `<OUT_BASE>/<scene>.log` for training and `<OUT_BASE>/<scene>_evaluation.log` for evaluation. Commands stop on failure. No trained weights, datasets or experiment logs are included in the source distribution.

`full_eval.py` is an inherited baseline evaluation utility; use the documented launchers for the factored-color configuration. `prune_adaptive.py` is a standalone inherited pruning experiment and is not called by the documented training launcher.
