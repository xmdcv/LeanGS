# Third-party sources and notices

This source tree includes components with separate licenses. Existing copyright headers and component license files have been retained.

| Component | Location | Included notice |
|---|---|---|
| FastGS backbone | Core source and optimized rasterizer | Root `LICENSE` reproduces the inherited FastGS MIT notice |
| Original Gaussian Splatting | Core source and derived components | `LICENSE_ORIGINAL.md`; identical `LICENSE.md` is also provided for source headers that reference that filename |
| Optimized rasterizer | `submodules/diff-gaussian-rasterization_fastgs` | Component `LICENSE.md` |
| Depth rasterizer | `submodules/diff-gaussian-rasterization-depth` | Component `LICENSE.md` |
| Simple nearest-neighbor extension | `submodules/simple-knn` | Component `LICENSE.md` |
| Fused SSIM | `submodules/fused-ssim` | Component `LICENSE` |
| GLM headers | Rasterizer `third_party/glm` directories | Component `copying.txt` |

Root notices do not replace component-specific notices. The original Gaussian Splatting notices include research and non-commercial use terms.

Upstream projects: [FastGS](https://github.com/fastgs/FastGS), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

The figures in `assets/` are from the LeanGS final manuscript. Their article context is [LeanGS, ACM Multimedia 2026](https://doi.org/10.1145/3767308.3835628); the article identifies a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license. The article's figure notice and the source-code component licenses describe different material.
