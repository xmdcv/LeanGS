import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from plyfile import PlyData, PlyElement

def adaptive_prune(dataset, opt, pipe, args):
    # 1. Load Model
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # 2. Setup
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    if args.sample_rate > 1:
        cameras = cameras[::args.sample_rate]
    
    n_points = gaussians.get_xyz.shape[0]
    global_hit_counts = torch.zeros(n_points, device="cuda", dtype=torch.float32)
    
    # 3. Compute Visibility (Hit Counts)
    with torch.no_grad():
        for cam in tqdm(cameras):
            # Metric Map = 1 means we count all pixels the gaussian projects to
            metric_map = torch.ones(
                int(cam.image_height) * int(cam.image_width), 
                dtype=torch.int, 
                device='cuda'
            )
            render_pkg = render_fastgs(cam, gaussians, pipe, background, args.mult, metric_map=metric_map)
            global_hit_counts += render_pkg["accum_metric_counts"].float()
            torch.cuda.empty_cache()

    # 4. Compute Scale Factor
    # We use the max scale dimension to represent the "projected potential"
    scales = gaussians.get_scaling
    max_scales = torch.max(scales, dim=1).values
    
    # Normalize scales to [0, 1]
    max_scales_norm = (max_scales - max_scales.min()) / (max_scales.max() - max_scales.min() + 1e-6)
    
    # 5. Compute Adaptive Score (The Core Innovation)
    # Score = Hit_Count / (Scale_Factor^alpha + epsilon)
    # Idea: Large points are expected to be hit more. If they aren't, they are redundant.
    #       Small points are hard to hit. Their hits are more valuable.
    alpha = args.scale_penalty 
    adaptive_score = global_hit_counts / (max_scales_norm + 0.01) ** alpha
    
    # 6. Determine Threshold
    target_count = int(n_points * (1 - args.prune_ratio))
    sorted_scores, _ = torch.sort(adaptive_score)
    threshold = sorted_scores[n_points - target_count] # Cutoff at the bottom X%
    
    prune_mask = adaptive_score < threshold
    gaussians.prune_points(prune_mask)

    # 7. Save Optimized PLY (Compact Storage: No Normals)
    save_path = os.path.join(args.model_path, f"point_cloud/iteration_{args.iteration}", "point_cloud_adaptive.ply")
    save_optimized_ply(gaussians, save_path)

def save_optimized_ply(gaussians, path):
    """
    Saves PLY without zero-normals to save space (~5-10% reduction).
    """
    mkdir_p(os.path.dirname(path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    def construct_list_of_attributes():
        l = ['x', 'y', 'z'] # No nx, ny, nz
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def mkdir_p(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Adaptive Pruning Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--prune_ratio", type=float, default=0.5, help="Ratio of points to prune (e.g. 0.5 for 50%%)")
    parser.add_argument("--sample_rate", type=int, default=1, help="Downsample cameras for speed")
    parser.add_argument("--scale_penalty", type=float, default=1.0, help="Alpha parameter for scale penalty")
    
    args = parser.parse_args(sys.argv[1:])
    
    from utils.system_utils import searchForMaxIteration
    if args.iteration == -1:
        args.iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    adaptive_prune(lp.extract(args), op.extract(args), pp.extract(args), args)
