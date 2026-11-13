"""
LR-VDCF renderer.
"""

import math
import torch

from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer
from scene.lr_vdcf_gaussian_model import LRVDCFGaussianModel


def render_lr_vdcf(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    mult,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=None,
    metric_map=None,
    weight_accum=None,
):
    is_lr_vdcf = isinstance(pc, LRVDCFGaussianModel)

    screenspace_points = (
        torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(
            int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width), dtype=torch.int, device="cuda"
        )

    sh_degree = 0 if is_lr_vdcf else pc.active_sh_degree

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        mult=mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    dc = None

    if override_color is not None:
        colors_precomp = override_color
    elif is_lr_vdcf:
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1)
        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
        colors_precomp = pc.get_color(dir_pp_normalized)

        features_dc = pc.get_features_dc
        n_points = features_dc.shape[0]
        flat_dc = features_dc.reshape(n_points, -1)
        if flat_dc.shape[1] < 3:
            pad = 3 - flat_dc.shape[1]
            flat_dc = torch.cat(
                [flat_dc, torch.zeros((n_points, pad), device=flat_dc.device, dtype=flat_dc.dtype)], dim=1
            )
        elif flat_dc.shape[1] > 3:
            flat_dc = flat_dc[:, :3]
        dc = flat_dc.view(n_points, 1, 3).contiguous().detach()
        shs = None
    else:
        if pipe.convert_SHs_python:
            from utils.sh_utils import eval_sh

            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest

    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        weight_accum=weight_accum,
    )

    if accum_metric_counts is not None:
        accum_metric_counts = accum_metric_counts.view(-1)
        if accum_metric_counts.shape[0] != means3D.shape[0]:
            padded_counts = torch.zeros((means3D.shape[0],), device=accum_metric_counts.device, dtype=accum_metric_counts.dtype)
            size = min(padded_counts.shape[0], accum_metric_counts.shape[0])
            padded_counts[:size] = accum_metric_counts[:size]
            accum_metric_counts = padded_counts

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts,
    }
