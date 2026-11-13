#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
from gaussian_renderer.lr_vdcf_renderer import render_lr_vdcf
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.lr_vdcf_gaussian_model import LRVDCFGaussianModel
import time


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    args,
    renders_subdir="renders",
    render_func=render_fastgs,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), renders_subdir)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        rendering = render_func(view, gaussians, pipeline, background, args.mult)["render"]
        end_time = time.time()
        total_time += (end_time - start_time)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    fps = len(views) / total_time
    print(f"Scene: {name}, FPS: {fps:.2f}")
    


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        if args.use_lr_vdcf:
            gaussians = LRVDCFGaussianModel(sh_degree=0, K=args.lr_vdcf_k, use_lr_vdcf=True)
            render_func = render_lr_vdcf
        else:
            gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
            render_func = render_fastgs
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, render_func=render_func)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, render_func=render_func)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--use_lr_vdcf", action="store_true", default=False)
    parser.add_argument("--lr_vdcf_k", type=int, default=9)
    args = get_combined_args(parser)
    print(f"Render args: use_lr_vdcf={args.use_lr_vdcf}, model_path={args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)