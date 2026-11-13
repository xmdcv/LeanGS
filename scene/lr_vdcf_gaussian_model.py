"""
Low-Rank View-Dependent Color Field (LR-VDCF) Gaussian Model.
Extended GaussianModel with low-rank view-dependent color support.
"""

import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn
from torch.autograd import Function

from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p
from utils.view_basis_network import LRVDCFColorModel, ViewBasisNetwork


class FeaturesDCTransposeFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is not None:
            if grad_output.shape[-2:] == (3, 1):
                return grad_output.transpose(1, 2).contiguous()
            if grad_output.shape[-2:] == (1, 3):
                return grad_output.contiguous()
            return grad_output
        return None


class LRVDCFGaussianModel(GaussianModel):
    def __init__(self, sh_degree, optimizer_type="default", K=6, basis_hidden_dim=128, use_lr_vdcf=True):
        super().__init__(sh_degree, optimizer_type)
        self.K = K
        self.use_lr_vdcf = use_lr_vdcf
        self._alpha_coeffs = torch.empty(0)

        if use_lr_vdcf:
            self.basis_network = ViewBasisNetwork(
                K=K, hidden_dim=basis_hidden_dim, use_positional_encoding=True
            ).cuda()
            self.color_model = LRVDCFColorModel(self.basis_network, K=K)
        else:
            self.basis_network = None
            self.color_model = None

    def create_from_pcd(self, pcd, spatial_lr_scale):
        super().create_from_pcd(pcd, spatial_lr_scale)
        num_points = self._xyz.shape[0]
        self._sync_features_rest_rows()
        if self.use_lr_vdcf:
            self._alpha_coeffs = nn.Parameter(torch.randn(num_points, self.K, device="cuda") * 0.01)
        else:
            self._alpha_coeffs = nn.Parameter(torch.zeros(num_points, self.K, device="cuda"))

    def _sync_features_rest_rows(self):
        """Keep _features_rest row count aligned with current number of points."""
        n_points = self._xyz.shape[0]
        device = self._xyz.device
        dtype = self._xyz.dtype
        if (
            not isinstance(self._features_rest, torch.Tensor)
            or self._features_rest.ndim != 3
            or self._features_rest.shape[0] != n_points
            or self._features_rest.shape[1] != 0
            or self._features_rest.shape[2] != 3
        ):
            self._features_rest = nn.Parameter(
                torch.zeros((n_points, 0, 3), device=device, dtype=dtype).requires_grad_(True)
            )

    @property
    def get_features_dc(self):
        if self._features_dc.shape[-2:] == (1, 3):
            return FeaturesDCTransposeFunction.apply(self._features_dc)
        return self._features_dc

    def get_base_color(self):
        features_dc = self._features_dc
        sh_dc = features_dc.transpose(1, 2)
        sh_dc_flat = sh_dc.squeeze(-1)
        return SH2RGB(sh_dc_flat)

    def get_color(self, view_dirs):
        base_rgb = self.get_base_color()
        if self.use_lr_vdcf and self.basis_network is not None:
            alpha = self._alpha_coeffs
            if alpha.shape[0] != base_rgb.shape[0]:
                alpha = alpha.reshape(base_rgb.shape[0], -1)
            elif alpha.dim() != 2:
                alpha = alpha.view(base_rgb.shape[0], -1)
            if alpha.shape[1] != self.K:
                if alpha.shape[1] > self.K:
                    alpha = alpha[:, : self.K]
                else:
                    pad = torch.zeros(
                        (alpha.shape[0], self.K - alpha.shape[1]), device=alpha.device, dtype=alpha.dtype
                    )
                    alpha = torch.cat([alpha, pad], dim=1)
            colors = self.color_model.compute_color(base_rgb, alpha, view_dirs)
        else:
            colors = torch.clamp(base_rgb, 0.0, 1.0)
        return colors

    def training_setup(self, training_args):
        super().training_setup(training_args)
        self._sync_features_rest_rows()
        if self.use_lr_vdcf and self._alpha_coeffs.numel() > 0:
            alpha_lr = training_args.lowfeature_lr * 0.5
            alpha_params = [{"params": [self._alpha_coeffs], "lr": alpha_lr, "name": "alpha_coeffs"}]
            if self.optimizer_type == "default":
                existing_params = [group for group in self.optimizer.param_groups]
                existing_params.extend(alpha_params)
                self.optimizer = torch.optim.Adam(existing_params, lr=0.0, eps=1e-15)

            basis_lr = training_args.lowfeature_lr * 0.1
            basis_params = [{"params": self.basis_network.parameters(), "lr": basis_lr, "name": "basis_network"}]
            if self.optimizer_type == "default":
                # LR-VDCF does not optimize f_rest; keep shoptimizer global-only.
                self.shoptimizer = torch.optim.Adam(basis_params, lr=0.0, eps=1e-15)

    def create_subset(self, mask):
        subset = LRVDCFGaussianModel(
            sh_degree=self.max_sh_degree,
            optimizer_type="default",
            K=self.K,
            use_lr_vdcf=self.use_lr_vdcf,
        )
        subset.active_sh_degree = self.active_sh_degree
        subset._xyz = self._xyz[mask].detach().clone()
        subset._features_dc = self._features_dc[mask].detach().clone()
        subset._features_rest = self._features_rest[mask].detach().clone()
        subset._alpha_coeffs = self._alpha_coeffs[mask].detach().clone()
        subset._scaling = self._scaling[mask].detach().clone()
        subset._rotation = self._rotation[mask].detach().clone()
        subset._opacity = self._opacity[mask].detach().clone()
        n = subset._xyz.shape[0]
        subset.max_radii2D = torch.zeros((n,), device="cuda", dtype=torch.float32)
        if self.use_lr_vdcf and self.basis_network is not None:
            subset.basis_network.load_state_dict(self.basis_network.state_dict())
            subset.basis_network.eval()
        return subset

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append(f"f_dc_{i}")
        for i in range(self.K):
            attrs.append(f"alpha_{i}")
        attrs.append("opacity")
        for i in range(self._scaling.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attrs.append(f"rot_{i}")
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        alpha = (
            self._alpha_coeffs.detach().cpu().numpy()
            if self._alpha_coeffs.numel() > 0
            else np.zeros((xyz.shape[0], self.K))
        )

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, alpha, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        if self.use_lr_vdcf and self.basis_network is not None:
            basis_network_path = path.replace(".ply", "_basis_network.pth")
            torch.save(self.basis_network.state_dict(), basis_network_path)
            print(f"Saved basis network to: {basis_network_path}")

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        alpha_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alpha_")]
        if len(alpha_names) > 0:
            alpha_names = sorted(alpha_names, key=lambda x: int(x.split("_")[-1]))
            alpha = np.zeros((xyz.shape[0], len(alpha_names)))
            for idx, attr_name in enumerate(alpha_names):
                alpha[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self.K = len(alpha_names)
        else:
            alpha = np.zeros((xyz.shape[0], self.K))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.zeros((xyz.shape[0], 0, 3), dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._alpha_coeffs = nn.Parameter(torch.tensor(alpha, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sync_features_rest_rows()

        basis_network_path = path.replace(".ply", "_basis_network.pth")
        possible_paths = [
            basis_network_path,
            os.path.join(os.path.dirname(path), os.path.basename(path).replace(".ply", "_basis_network.pth")),
            os.path.join(os.path.dirname(path), "basis_network.pth"),
            os.path.join(os.path.dirname(path), "point_cloud_basis_network.pth"),
        ]

        basis_loaded = False
        if self.use_lr_vdcf:
            for basis_path in possible_paths:
                if os.path.exists(basis_path):
                    try:
                        saved_state = torch.load(basis_path)
                        current_state = self.basis_network.state_dict()
                        if set(saved_state.keys()) == set(current_state.keys()):
                            self.basis_network.load_state_dict(saved_state)
                            print(f"Loaded basis network from: {basis_path}")
                            basis_loaded = True
                            break
                        print(f"Warning: Basis network architecture mismatch at {basis_path}")
                        print(
                            f"  Saved keys: {len(saved_state.keys())}, Current keys: {len(current_state.keys())}"
                        )
                        print("  Using newly initialized basis network with updated architecture...")
                        basis_loaded = True
                        break
                    except Exception as exc:
                        print(f"Warning: Failed to load basis network from {basis_path}: {exc}")
                        continue
            if not basis_loaded:
                print("Basis network not found, initializing new one")

        if self.use_lr_vdcf and self.basis_network is not None:
            self.color_model = LRVDCFColorModel(self.basis_network, K=self.K)

    def densify_and_split_fastgs(self, metric_mask, filter_mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")

        if metric_mask is None:
            metric_mask = torch.zeros((n_init_points,), device="cuda", dtype=torch.bool)
        else:
            metric_mask = metric_mask.reshape(-1).bool()
            if metric_mask.numel() != n_init_points:
                padded = torch.zeros((n_init_points,), device=metric_mask.device, dtype=metric_mask.dtype)
                size = min(n_init_points, metric_mask.numel())
                padded[:size] = metric_mask[:size]
                metric_mask = padded

        if filter_mask is None:
            filter_mask = torch.zeros((n_init_points,), device="cuda", dtype=torch.bool)
        else:
            filter_mask = filter_mask.reshape(-1).bool()
            if filter_mask.numel() != n_init_points:
                padded = torch.zeros((n_init_points,), device=filter_mask.device, dtype=filter_mask.dtype)
                size = min(n_init_points, filter_mask.numel())
                padded[:size] = filter_mask[:size]
                filter_mask = padded

        selected_pts_mask = torch.logical_and(filter_mask, metric_mask)
        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self.get_rotation[selected_pts_mask].repeat(N, 1)
        from utils.general_utils import build_rotation

        new_xyz = torch.bmm(build_rotation(rots), samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_alpha_coeffs = self._alpha_coeffs[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        tmp_radii = self.tmp_radii
        if tmp_radii.shape[0] != n_init_points:
            fixed = torch.zeros((n_init_points,), device=tmp_radii.device, dtype=tmp_radii.dtype)
            size = min(n_init_points, tmp_radii.shape[0])
            fixed[:size] = tmp_radii[:size]
            tmp_radii = fixed
        new_tmp_radii = tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz, new_features_dc, new_alpha_coeffs, new_opacity, new_scaling, new_rotation, new_tmp_radii
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter_mask):
        n_init_points = self.get_xyz.shape[0]
        if metric_mask is None:
            metric_mask = torch.zeros((n_init_points,), device="cuda", dtype=torch.bool)
        else:
            metric_mask = metric_mask.reshape(-1).bool()
            if metric_mask.numel() != n_init_points:
                padded = torch.zeros((n_init_points,), device=metric_mask.device, dtype=metric_mask.dtype)
                size = min(n_init_points, metric_mask.numel())
                padded[:size] = metric_mask[:size]
                metric_mask = padded

        if filter_mask is None:
            filter_mask = torch.zeros((n_init_points,), device="cuda", dtype=torch.bool)
        else:
            filter_mask = filter_mask.reshape(-1).bool()
            if filter_mask.numel() != n_init_points:
                padded = torch.zeros((n_init_points,), device=filter_mask.device, dtype=filter_mask.dtype)
                size = min(n_init_points, filter_mask.numel())
                padded[:size] = filter_mask[:size]
                filter_mask = padded

        selected_pts_mask = torch.logical_and(metric_mask, filter_mask)
        if selected_pts_mask.sum() == 0:
            return

        tmp_radii = self.tmp_radii
        if tmp_radii.shape[0] != n_init_points:
            fixed = torch.zeros((n_init_points,), device=tmp_radii.device, dtype=tmp_radii.dtype)
            size = min(n_init_points, tmp_radii.shape[0])
            fixed[:size] = tmp_radii[:size]
            tmp_radii = fixed

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_alpha_coeffs = self._alpha_coeffs[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = tmp_radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_alpha_coeffs, new_opacities, new_scaling, new_rotation, new_tmp_radii
        )

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        if "alpha_coeffs" in optimizable_tensors:
            self._alpha_coeffs = optimizable_tensors["alpha_coeffs"]
        else:
            self._alpha_coeffs = nn.Parameter(self._alpha_coeffs[valid_points_mask].contiguous())
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sync_features_rest_rows()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def densification_postfix(
        self, new_xyz, new_features_dc, new_alpha_coeffs, new_opacities, new_scaling, new_rotation, new_tmp_radii
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "alpha_coeffs": new_alpha_coeffs,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._alpha_coeffs = optimizable_tensors["alpha_coeffs"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sync_features_rest_rows()

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            if name not in tensors_dict:
                continue

            extension_tensor = tensors_dict[name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    def replace_tensor_to_optimizer(self, *args):
        if len(args) == 2:
            return super().replace_tensor_to_optimizer(args[0], args[1])
        if len(args) != 1 or not isinstance(args[0], dict):
            raise TypeError("replace_tensor_to_optimizer expects (tensor, name) or (tensor_dict)")

        tensor_dict = args[0]
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in tensor_dict:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor_dict[name])
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor_dict[name])
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor_dict[name].requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        Prune point-wise tensors in optimizer state.
        Skip global tensors (e.g., basis network params) that are not indexed by points.
        """
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer:
            optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                param = group["params"][0]
                is_pointwise = param.ndim > 0 and param.shape[0] == mask.shape[0]
                if not is_pointwise:
                    continue

                stored_state = opt.state.get(param, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[param]
                    group["params"][0] = nn.Parameter((param[mask].requires_grad_(True)))
                    opt.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(param[mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
