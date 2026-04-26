import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, head_layer=256):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof, head_layer=head_layer).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    # Changed — Option A: accept a per-Gaussian human mask. When provided,
    # only human Gaussians go through the deformation MLP; background
    # Gaussians receive zero deltas. This implements the human/background
    # split without changing the MLP architecture. Passing is_human=None or
    # an all-True mask preserves the original behaviour (all Gaussians
    # deformed), which is what non-NeuMan scenes want.
    def step(self, xyz, time_emb, is_human=None):
        if is_human is None:
            return self.deform(xyz, time_emb)

        N = xyz.shape[0]
        human_idx = is_human.nonzero(as_tuple=False).squeeze(-1)
        n_human = int(human_idx.numel())

        # Degenerate cases: route nothing or route everything.
        if n_human == 0:
            zeros3 = torch.zeros(N, 3, device=xyz.device, dtype=xyz.dtype)
            zeros4 = torch.zeros(N, 4, device=xyz.device, dtype=xyz.dtype)
            return zeros3, zeros4, zeros3
        if n_human == N:
            return self.deform(xyz, time_emb)

        d_xyz_h, d_rot_h, d_scale_h = self.deform(xyz[human_idx], time_emb[human_idx])

        d_xyz = torch.zeros(N, d_xyz_h.shape[-1], device=xyz.device, dtype=d_xyz_h.dtype)
        d_rot = torch.zeros(N, d_rot_h.shape[-1], device=xyz.device, dtype=d_rot_h.dtype)
        d_scale = torch.zeros(N, d_scale_h.shape[-1], device=xyz.device, dtype=d_scale_h.dtype)
        d_xyz[human_idx] = d_xyz_h
        d_rot[human_idx] = d_rot_h
        d_scale[human_idx] = d_scale_h
        return d_xyz, d_rot, d_scale

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
