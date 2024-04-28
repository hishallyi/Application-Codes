"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 卓宇师兄写的图像单应变换函数
"""

import torch
from kornia.utils import create_meshgrid
import torch.nn.functional as F


def homo_warp(src_feat, src_proj, ref_proj_inv, depth_values, back=0):
    # src_feat: (B, V, C, H, W)
    # src_proj: (B, V, 4, 4)
    # ref_proj_inv: (B, 4, 4)
    # depth_values: (B, D)
    # out: (B, V, C, D, H, W)
    B, V, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype

    if back == 0:
        transform = src_proj.cuda() @ ref_proj_inv.unsqueeze(1)
    elif back == 1:
        transform = torch.inverse(src_proj.cuda() @ ref_proj_inv.unsqueeze(1))

    R = transform[:, :, :3, :3]  # (B, V, 3, 3)
    T = transform[:, :, :3, 3:]  # (B, V, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H * W)  # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
    ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, 1)  # (B, 3, D, H*W)
    ref_grid_d = ref_grid_d.view(B, 3, D * H * W)
    src_grid_d = R @ ref_grid_d.unsqueeze(1) + T  # (B, V, 3, D*H*W)
    del ref_grid_d, ref_grid, transform, R, T  # release (GPU) memory
    div_val = src_grid_d[:, :, -1:]
    div_val[div_val < 1e-4] = 1e-4
    src_grid = src_grid_d[:, :, :2] / div_val  # divide by depth (B, V, 2, D*H*W)
    del src_grid_d, div_val
    src_grid[:, :, 0] = src_grid[:, :, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, :, 1] = src_grid[:, :, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 1, 3, 2)  # (B, V, D*H*W, 2)
    src_grid = src_grid.view(B, V, D, H * W, 2)

    warped_src_feats = []
    for i in range(V):
        warped_src_feat = F.grid_sample(src_feat[:, i], src_grid[:, i].to(torch.float32), mode='bilinear',
                                        padding_mode='zeros', align_corners=True)  # (B, C, D, H*W)
        warped_src_feat = warped_src_feat.view(B, C, D, H, W)
        warped_src_feats.append(warped_src_feat)

    warped_src_feats = torch.stack(warped_src_feats, dim=1)  # (B, V, C, D, H, W)
    return warped_src_feats
