# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import math, torch, numpy as np


def ts2tsemb1d(ts, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    ts = ts * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ts.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos = ts[..., 0, None] / dim_t
    posemb = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def time_position_embedding(track_num, frame_num, embed_dims, device):
    ts = torch.arange(0, 1 + 1e-5, 1/(frame_num - 1), dtype=torch.float32, device=device)
    ts = ts[None, :] * torch.ones((track_num, frame_num), dtype=torch.float32, device=device)
    ts_embed = ts2tsemb1d(ts.view(track_num * frame_num, 1), num_pos_feats=embed_dims).view(track_num, frame_num, embed_dims)
    return ts_embed


def xyz_ego_transformation(xyz, l2g0, l2g1, pc_range, 
                           src_normalized=True, tgt_normalized=True):
    """Transform xyz coordinates from l2g0 to l2g1
       xyz has to be denormalized
    """
    # denormalized to the physical coordinates
    if src_normalized:
        xyz = denormalize(xyz, pc_range)
    
    # to global, then to next local
    if torch.__version__ < '1.9.0':
        g2l1 = torch.tensor(np.linalg.inv(l2g1.cpu().numpy())).type(torch.float).to(l2g1.device)
    else:
        g2l1 = torch.linalg.inv(l2g1).type(torch.float)
    xyz = xyz @ l2g0[:3, :3].T + l2g0[:3, 3] - l2g1[:3, 3]
    xyz = xyz @ g2l1[:3, :3].T

    # normalize to 0-1
    if tgt_normalized:
        xyz = normalize(xyz, pc_range)
    return xyz


def normalize(xyz, pc_range):
    xyz[..., 0:1] = (xyz[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
    xyz[..., 1:2] = (xyz[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
    xyz[..., 2:3] = (xyz[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return xyz

def denormalize(xyz, pc_range):
    xyz[..., 0:1] = xyz[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    xyz[..., 1:2] = xyz[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    xyz[..., 2:3] = xyz[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return xyz
