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

from pathlib import Path
import os
import re
import math
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


# Changed — read NeuMan source_path from the run's cfg_args so we can locate
# the segmentation masks alongside the rendered frames. Returns None if the
# file is missing or source_path cannot be parsed.
def _read_source_path_from_cfg(scene_dir):
    cfg_path = Path(scene_dir) / "cfg_args"
    if not cfg_path.exists():
        return None
    try:
        txt = cfg_path.read_text()
    except Exception:
        return None
    m = re.search(r"source_path=['\"]([^'\"]+)['\"]", txt)
    return m.group(1) if m else None


# Changed — load a NeuMan binary human mask at the rendered image's resolution.
# NeuMan segmentation PNGs encode human=0, background=255 (verified on the
# bike scene: 0 and 255 are the only values; 0 covers ~7.5% of the frame,
# consistent with a small foreground subject). The returned mask is 1.0 on
# human pixels, 0.0 on background; callers should use (1 - mask) for BG.
# Returns a float tensor of shape (1, 1, H, W) on CUDA, or None if not found.
def _load_mask(mask_path, H, W):
    if not mask_path.exists():
        return None
    m = Image.open(mask_path)
    if m.mode != "L":
        m = m.convert("L")
    m = tf.to_tensor(m).unsqueeze(0).cuda()  # (1, 1, h, w), values in [0, 1]
    if m.shape[-2:] != (H, W):
        # Nearest-neighbor so we do not invent soft pixels at the silhouette.
        m = F.interpolate(m, size=(H, W), mode="nearest")
    # tf.to_tensor scales uint8 → [0, 1]; human pixels (raw 0) stay at 0.
    return (m == 0).float()


# Changed — masked PSNR: MSE computed only over pixels where mask==1, then
# converted to dB. Returns None if the mask covers fewer than `min_frac` of
# pixels (metric is too noisy / undefined).
def _masked_psnr(pred, gt, mask, min_frac=0.005):
    # pred, gt: (1, 3, H, W) in [0, 1]; mask: (1, 1, H, W) in {0, 1}.
    n_pix = mask.sum().item()
    total = mask.numel()
    if n_pix < min_frac * total:
        return None
    sq_err = (pred - gt) ** 2  # (1, 3, H, W)
    # Broadcast mask across the 3 channels, then divide by (n_pix * 3).
    mse = (sq_err * mask).sum() / (n_pix * pred.shape[1])
    if mse.item() <= 0:
        return float("inf")
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()


def evaluate(model_paths, source_path_override=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # Changed — resolve the NeuMan source_path so we can read masks.
            # CLI override wins; otherwise fall back to cfg_args.
            src = source_path_override or _read_source_path_from_cfg(scene_dir)
            seg_dir = Path(src) / "segmentations" if src else None
            if seg_dir is None or not seg_dir.exists():
                print(f"  [warn] segmentations dir not found "
                      f"(src={src!r}); human/bg PSNR will be skipped.")
                seg_dir = None

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                if not method.startswith("ours"):
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                # Changed — accumulators for masked PSNR variants; per-frame
                # values (NaN if the frame's mask is unusable).
                psnrs_full = []
                psnrs_human = []
                psnrs_bg = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                    # Changed — full-frame PSNR as a plain float, then
                    # human/BG PSNR if a mask is available for this frame.
                    pv_full = psnr(renders[idx], gts[idx]).mean().item()
                    psnrs_full.append(pv_full)

                    pv_h = pv_b = None
                    if seg_dir is not None:
                        _, _, H, W = renders[idx].shape
                        mask = _load_mask(seg_dir / image_names[idx], H, W)
                        if mask is not None:
                            pv_h = _masked_psnr(renders[idx], gts[idx], mask)
                            pv_b = _masked_psnr(renders[idx], gts[idx], 1.0 - mask)
                    psnrs_human.append(pv_h)
                    psnrs_bg.append(pv_b)

                # Changed — aggregate masked PSNRs as mean over frames where
                # the metric is defined (same aggregation style as existing
                # SSIM/PSNR/LPIPS: mean of per-frame values).
                def _mean(vals):
                    good = [v for v in vals if v is not None and math.isfinite(v)]
                    return float(sum(good) / len(good)) if good else float("nan")

                mean_full = _mean(psnrs_full)
                mean_human = _mean(psnrs_human)
                mean_bg = _mean(psnrs_bg)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                # Changed — report the three PSNR variants side-by-side.
                print("  PSNR (full)  : {:>12.7f}".format(mean_full))
                print("  PSNR (human) : {:>12.7f}".format(mean_human))
                print("  PSNR (bg)    : {:>12.7f}".format(mean_bg))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS": torch.tensor(lpipss).mean().item(),
                                                     # Changed — new PSNR splits.
                                                     "PSNR_full": mean_full,
                                                     "PSNR_human": mean_human,
                                                     "PSNR_bg": mean_bg})
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                     # Changed — per-view masked PSNRs (None where undefined).
                     "PSNR_full": {name: v for v, name in zip(psnrs_full, image_names)},
                     "PSNR_human": {name: v for v, name in zip(psnrs_human, image_names)},
                     "PSNR_bg": {name: v for v, name in zip(psnrs_bg, image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "->", repr(e))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    # Changed — optional override for the NeuMan source_path. Needed when
    # evaluating a run on a different machine than it was trained on, where
    # the cfg_args path (e.g. /scratch/.../bike) does not exist locally.
    parser.add_argument('--source_path', '-s', type=str, default=None,
                        help="Override NeuMan scene dir (must contain segmentations/).")
    args = parser.parse_args()
    evaluate(args.model_paths, source_path_override=args.source_path)
