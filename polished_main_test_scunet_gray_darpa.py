#!/usr/bin/env python3
"""
polished_main_test_scunet_gray_darpa.py

Script to denoise grayscale DARPA dataset using SCUNet.
"""
import argparse
import logging
from pathlib import Path

import bm3d
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from utils.utils_image import uint2single, single2tensor4
from models.network_scunet import SCUNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Denoise grayscale images using SCUNet."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Path to directory containing SNR case subfolders with .tif images."
    )
    parser.add_argument(
        "--snr_case", choices=["LowSNR", "MedSNR", "HighSNR"], default="MedSNR",
        help="Which SNR case folder to process."
    )
    parser.add_argument(
        "--model_dir", type=Path, default=Path("model_zoo"),
        help="Directory containing the pretrained model weights."
    )
    parser.add_argument(
        "--model_name", type=str, default="scunet_gray_25",
        help="Filename (without extension) of the pretrained model."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./result"),
        help="Directory where outputs will be saved."
    )
    parser.add_argument(
        "--clip_value", type=float, default=60.0,
        help="Max value for clipping outside ROI."
    )
    parser.add_argument(
        "--box", type=int, nargs=4, metavar=("y1", "y2", "x1", "x2"),
        default=[100, 300, 80, 250],
        help="ROI box for preserving values (y1 y2 x1 x2)."
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )
    return parser.parse_args()


def setup_logger(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def key_func(fname: str) -> int:
    """Extract angle in degrees from filename (e.g., 'thetaXXdeg')."""
    return int(Path(fname).stem.split('theta')[-1].split('deg')[0])


def load_projections(input_dir: Path, case: str):
    """Load .tif projections, apply BM3D, and convert to log scale tensor."""
    folder = input_dir / case
    file_paths = sorted(
        [str(p) for p in folder.glob("*.tif") if "FullResponse" in p.name],
        key=key_func
    )
    logging.info(f"Found {len(file_paths)} projection files in {folder}")

    # Load and stack
    rad_stack = torch.stack([
        torch.from_numpy(imageio.imread(fp).astype(np.float32))
        for fp in file_paths
    ], dim=0).unsqueeze(0)

    # Mean-pool chunks to match original behaviour
    num_slices = rad_stack.shape[1]
    rad_processed = torch.cat([
        chunk.mean(1, keepdim=True)
        for chunk in rad_stack.chunk(num_slices, dim=1)
    ], dim=1)

    # Apply BM3D slice-wise
    bm3d_slices = []
    for i in range(rad_processed.shape[1]):
        slice_img = rad_processed[0, i]
        sigma = slice_img.std().item() / 2
        denoised = bm3d.bm3d(slice_img.numpy(), sigma)
        bm3d_slices.append(torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0))
    rad_denoised = torch.cat(bm3d_slices, dim=1).float()

    # Convert to negative log scale
    proj = -(rad_denoised / rad_denoised.max()).log()
    angles = torch.tensor([key_func(fp) for fp in file_paths], dtype=torch.float32)
    logging.info(f"Projection tensor shape: {proj.shape}")
    return proj, angles


def load_model(model_dir: Path, model_name: str, device: torch.device):
    """Load pretrained SCUNet model."""
    path = model_dir / f"{model_name}.pth"
    model = SCUNet(in_nc=1, config=[4] * 7, dim=64).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logging.info(f"Loaded model from {path}")
    return model


def denoise_stack(proj: torch.Tensor, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """Denoise each slice, rescale, and return numpy stack."""
    num_slices = proj.shape[1]
    output = []
    for i in tqdm(range(num_slices), desc="Denoising slices"):
        slice_fp = proj[0, i]
        minv, maxv = slice_fp.min().item(), slice_fp.max().item()
        norm = (slice_fp - minv) / (maxv - minv + 1e-8)
        img_uint8 = (norm * 255.0).round().to(torch.uint8).cpu().numpy()

        img_hw = uint2single(img_uint8)
        img_tensor = single2tensor4(img_hw[..., None]).to(device)

        with torch.no_grad():
            den = model(img_tensor).squeeze(0)

        den_np = den.cpu().numpy()
        den_rescaled = den_np * (maxv - minv) + minv
        output.append(den_rescaled)
    return np.stack(output, axis=0)


def apply_clipping_mask(stack: np.ndarray, clip_value: float, box: tuple) -> np.ndarray:
    """Clip values outside the given ROI box to clip_value."""
    y1, y2, x1, x2 = box
    mask = np.zeros(stack.shape[1:], dtype=bool)
    mask[y1:y2, x1:x2] = True
    clipped = np.clip(stack, 0, clip_value)
    return np.where(mask[None, ...], stack, clipped)


def save_results(stack: np.ndarray, output_dir: Path):
    """Save denoised stack as .npy file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "denoised_stack.npy"
    np.save(out_path, stack)
    logging.info(f"Saved denoised stack to {out_path}")


def main():
    args = parse_args()
    setup_logger(args.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj, angles = load_projections(args.input_dir, args.snr_case)
    model = load_model(args.model_dir, args.model_name, device)

    denoised = denoise_stack(proj.to(device), model, device)
    clipped = apply_clipping_mask(denoised, args.clip_value, tuple(args.box))
    save_results(clipped, args.output_dir)


if __name__ == "__main__":
    main()
