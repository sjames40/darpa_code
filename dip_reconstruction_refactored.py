import argparse
import glob
import os
from pathlib import Path

import bm3d
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as compute_psnr
from torchmetrics.functional import total_variation as compute_tv

from physics.ct import PBCT
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Image Prior CT Reconstruction"
    )
    parser.add_argument(
        "--input_pattern", type=str, required=True,
        help="Glob pattern for input TIFF radiographs"
    )
    parser.add_argument(
        "--scu_npy", type=Path, required=True,
        help="NumPy file with precomputed denoised radiographs"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to save the reconstruction results (torch.save format)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation"
    )
    parser.add_argument("--num_views", type=int, default=20,
                        help="Number of projection angles to average")
    parser.add_argument("--mac", type=float, default=1/50,
                        help="Multiplicative attenuation coefficient")
    parser.add_argument("--tv_lambda", type=float, default=0.0,
                        help="Weight for total variation regularization")
    parser.add_argument("--z_consistency_lambda", type=float, default=0.0,
                        help="Weight for z-consistency regularization across slices")
    parser.add_argument("--random_seed", type=int, default=123,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def load_and_sort_images(pattern: str) -> list[Path]:
    def angle_key(path: Path):
        name = path.stem
        angle_str = name.split('theta')[-1].split('deg')[0]
        return int(angle_str)

    files = [Path(p) for p in glob.glob(pattern)]
    files = [p for p in files if 'FullResponse' in p.name]
    files.sort(key=angle_key)
    return files


def stack_and_average(images: list[Path], num_views: int) -> torch.Tensor:
    # Load, stack and average projections
    arrs = [imageio.imread(str(p)).astype(np.float32) for p in images]
    tensor = torch.from_numpy(np.stack(arrs, axis=0))  # [N, H, W]
    tensor = tensor.unsqueeze(0)  # [1, N, H, W]
    chunks = torch.chunk(tensor, chunks=num_views, dim=1)
    averaged = torch.cat([c.mean(dim=1, keepdim=True) for c in chunks], dim=1)
    return averaged  # [1, num_views, H, W]


def load_scu(scu_path: Path) -> torch.Tensor:
    arr = np.load(scu_path)
    return torch.from_numpy(arr).unsqueeze(0)  # [1, num_views, H, W]


def compute_projections(radiographs: torch.Tensor) -> torch.Tensor:
    # Negative log-normalized projections
    proj = -(radiographs / radiographs.max()).log()
    proj = torch.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    proj[torch.isnan(proj)] = 0.0
    return proj


def reconstruct(
    proj: torch.Tensor,
    angles: torch.Tensor,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
) -> tuple[torch.Tensor, list[float]]:
    device = args.device
    num_slices = proj.shape[2]
    pbct = PBCT(
        num_views=args.num_views,
        num_rows=1,
        num_cols=proj.shape[3],
        batch_size=3,
        device=device,
        angles=angles.to(device)
    )

    recon_slices = []
    losses = []

    for idx in range(num_slices):
        # Determine neighboring slice indices
        slice_idxs = [max(idx-1, 0), idx, min(idx+1, num_slices-1)]

        # Prepare input block
        inp = proj[0, :, slice_idxs, :].cpu()
        inp = inp.unsqueeze(1).to(device)  # [3,1,H,W]

        target = inp.clone().to(device)
        theta_block = proj.clone().to(device)

        # Inner optimization
        steps = 50 if idx == 0 else 20
        for _ in range(steps):
            optimizer.zero_grad()
            out = net(inp).abs()

            pred = pbct.A(out)  # [1,1,detectors,angles]
            pred = pred.permute(0, 2, 1, 3) * args.mac

            mse = F.mse_loss(pred, target)
            reg = args.tv_lambda * compute_tv(out)
            loss = mse + reg

            if args.z_consistency_lambda > 0 and idx > 0:
                prev_mid = recon_slices[-1].to(device)
                curr_mid = out[1, 0]
                loss += args.z_consistency_lambda * compute_tv((curr_mid - prev_mid).unsqueeze(0))

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            inp = out.detach()

        recon_slices.append(inp[1, 0].cpu())

    recon = torch.stack(recon_slices, dim=0)
    return recon, losses


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)

    files = load_and_sort_images(args.input_pattern)
    radiographs = stack_and_average(files, args.num_views)

    scu = load_scu(args.scu_npy) if args.scu_npy.exists() else radiographs
    proj = compute_projections(scu)

    angles = torch.tensor([int(str(p).split('theta')[-1].split('deg')[0]) for p in files], dtype=torch.float32)

    net = UNet(1, 1).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    recon, losses = reconstruct(proj, angles, net, optimizer, args)

    results = {
        'recon': recon,
        'losses': losses,
        'theta': angles,
        'mac': args.mac,
        'tv_lambda': args.tv_lambda,
        'z_consistency_lambda': args.z_consistency_lambda,
        'random_seed': args.random_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)
    print(f"Reconstruction saved to {args.output}")


if __name__ == "__main__":
    main()
