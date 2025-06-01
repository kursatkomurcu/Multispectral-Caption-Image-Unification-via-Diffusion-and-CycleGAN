import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import tifffile as tiff
from safetensors.torch import save_file
from torch.cuda.amp import GradScaler, autocast
from threading import Thread

from model import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN with Sentinel and RGB data")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/your/base/dir",
        help="Base directory for data and outputs"
    )
    parser.add_argument(
        "--band_folder",
        type=str,
        default="/your/band/folder", 
        help="Relative path (under base_dir) to multispectral band folder"
    )
    parser.add_argument(
        "--root_A",
        type=str,
        default="/your/rgb/folder",
        help="Relative path (under base_dir) to RGB folder"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Relative path (under base_dir) to save model checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Relative path (under base_dir) for generated outputs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=124,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for optimizers"
    )
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=10.0,
        help="Weight for cycle consistency loss"
    )
    parser.add_argument(
        "--lambda_sam",
        type=float,
        default=1.0,
        help="Weight for SAM loss"
    )
    parser.add_argument(
        "--lambda_hist",
        type=float,
        default=1.0,
        help="Weight for histogram loss"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor for DataLoader"
    )
    return parser.parse_args()

def async_save(path, data):
    torch.save(data, path)

def main():
    args = parse_args()

    # Device & Performance Flags 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32   = True

    # Paths
    base_dir    = args.base_dir
    band_folder = os.path.join(base_dir, args.band_folder)
    root_A      = os.path.join(base_dir, args.root_A)
    models_dir  = os.path.join(base_dir, args.models_dir)
    output_dir  = os.path.join(base_dir, args.output_dir, os.path.basename(band_folder))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters 
    batch_size    = args.batch_size
    n_epochs      = args.n_epochs
    lr            = args.lr
    lambda_cycle  = args.lambda_cycle
    lambda_sam    = args.lambda_sam
    lambda_hist   = args.lambda_hist

    sam_loss_fn = SAMLoss()

    # Channels 
    input_nc_A = 3    # RGB
    input_nc_B = 13   # Sentinel bands

    # Transforms & Dataset 
    transform_A = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    transform_B = transforms.Compose([
        SentinelToTensor(),
        transforms.Normalize((0.5,)*13, (0.5,)*13)
    ])

    dataset = ImageDataset(
        root_A=root_A,
        root_B=band_folder,
        transform_A=transform_A,
        transform_B=transform_B,
        sample_percentage=1.0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    print("Batches per epoch:", len(dataloader))

    # Models
    G     = Generator(input_nc_A, output_nc=10, n_residual_blocks=9).to(device)
    F_net = Generator(input_nc_B, output_nc=input_nc_A, n_residual_blocks=9).to(device)
    D_A   = Discriminator(input_nc_A).to(device)
    D_B   = Discriminator(input_nc_B).to(device)

    # Optimizers & Criterion 
    criterion     = nn.L1Loss()
    optimizer_G   = optim.Adam(
        list(G.parameters()) + list(F_net.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    scaler = GradScaler()

    # Training Loop 
    for epoch in range(1, n_epochs+1):
        for i, batch in enumerate(dataloader, 1):
            real_A = batch['A'].to(device, non_blocking=True)
            real_B = batch['B'].to(device, non_blocking=True)

            # Generator Step 
            optimizer_G.zero_grad()
            with autocast(dtype=torch.bfloat16):
                fake_extra = G(real_A)
                fake_B = torch.empty(real_A.size(0), input_nc_B, 64, 64, device=device)
                fake_B[:, 0]   = fake_extra[:, 0]
                fake_B[:, 1:4] = real_A
                fake_B[:, 4:]  = fake_extra[:, 1:]

                loss_gan_B = criterion(D_B(fake_B), torch.ones_like(D_B(fake_B)))
                fake_A     = F_net(real_B)
                loss_gan_A = criterion(D_A(fake_A), torch.ones_like(D_A(fake_A)))

                recov_A     = F_net(fake_B)
                fake_extra2 = G(fake_A)
                recov_B     = torch.empty_like(real_B)
                recov_B[:, 0]   = fake_extra2[:, 0]
                recov_B[:, 1:4] = fake_A
                recov_B[:, 4:]  = fake_extra2[:, 1:]

                # SAM loss (works in bfloat16)
                sam_A = sam_loss_fn(recov_A, real_A)
                sam_B = sam_loss_fn(recov_B, real_B)

            # histogram_loss not supported in bfloat16, compute in float32
            recov_A_fp = recov_A.float()
            recov_B_fp = recov_B.float()
            real_A_fp  = real_A.float()
            real_B_fp  = real_B.float()
            hist_A     = histogram_loss(recov_A_fp, real_A_fp)
            hist_B     = histogram_loss(recov_B_fp, real_B_fp)

            loss_sam   = lambda_sam  * (sam_A + sam_B)
            loss_hist  = lambda_hist * (hist_A + hist_B)
            loss_cycle = lambda_cycle * (loss_sam + loss_hist)

            loss_G = loss_gan_B + loss_gan_A + loss_cycle

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Discriminator A Step 
            optimizer_D_A.zero_grad()
            with autocast(dtype=torch.bfloat16):
                real_loss = criterion(D_A(real_A), torch.ones_like(D_A(real_A)))
                fake_loss = criterion(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
                loss_D_A  = 0.5 * (real_loss + fake_loss)
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
            scaler.update()

            # Discriminator B Step 
            optimizer_D_B.zero_grad()
            with autocast(dtype=torch.bfloat16):
                real_loss = criterion(D_B(real_B), torch.ones_like(D_B(real_B)))
                fake_loss = criterion(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
                loss_D_B  = 0.5 * (real_loss + fake_loss)
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
            scaler.update()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}] Batch [{i}/{len(dataloader)}] "
                    f"L_G: {loss_G.item():.4f} "
                    f"(advA: {loss_gan_A.item():.4f}, advB: {loss_gan_B.item():.4f}, "
                    f"sam: {(sam_A+sam_B).item():.4f}, hist: {(hist_A+hist_B).item():.4f}), "
                    f"L_D_A: {loss_D_A.item():.4f} "
                    f"L_D_B: {loss_D_B.item():.4f}"
                )

        # Async checkpoint every epoch 
        ckpt = {
            "epoch": epoch,
            "G_state_dict": G.state_dict(),
            "F_state_dict": F_net.state_dict(),
            "optG_state_dict": optimizer_G.state_dict(),
            "optDA_state_dict": optimizer_D_A.state_dict(),
            "optDB_state_dict": optimizer_D_B.state_dict()
        }
        path = os.path.join(models_dir, f"ckpt_epoch_{epoch}.pth")
        Thread(target=async_save, args=(path, ckpt)).start()

    # Final model save 
    save_file(G.state_dict(), os.path.join(models_dir, "G_model.safetensors"))
    save_file(F_net.state_dict(), os.path.join(models_dir, "F_model.safetensors"))

if __name__ == "__main__":
    main()
