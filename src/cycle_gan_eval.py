import argparse
import os
import glob
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import tifffile as tiff
from PIL import Image
from skimage.transform import resize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import *
from safetensors.torch import safe_open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def extract_natural_composite(ms_image):
    """
    ms_image: NumPy array, shape (H, W, 13) – multispectral (0-1 range).
    3 channel image is returned (H, W, 3) by taking channels [3,2,1] (e.g. R,G,B).
    """
    return ms_image[:, :, [3, 2, 1]]

def process_ms_image(path, target_size=(64, 64)):
    """
    Scales a .tif multispectral image to the range [0,1] and size (H,W,13).
    """
    ms = tiff.imread(path).astype(np.float32)
    if ms.max() > 1:
        ms /= 255.0
    if ms.shape[0] != target_size[0] or ms.shape[1] != target_size[1]:
        ms = resize(ms, (target_size[0], target_size[1], ms.shape[2]), preserve_range=True)
    return ms  # (H, W, 13)

def SAM_metric(y_true, y_pred):
    numerator = np.sum(y_true * y_pred, axis=-1)
    denom = np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1) + 1e-8
    cos_angle = np.clip(numerator / denom, -1, 1)
    angles = np.arccos(cos_angle)
    return np.mean(angles) * (180 / np.pi)

def SID_metric(y_true, y_pred):
    y_true = np.clip(y_true, 1e-8, None)
    y_pred = np.clip(y_pred, 1e-8, None)
    p = y_true / (np.sum(y_true, axis=-1, keepdims=True) + 1e-8)
    q = y_pred / (np.sum(y_pred, axis=-1, keepdims=True) + 1e-8)
    divergence = np.sum(p * np.log((p+1e-8)/(q+1e-8)) + q * np.log((q+1e-8)/(p+1e-8)), axis=-1)
    return np.mean(divergence)

def ERGAS_metric(y_true, y_pred, scale=4):
    mse_per_band = np.mean((y_true - y_pred)**2, axis=(0,1))
    mean_per_band = np.mean(y_true, axis=(0,1))
    return 100/scale * np.sqrt(np.mean(mse_per_band/(mean_per_band**2 + 1e-8)))

def MAE_metric(y_true, y_pred):
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())

def MSE_metric(y_true, y_pred):
    return mean_squared_error(y_true.flatten(), y_pred.flatten())

def get_clip_embeddings(paths, preprocess, model, batch_size=32):
    embs = []
    for i in range(0, len(paths), batch_size):
        print(f"Encoding images {i+1} to {min(i+batch_size, len(paths))} of {len(paths)}...")
        batch = [preprocess(Image.open(p).convert("RGB")).to(device) for p in paths[i : i + batch_size]]
        bt = torch.stack(batch)
        with torch.no_grad(), torch.cuda.amp.autocast():
            e = model.encode_image(bt)
            e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)

def median_heuristic_sigma(x, y):
    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        sample = z[torch.randperm(len(z))][: min(len(z), 1000)]
        d2 = torch.cdist(sample, sample, p=2).reshape(-1)
        return torch.median(d2).item()

def compute_mmd(x, y, sigma=None):
    m, n = x.size(0), y.size(0)
    if sigma is None:
        sigma = median_heuristic_sigma(x, y)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    Kxx = torch.exp(-xx / (2 * sigma ** 2))
    Kyy = torch.exp(-yy / (2 * sigma ** 2))
    Kxy = torch.exp(-xy / (2 * sigma ** 2))
    term_x = (Kxx.sum() - torch.trace(Kxx)) / (m * (m - 1))
    term_y = (Kyy.sum() - torch.trace(Kyy)) / (n * (n - 1))
    term_xy = 2 * Kxy.sum() / (m * n)
    return term_x + term_y - term_xy

def parse_args():
    parser = argparse.ArgumentParser(description="Full evaluation of generated vs. real multispectral data")
    parser.add_argument(
        "--rgb_dir",
        type=str,
        default='/your/rgb/path',
        required=True,
        help="Directory containing generated RGB images"
    )
    parser.add_argument(
        "--ms_dir",
        type=str,
        default='/your/multispectral/path',
        required=True,
        help="Directory containing ground-truth multispectral .tif images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='/your/model/path/G_model.safetensors',
        required=True,
        help="Path to the saved G_model.safetensors file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=512,
        help="Size to which RGB images are resized before random crop"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=64,
        help="Size of random crop for RGB images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rgb_dir = args.rgb_dir
    ms_dir  = args.ms_dir

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    ms_files  = sorted(glob.glob(os.path.join(ms_dir, "*.tif")))
    print("Total RGB images:", len(rgb_files))
    print("Total multispectral images:", len(ms_files))

    num_samples = len(ms_files)
    selected_rgb_files = random.sample(rgb_files, num_samples)

    # Load the model
    G = Generator(input_nc=3, output_nc=10).to(device)
    with safe_open(args.model_path, framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    G.load_state_dict(state_dict)
    G.eval().to(device)

    transform_rgb = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),  # you can comment this line if you evaluate on eurosat_rgb dataset. Because it's default size is 64x64
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    sam_list, sid_list, ergas_list = [], [], []
    mae_list, mse_list = [], []

    gen_composites = []
    real_composites = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)

        for rgb_path, ms_path in zip(selected_rgb_files[start:end], ms_files[start:end]):
            rgb_img = Image.open(rgb_path).convert("RGB")
            inp = transform_rgb(rgb_img).unsqueeze(0).to(device)

            with torch.no_grad():
                fake_extra = G(inp)
            fake_B = torch.empty(1, 13, inp.size(2), inp.size(3), device=device)
            fake_B[:,0]   = fake_extra[:,0]
            fake_B[:,1:4] = inp
            fake_B[:,4:]  = fake_extra[:,1:]
            fake_B = (fake_B + 1)/2.0  # [0,1]

            fake_np = fake_B.squeeze(0).cpu().numpy().transpose(1,2,0)
            true_ms = process_ms_image(ms_path, target_size=(args.crop_size, args.crop_size))

            gen_comp = extract_natural_composite(fake_np)
            real_comp = extract_natural_composite(true_ms)

            gen_composites.append(gen_comp)
            real_composites.append(real_comp)

            sam_list.append(SAM_metric(real_comp, gen_comp))
            sid_list.append(SID_metric(real_comp, gen_comp))
            ergas_list.append(ERGAS_metric(real_comp, gen_comp))
            mae_list.append(MAE_metric(real_comp, gen_comp))
            mse_list.append(MSE_metric(real_comp, gen_comp))

        print(f"Processed batch {batch_idx+1}/{num_batches}")

    print("Final Metrics:")
    print("SAM: ",  np.mean(sam_list), "degrees")
    print("SID: ",  np.mean(sid_list))
    print("ERGAS: ",np.mean(ergas_list))
    print("MAE: ",  np.mean(mae_list))
    print("MSE: ",  np.mean(mse_list))

    # Prepare for CLIP embeddings
    # Assumes 'model' and 'preprocess' are defined in the global scope
    real_images = []
    gen_images = []

    for idx, comp in enumerate(real_composites):
        img = (comp * 255).astype(np.uint8)
        real_images.append(Image.fromarray(img))

    for idx, comp in enumerate(gen_composites):
        img = (comp * 255).astype(np.uint8)
        gen_images.append(Image.fromarray(img))

    # Save composites temporarily to disk for CLIP embedding extraction
    os.makedirs("eval_temp/real", exist_ok=True)
    os.makedirs("eval_temp/gen", exist_ok=True)
    real_files = []
    gen_files = []

    for i, img in enumerate(real_images):
        path = f"eval_temp/real/real_{i:05d}.png"
        img.save(path)
        real_files.append(path)

    for i, img in enumerate(gen_images):
        path = f"eval_temp/gen/gen_{i:05d}.png"
        img.save(path)
        gen_files.append(path)

    print("\nComputing CLIP embeddings for real images...")
    real_emb = get_clip_embeddings(real_files, preprocess, model)

    print("\nComputing CLIP embeddings for generated images...")
    gen_emb  = get_clip_embeddings(gen_files, preprocess, model)

    sigma = median_heuristic_sigma(real_emb, gen_emb)
    print(f"Using sigma = {sigma:.4f}")

    cmmd2 = compute_mmd(real_emb, gen_emb, sigma=sigma)
    print(f"CMMD² (CLIP-based MMD): {cmmd2:.6f}")

    # Clean up temporary files if desired
    # shutil.rmtree("eval_temp")

if __name__ == "__main__":
    main()
