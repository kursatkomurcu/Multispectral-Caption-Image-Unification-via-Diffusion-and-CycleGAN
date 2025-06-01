import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion UNet on a custom satellite image dataset"
    )
    parser.add_argument(
        "--data_root", type=str, default='/path/to/your/data_for_diffusion', required=True,
        help="Path to the directory containing .jpg images and corresponding .txt captions"
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="Path to a checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--save_root", type=str, default='/path/to/your/output_dir', required=True,
        help="Directory where checkpoints and final model will be saved"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=20000,
        help="Save a checkpoint every N training steps"
    )
    parser.add_argument(
        "--model_id", type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Pretrained model identifier for Stable Diffusion"
    )
    return parser.parse_args()


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()  # fully load to check integrity
        return True
    except Exception as e:
        print(f"Warning: Skipping invalid or truncated image: {path} ({e})")
        return False


class SatelliteDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        root_dir: directory with .jpg images and .txt captions
        transform: torchvision transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        all_images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.image_files = []

        for image_file in all_images:
            caption_file = image_file.replace('.jpg', '.txt')
            img_path = os.path.join(root_dir, image_file)
            cap_path = os.path.join(root_dir, caption_file)

            if not os.path.exists(cap_path):
                print(f"Warning: Caption file not found, skipping: {cap_path}")
                continue

            if is_valid_image(img_path):
                self.image_files.append(image_file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_file = self.image_files[idx]
        caption_file = image_file.replace('.jpg', '.txt')
        img = Image.open(os.path.join(self.root_dir, image_file)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(os.path.join(self.root_dir, caption_file), 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return {"image": img, "caption": caption}


class ResumeSampler(Sampler):
    def __init__(self, data_source: Dataset, start_index: int = 0):
        self.data_source = data_source
        self.start_index = start_index

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        return iter(indices[self.start_index:] + indices[:self.start_index])

    def __len__(self):
        return len(self.data_source)


def main():
    args = parse_args()

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load dataset
    dataset = SatelliteDataset(root_dir=args.data_root, transform=transform)

    # Determine starting global_step from checkpoint metadata if available
    global_step = 0
    if args.resume_checkpoint:
        meta_path = os.path.join(args.resume_checkpoint, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as mf:
                meta = json.load(mf)
            global_step = meta.get('global_step', 0)
            print(f"Resuming training from step {global_step}")
        else:
            print("No metadata found in checkpoint, starting from step 0")

    # Compute resume index for sampler
    stored_index = (global_step * args.batch_size) % len(dataset)
    sampler = ResumeSampler(dataset, start_index=stored_index)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4
    )

    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae"
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet"
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # Freeze non-finetuned parts
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Prepare optimizer and accelerator
    accelerator = Accelerator()
    optimizer = optim.AdamW(unet.parameters(), lr=args.learning_rate)
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )

    # Training loop
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            images = batch["image"].to(device)
            captions = batch["caption"]

            inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(device)

            with torch.no_grad():
                encoder_states = text_encoder(input_ids)[0]

            latents = vae.encode(images).latent_dist.sample()
            latents *= 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, encoder_states).sample

            loss = nn.MSELoss()(noise_pred, noise)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            global_step += 1
            if global_step % 100 == 0:
                print(f"Epoch {epoch+1}, step {global_step}, loss: {loss.item():.4f}")

            # Save checkpoint
            if global_step % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(args.save_root, 'checkpoint')
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.save_state(ckpt_dir)
                meta = {'global_step': global_step}
                with open(os.path.join(ckpt_dir, 'metadata.json'), 'w') as mf:
                    json.dump(meta, mf)
                print(f"Saved checkpoint at step {global_step} to {ckpt_dir}")

    # Final model save
    final_dir = os.path.join(args.save_root, 'fine_tuned_unet')
    os.makedirs(final_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    unet.save_pretrained(final_dir)
    print(f"Fine-tuning complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
