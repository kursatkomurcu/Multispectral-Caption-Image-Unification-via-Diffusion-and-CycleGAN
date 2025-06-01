import os
import argparse
import torch
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file as safe_load

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using a fine-tuned Stable Diffusion UNet"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default='/your/sd/checkpoint-step', required=True,
        help="Directory containing the fine-tuned `model.safetensors` UNet checkpoint"
    )
    parser.add_argument(
        "--base_model_id", type=str, default="stabilityai/stable-diffusion-2-1-base",
        help="Pretrained Stable Diffusion base model identifier"
    )
    parser.add_argument(
        "--csv_path", type=str, default='/your/updated_annotations.csv', required=True,
        help="Path to CSV file with columns `generated_captions` and `filepath`"
    )
    parser.add_argument(
        "--output_dir", type=str, default='/your/save/folder/for/generated_images', required=True,
        help="Directory where generated images will be saved"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run the pipeline on (e.g. 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of inference steps per generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the base UNet architecture and apply fine-tuned weights
    base_unet = UNet2DConditionModel.from_pretrained(
        args.base_model_id,
        subfolder="unet",
        torch_dtype=torch.float16
    )
    state_dict = safe_load(os.path.join(args.checkpoint_dir, "model.safetensors"))
    base_unet.load_state_dict(state_dict)
    unet = base_unet

    # Load VAE, text encoder, tokenizer, and scheduler
    vae = AutoencoderKL.from_pretrained(
        args.base_model_id,
        subfolder="vae",
        torch_dtype=torch.float16
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.base_model_id,
        subfolder="tokenizer"
    )
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.base_model_id,
        subfolder="scheduler"
    )

    # Disable safety checker and feature extractor
    safety_checker = None
    feature_extractor = None

    # Build the pipeline and move to chosen device
    pipe = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor
    ).to(args.device)

    # Read prompts and filepaths from CSV
    df = pd.read_csv(args.csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        prompt = row["generated_captions"]
        original_path = row["filepath"]
        filename = os.path.basename(original_path)
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_generated.png"
        output_path = os.path.join(args.output_dir, new_filename)

        # Skip if already generated
        if os.path.exists(output_path):
            print(f"🚫 Row {idx}: {output_path} already exists, skipping.")
            continue

        # Generate and save image
        result = pipe(prompt, num_inference_steps=args.num_steps)
        image = result.images[0]
        image.save(output_path)

        print(f"✅ Row {idx}: Generated and saved image to {output_path} (prompt: '{prompt}')")

if __name__ == "__main__":
    main()

