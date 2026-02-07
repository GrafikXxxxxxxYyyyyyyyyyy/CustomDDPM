# /usr/bin/env python3
import os
import math
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from accelerate import Accelerator
from torchvision import transforms
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration
from safetensors.torch import save_model, load_model

from src.eval import InferencePipeline
from src.models import UNetDenoisingModel



def parse_args():
    parser = argparse.ArgumentParser(description="Example of a training DDPM script")

    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data")
    parser.add_argument("--output_dir", type=str, default="ddpm-model", help="The output directory")
    parser.add_argument("--resolution", type=int, default=128, help="The resolution for input images")
    parser.add_argument("--center_crop", type=bool, default=True, help="Whether to center crop the input images to the resolution")
    parser.add_argument("--random_flip", type=bool, default=False, help="Whether to randomly flip images horizontally")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="The number of subprocesses to use for data loading")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Whether to use [tensorboard] or [wandb]")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard logging directory")
    parser.add_argument("--save_model_every", type=int, default=1, help="Save model every N epochs")

    args = parser.parse_args()

    return args



def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    # 1. Initialize the model
    model = UNetDenoisingModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        up_block_types=(
            "AttnUpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
            "UpBlock",
        ),
    )
    print("Model initialized!")


    # 2. Initialize the scheduler
    noise_scheduler = DDPMScheduler()
    print("Scheduler initialized!")


    # 3. Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    print("Optimizer initialized!")


    # 4. Preprocessing the datasets and DataLoaders creation.
    dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, split="train")

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(img.convert("RGB")) for img in examples["image"]]
        return {"input": images}

    dataset.set_transform(transform_images)    

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    print("Train dataloader initialized!")


    # 5. Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    print("LR scheduler initialized!")


    # 6. Prepare everything with our `accelerator`
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    print("Everything prepared!")


    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        print(f"***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size = {total_batch_size}")
        print(f"  Total optimization steps = {max_train_steps}")


    # 7. Train
    global_step = 0
    first_epoch = 0
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(weight_dtype)

            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps).sample

                loss = F.mse_loss(model_output.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_model_every == 0 or epoch == args.num_epochs - 1:
                denoiser = accelerator.unwrap_model(model)
                
                pipeline = InferencePipeline(
                    denoiser=denoiser,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)

                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                ).images

                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)

                # Save model
                model_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.safetensors")
                save_model(denoiser, model_path)
                print(f"Model saved to {model_path}")

    accelerator.end_training()
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
