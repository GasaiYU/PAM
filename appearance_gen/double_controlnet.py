import argparse
import logging
import math
import os
import random
import shutil

from src.flux.model import DoubleControl 
import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from src.dataset.hoivideodataset import get_train_set, get_valid_set
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from src.flux.xflux_pipeline import DoubleControlSampler
from diffusers.optimization import get_scheduler
from einops import rearrange
from src.flux.sampling import prepare
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_controlnet_extend,load_t5)
import wandb

logger = get_logger(__name__, log_level="INFO")

def debug_save_image(image, filename):
    image = (image + 1.0) * 127.5
    image = image.clamp(0, 255).to(torch.uint8)
    image = image[0]
    image = image.cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray(image)
    image.save(filename)

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        required=True,
        help="path to data root",
    )
    parser.add_argument(
        "--train_caption_column",
        type=str,
        default=None,
        required=True,
        help="path to train caption column",
    )
    parser.add_argument(
        "--train_video_column",
        type=str,
        default=None,
        required=True,
        help="path to train video column",
    )
    parser.add_argument(
        "--train_depth_column",
        type=str,
        default=None,
        required=True,
        help="path to train depth column",
    )
    parser.add_argument(
        "--train_label_column",
        type=str,
        default=None,
        required=True,
        help="path to train label column",
    )
    parser.add_argument(
        "--valid_caption_column",
        type=str,
        default=None,
        required=True,
        help="path to valid caption column",
    )
    parser.add_argument(
        "--valid_video_column",
        type=str,
        default=None,
        required=True,
        help="path to valid video column",
    )   
    parser.add_argument(
        "--valid_depth_column",
        type=str,
        default=None,
        required=True,
        help="path to valid depth column",
    )
    parser.add_argument(
        "--valid_label_column",
        type=str,
        default=None,
        required=True,
        help="path to valid label column",
    )
    args = parser.parse_args()
    return args.config

def get_loss(gt, pred, front_mask, back_weight=0.3):
    weight_vector = torch.ones_like(gt, dtype=gt.dtype, device=gt.device)
    weight_vector[~front_mask] = back_weight
    
    loss = F.mse_loss(gt, pred, reduction="none")
    loss = loss * weight_vector
    return loss.mean() 

def get_down_mask(valid_up_mask, downsample_factor):
    vae_scale_factor = downsample_factor
    valid_mask_down = torch.max_pool2d(
                            valid_up_mask.float(), 
                            vae_scale_factor,
                            vae_scale_factor
                        ).bool()
    valid_mask_down = rearrange(valid_mask_down, "b ch (h ph) (w pw) -> b (h w) (ch ph pw)", ph=2, pw=2)
    valid_mask_down = valid_mask_down.repeat((1, 1, 16))
    return valid_mask_down

def main():
    wandb.login(key="You wandb key")
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    change_world_size = True
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print("DEVICE", accelerator.device)
    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)
    dit.to(accelerator.device)
    
    controlnet1 = load_controlnet_extend(name=args.model_name, device=accelerator.device, transformer=dit,condition_in_channels=6,depth=2)
    controlnet2 = load_controlnet_extend(name=args.model_name, device=accelerator.device, transformer=dit,condition_in_channels=6,depth=2)
    controlnet1.train()
    controlnet2.train()
    controlnet = DoubleControl(controlnet1, controlnet2)
    controlnet.train()
    optimizer_cls = torch.optim.AdamW

    print(sum([p.numel() for p in controlnet1.parameters() if p.requires_grad]) * 2 / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in controlnet1.parameters() if p.requires_grad]+[p for p in controlnet2.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = get_train_set(args.data_root, args.train_caption_column, args.train_video_column, args.train_depth_column, args.train_label_column)
    validset = get_valid_set(args.data_root, args.valid_caption_column, args.valid_video_column, args.valid_depth_column, args.valid_label_column)
    
    if accelerator.is_main_process:
        valid_data=[validset[4], validset[14], validset[24], validset[34], validset[44], dataset[11]]

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, deepcopy(train_dataloader), lr_scheduler
    )
    
    weight_dtype=torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        
    if accelerator.is_main_process:
        wandb.init(project='ctrlhoi',name=args.exp_name)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif change_world_size:
            accelerator.print(f"Resuming from checkpoint {path}")
            state_dict= torch.load(os.path.join(args.output_dir, path, "controlnet.bin"))
            state_dict = {"module." + k: v for k, v in state_dict.items()}
            controlnet.load_state_dict(state_dict)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    global_step = initial_global_step
    if accelerator.is_main_process:
        pipe = DoubleControlSampler(clip, t5, vae, dit, controlnet1, controlnet2, accelerator. device)
                
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                img, mask, prompts, depth, normal, hand, seg_mask = batch['video'].to(accelerator.device), batch['mask'].to(accelerator.device), batch['prompt'],batch['depth_map'].to(accelerator.device),batch['normal_map'].to(accelerator.device),batch['hand_keypoints'].to(accelerator.device),batch['seg_mask'].to(accelerator.device)

                ii = 1.0
                jj = 1.0
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    depth = depth.to(torch.float32)
                    normal = normal.to(torch.float32)
                    hand = hand.to(torch.float32)
                    seg_mask = seg_mask.to(torch.float32)
                    
                bs = img.shape[0]
                t = torch.tensor([random.choice(timesteps) for _ in range(bs)]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                
                x_t = (1 - t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2])) * x_1 + t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2]) * x_0
                physic_conditions = torch.cat([depth, normal], dim=1)
                segmantic_conditions = torch.cat([hand, seg_mask], dim=1)

                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)
                inp['img'] = x_t
                
                block_res_samples_1 = controlnet1(
                    img=inp['img'].to(weight_dtype),
                    y=inp['y'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    controlnet_cond=physic_conditions.to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )
                
                block_res_samples_2 = controlnet2(
                    img=inp['img'].to(weight_dtype),
                    y=inp['y'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    controlnet_cond=segmantic_conditions.to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )
                
                # Predict the noise residual and compute loss
                model_pred = dit(
                    img=inp['img'].to(weight_dtype),
                    y=inp['y'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    block_controlnet_hidden_states=[
                        sample1.to(dtype=weight_dtype)*ii+sample2.to(dtype=weight_dtype)*jj for sample1,sample2 in zip(block_res_samples_1,block_res_samples_2)
                    ],
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet1.parameters(), args.max_grad_norm)
                    accelerator.clip_grad_norm_(controlnet2.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    wandb.log(data={"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    accelerator.save_state(save_path)
                    unwrapped_model = accelerator.unwrap_model(controlnet)
                    torch.save(unwrapped_model.state_dict(), os.path.join(save_path, 'controlnet.bin'))
                    logger.info(f"Saved state to {save_path}")
                    
                if global_step % args.valid_iter == 1 and accelerator.is_main_process:
                    with torch.no_grad():
                        for i,valid_sample in enumerate(valid_data):
                            image1=pipe.infer_data(valid_sample,seed=123,dtype=torch.float32,control_gs=(1.0,1.0,0,0))
                            wandb.log({f'valid/image_physical_{i}':wandb.Image(Image.fromarray(image1))},step=global_step)
                            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
