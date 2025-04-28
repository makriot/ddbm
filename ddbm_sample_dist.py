"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist
from tqdm import tqdm

from ddbm import dist_util, logger
from ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.random_util import get_generator
from ddbm.karras_diffusion import karras_sample, forward_sample

# from datasets import load_data
from pix2pix_utils import load_data

from pathlib import Path

from PIL import Image



def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def sample(args):

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:

        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    

    all_images = []
    

    # all_dataloaders = load_data(
    #     data_dir=args.data_dir,
    #     dataset=args.dataset,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     include_test=True,
    #     seed=args.seed,
    #     num_workers=args.num_workers,
    # )

    all_dataloaders = load_data(data_dir=args.data_dir, 
                                dataset=args.dataset, 
                                batch_size=args.batch_size, 
                                image_size=args.image_size,
                                seed=args.seed,
                                num_workers=args.num_workers,
                                order=args.order)


    de_normalize_a, de_normalize_b = all_dataloaders[-2:]
    all_dataloaders = all_dataloaders[:-2]
    if args.split =='train':
        dataloader = all_dataloaders[1]
    elif args.split == 'test':
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError
    # args.num_samples = len(dataloader.dataset)


    print("Sampling...")
    cnt_samples = 0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if cnt_samples >= args.num_samples + args.batch_size:
            break
        cnt_samples += args.batch_size
        if i == len(dataloader):
            break
        
        x0_image = data[0]
        
        x0 = x0_image.to(dist_util.dev()) * 2 -1
        
        y0_image = data[1].to(dist_util.dev())
        
        y0 = y0_image.to(dist_util.dev()) * 2 - 1
        model_kwargs = {'xT': y0}
        index = data[2].to(dist_util.dev())

        sample, path, nfe = karras_sample(
            diffusion,
            model,
            y0,
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=diffusion.sigma_min,
            sigma_max=diffusion.sigma_max,
            churn_step_ratio=args.churn_step_ratio,
            rho=args.rho,
            guidance=args.guidance
        )
        # sample = (sample + 1) / 2
        sample = de_normalize_a(sample)
        sample = sample * 255.0
        sample = sample.clamp(0, 255).to(th.uint8)
        # sample = (sample * 255).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous().to(dist_util.dev())

        gathered_samples = [th.zeros(sample.shape, dtype=sample.dtype, device=sample.device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if index is not None:
            gathered_index = [th.zeros(index.shape, dtype=index.dtype, device=index.device) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_index, index)
            gathered_samples = th.cat(gathered_samples)
            gathered_index = th.cat(gathered_index)
            gathered_samples = gathered_samples[th.argsort(gathered_index)]
        else:
            gathered_samples = th.cat(gathered_samples)

        num_display = min(32, sample.shape[0])
        if i == 0 and dist.get_rank() == 0:
            vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
            if x0 is not None:
                vutils.save_image(x0_image[:num_display], f'{sample_dir}/x_{i}.png',nrow=int(np.sqrt(num_display)))
            vutils.save_image(y0_image[:num_display]/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
            
            
        all_images.append(gathered_samples.detach().cpu().numpy())
        

    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_data=0.5,
        sigma_min=0.0001,
        sigma_max=1.0,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0.5**2 / 2,  # default: 0
        image_size=64,
        in_channels=3,
        num_channels=64,
        num_res_blocks=1,
        num_heads=4,  # will be ignored if num_head_channels is set
        num_heads_upsample=-1,
        num_head_channels=32,  # -1 for simple attention (num_heads = [2, 4, 8, 16])
        unet_type='adm',
        attention_resolutions="16,8,4", #"32,16,8",
        channel_mult="",  # auto choose, depends on image_size
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=True,  # False with simple attention
        use_new_attention_order=False,
        attention_type='flash',
        learn_sigma=False,
        condition_mode=None,
        pred_mode='vp',
        weight_schedule="karras",
    )
    return res


def create_argparser():
    defaults = dict(
        data_dir="data", ## only used in bridge
        dataset='cityscapes',
        clip_denoised=True,
        num_samples=8,
        batch_size=8,
        sampler="heun",
        split='train',  # val
        churn_step_ratio=0.33,  # 0.0
        rho=7.0,
        steps=40,
        model_path="./workdir/1/ema_0.9999_010000.pt",
        exp="1",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
        dataset_order=False,
        order=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--devices", type=int, nargs="+", help="GPU devices (for example 0 1 2)")
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    sample(args)
