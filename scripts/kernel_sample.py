"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['OPENAI_LOGDIR'] = 'logs_test_2e-5'

if not os.path.exists(os.environ['OPENAI_LOGDIR']):
    os.mkdir(os.environ['OPENAI_LOGDIR'])

import torch
import numpy as np
import torch as th

import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import save_image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if torch.cuda.is_available():
        model.load_state_dict(
        dist_util.load_state_dict(args.model_path)
    )
    
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    
    # kernel_code = th.randn((1, 1, 16, 16), device=dist_util.dev())
    kernel_code = th.load('tensor_2.pt').to(dist_util.dev())
    # th.save(kernel_code, 'tensor_2.pt')

    all_images = []

    t0 = time.time()
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            kernel_code,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample.clamp(0, 0.15)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    t1 = time.time()
    arr = np.concatenate(all_images, axis=0)
    arr = th.tensor(arr)

    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr)

    filename = 'generated_samples' + '.png'

    # rescale the maximum value to 1 for visualization, from
    samples_max, _ = arr.flatten(2).max(2, keepdim=True)
    samples = arr / samples_max.unsqueeze(3)
    save_image(samples, os.path.join(logger.get_dir(), filename), nrow=16, normalize=True)

    dist.barrier()
    logger.log("sampling complete")
    logger.log("sampling time: {}".format(t1-t0))


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
