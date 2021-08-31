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

def get_network_description(network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = 'Kernel Prior'
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n


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
    else:
        model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    model.to(dist_util.dev())
    model.eval()

    s, n = get_network_description(model)

    logger.log("Loaded model {} with {} parameters".format(s, n))

    # print(args.num_samples)#default 10000

    args.num_samples = 10

    logger.log("sampling...")
    
    all_images = []
    all_labels = []
    t0 = time.time()
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    t1 = time.time()
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)


    samples = post_process(torch.from_numpy(arr))
    print(samples.shape)
    filename = 'generated_samples' + '.png'

    # rescale the maximum value to 1 for visualization, from
    samples_max, _ = samples.flatten(2).max(2, keepdim=True)
    samples = samples / samples_max.unsqueeze(3)
    save_image(samples, os.path.join(logger.get_dir(), filename), nrow=10, normalize=True)

    dist.barrier()
    logger.log("sampling complete")
    logger.log("sampling time: {}".format(t1-t0))

def post_process(x):
        # inverse process of pre_process in dataloader
        x = x.view(x.shape[0], 1, int(16), int(16))
        x = ((torch.sigmoid(x) - 1e-06) / (1 - 2 * 1e-06))
        x = x * 0.16908
        return x


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
