"""
Train a diffusion model on images.
"""

import argparse
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datetime import datetime
import torch.nn as nn
from improved_diffusion import dist_util, logger
from improved_diffusion.kernel_datasets import load_kernel
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

def get_network_description(network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = 'Kernel Prior'
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def main():
    args = create_argparser().parse_args()
    os.environ['OPENAI_LOGDIR'] = 'experiments/logs_{}_{}_{}'.format(args.diffusion_steps, args.lr, 'KL' if args.use_kl else 'MSE')

    mkdir_and_rename(os.environ['OPENAI_LOGDIR'])
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    print(args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    s, n = get_network_description(model)

    logger.log("Loaded model {} with {} parameters".format(s, n))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_kernel(
        data_dir=args.data_dir,
        data_type='npy',
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=20,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
