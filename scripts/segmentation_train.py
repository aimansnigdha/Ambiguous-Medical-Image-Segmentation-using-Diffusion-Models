"""
Train a diffusion model on images.
"""
import sys
import os
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from torch.utils.data.sampler import SubsetRandomSampler
from guided_diffusion.lidcloader import LIDCDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
import torch
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8097)

def main():
    args = create_argparser().parse_args()
    world_size = args.ngpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "6,7,8"
    
    torch.distributed.init_process_group(
    'gloo',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,)

    #dist_util.setup_dist()
    logger.configure()

    logger.log("creating model, diffusion, prior and posterior distribution...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    torch.cuda.set_device(args.local_rank)
    
    model.to(dist_util.dev())
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)
    
    prior.to(dist_util.dev())
    prior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prior)
    prior = torch.nn.parallel.DistributedDataParallel(
    prior,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)
        
    posterior.to(dist_util.dev())
    
    posterior.to(dist_util.dev())
    posterior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(posterior)
    posterior = torch.nn.parallel.DistributedDataParallel(
    posterior,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)
    
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")
    ds = LIDCDataset(args.data_dir, test_flag=False)

        
    sampler = torch.utils.data.distributed.DistributedSampler(
    ds,
    num_replicas=args.ngpu,
    rank=args.local_rank,
)
    
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        prior = prior,
        posterior = posterior,
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
        data_dir="./data/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=4)
    parser.add_argument('--ngpu', type=int, default=3)
   
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
