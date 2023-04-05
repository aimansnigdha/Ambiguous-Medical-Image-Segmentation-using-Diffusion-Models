"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import torch
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8097)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.lidcloader import LIDCDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    args = create_argparser().parse_args()
    
    world_size = args.ngpu

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    os.environ["MASTER_PORT"] = str(1028)
    
    torch.distributed.init_process_group(
    'gloo',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,)
    
    logger.configure()

    logger.log("creating model and diffusion...")
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
    
    ds = LIDCDataset(args.data_dir, test_flag=True)
    
        
    sampler = torch.utils.data.distributed.DistributedSampler(
    ds,
    num_replicas=args.ngpu,
    rank=args.local_rank,
)
  
    
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        sampler = sampler,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, label, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        slice_ID=path[0].split("/", -1)[3]
        
        #viz.image(visualize(img[0]), opts=dict(caption="img input0"))
        viz.image(visualize(img[0,0,...]), opts=dict(caption="image input"))
        viz.image(visualize(label), opts=dict(caption="gt"))
        #viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        #viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        viz.image(visualize(img[0, 4, ...]), opts=dict(caption="noise"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            th.save(s, './results/'+str(slice_ID)+'_output'+str(i)) #save the generated mask

def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=2)
    parser.add_argument('--ngpu', type=int, default=2)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
