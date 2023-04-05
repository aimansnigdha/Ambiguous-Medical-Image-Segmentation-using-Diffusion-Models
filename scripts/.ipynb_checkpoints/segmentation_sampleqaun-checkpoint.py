"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
b_size = 5

import argparse
import os
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


def iou_score_cal(output, target):
    smooth = 1e-5
    # pdb.set_trace()
    if th.is_tensor(output):
        output = th.sigmoid(output).data.cpu().numpy()
    if th.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou


def dice_coef_cal(output, target):
    smooth = 1e-5

    output = th.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def generalized_energy_distance_iou(predictions, masks):
    # pdb.set_trace()
    # num_samples = len(predictions)
    n = predictions.shape[0]
    m = masks.shape[0]
    d1=0
    d2=0
    d3=0
    for i in range(n):
        for j in range(m):
            # pdb.set_trace()
            d1 = d1 + (1-iou_score_cal(predictions[i],masks[j]))

    for i in range(n):
        for j in range(n):
            d2 = d2 + (1-iou_score_cal(predictions[i],predictions[j]))

    for i in range(m):
        for j in range(m):
            d3 = d3 + (1-iou_score_cal(masks[i],masks[j]))

    d1 = d1 *(2/(n*m))
    d2 = d2* (1/(n*n))
    d3 = d3* (1/(m*m))

    ed = d1 - d2 - d3

    scores = iou_score_cal(predictions[0],masks[0])

    return ed, scores

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    iou_score = 0
    ged_score = 0

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    ds = LIDCDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
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
        viz.image(visualize(label[0]), opts=dict(caption="gt"))
        #viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        #viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        viz.image(visualize(img[0, 4, ...]), opts=dict(caption="noise"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        predictions = []
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
            
            predictions.append(s)
            
            
            
            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            th.save(s, './results/'+str(slice_ID)+'_output'+str(i)) #save the generated mask
        
        predictions = th.cat(predictions, 0)
        # pdb.set_trace()
        iou_score_iter, ged_score_iter = generalized_energy_distance_iou(predictions, label)
        iou_score += iou_score_iter
        ged_score += ged_score_iter
    
    
    
    ged = ged_score/len(data)
    iou = iou_score/len(data)
    
    print("IOU Avg :{}, GED Avg :{}".format(iou,ged))
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
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
