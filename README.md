# Ambiguous Medical Image Segmentation using Diffusion Models

We provide the official Pytorch implementation of the paper [Ambiguous Medical Image Segmentation using Diffusion Models](https://aimansnigdha.github.io/cimd/)


The implementation of diffusion model segmentation model presented in the paper is based on [Diffusion Models for Implicit Image Segmentation Ensembles](https://arxiv.org/abs/2112.03145). The Gaussian encoders are from the Pytroch implementation of [Probabilistic Unet](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).

## Paper Abstract


Collective insights from a group of experts have always proven to outperform an individual's best diagnostic for clinical tasks. For the task of medical image segmentation, existing research on AI-based alternatives focuses more on developing models that can imitate the best individual rather than harnessing the power of expert groups. In this paper, we introduce a single diffusion model-based approach that produces multiple plausible outputs by learning a distribution over group insights.  Our proposed model generates a distribution of segmentation masks by leveraging the inherent stochastic sampling process of diffusion using only minimal additional learning. We demonstrate on three different medical image modalities- CT, ultrasound, and MRI that our model is capable of producing several possible variants while capturing the frequencies of their occurrences. Comprehensive results show that our proposed approach outperforms existing state-of-the-art ambiguous segmentation networks in terms of accuracy while preserving naturally occurring variation. We also propose a new metric to evaluate the diversity as well as the accuracy of segmentation predictions that aligns with the interest of clinical practice of collective insights.


## Data

We evaluated our method on the [LIDC dataset](https://wiki.cancerimagingarchive.net/).
For our dataloader, the expert annotations as well as the original images need to be stored in the following structure:

```
data
└───training
│   └───0
│       │   image_0.jpg
│       │   label0_.jpg
│       │   label1_.jpg
│       │   label2_.jpg
│       │   label3_.jpg
│   └───1
│       │  ...
└───testing
│   └───3
│       │   image_3.jpg
│       │   label0_.jpg
│       │   label1_.jpg
│       │   label2_.jpg
│       │   label3_.jpg
│   └───4
│       │  ...

```
An example can be seen in folder *data*.

## Usage

We set the flags as follows:

```

MODEL_FLAGS="--image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 20"

```
To train the ambiguous segmentation model, run

```
!CUDA_VISIBLE_DEVICES=0,4 python -m torch.distributed.launch --nproc_per_node=2 scripts/segmentation_train.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
For sampling an ensemble of 4 segmentation masks with the DDPM approach, run:

```
python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=4 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated segmentation masks will be stored in the *results* folder. A visualization can be done using [Visdom](https://github.com/fossasia/visdom). If you encounter high frequency noise, you can use noise filters such as [median blur](https://www.tutorialspoint.com/opencv/opencv_median_blur.htm) in post-processing step.

## Reference Codes

1. [Diffusion Models for Implicit Image Segmentation Ensembles](https://github.com/JuliaWolleb/Diffusion-based-Segmentation). 
2. [Probabilistic Unet](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).
